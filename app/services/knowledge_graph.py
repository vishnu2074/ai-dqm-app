# app/services/knowledge_graph.py

import os
import tempfile
from collections import defaultdict

import pandas as pd
from difflib import SequenceMatcher
from azure.storage.blob import BlobServiceClient

from app.agents.key_detection_agent import KeyDetectionAgent
from app.agents.llm_agent import LLMAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.parent_agent import ParentAgent
from app.agents.profiling_agent import ProfilingAgent
from app.agents.relationship_agent import RelationshipAgent
from app.models import Dataset
from app.services.dq_scores import _load_dataframe_for_dataset


class KnowledgeGraphService:

    # ------------------------------------------------------------------
    # SHARED NODE / EDGE HELPERS
    # ------------------------------------------------------------------

    def _classify_key(self, stats: dict) -> str:
        if not stats:
            return "unknown"
        row_count = stats.get("row_count", 0)
        distinct  = stats.get("distinct_count", 0)
        nulls     = stats.get("null_count", 0)
        if row_count > 0 and distinct == row_count and nulls == 0:
            return "primary"
        elif distinct > 0 and distinct < row_count:
            return "foreign"
        return "none"

    def _make_table_node(self, nid: str, label: str) -> dict:
        return {"id": nid, "label": label, "type": "table"}

    def _make_column_node(self, nid: str, col_name: str, stats: dict) -> dict:
        return {
            "id":       nid,
            "label":    col_name,
            "type":     "column",
            "stats":    stats,
            "key_type": self._classify_key(stats),
        }

    def _dedup_edges(self, edges: list) -> list:
        """Keep the highest-confidence edge per unique column pair."""
        structural = [e for e in edges if not e.get("relationship")]
        best: dict = {}
        for e in edges:
            if not e.get("relationship"):
                continue
            key = (e["source"], e["target"],
                   e.get("source_column"), e.get("target_column"))
            if key not in best or e.get("confidence", 0) > best[key].get("confidence", 0):
                best[key] = e
        return structural + list(best.values())

    def _limit_edges_per_column(self, edges: list, max_edges: int = 3) -> list:
        """Cap outgoing relationship edges per source column node."""
        count: dict = defaultdict(int)
        result = []
        for e in edges:
            if not e.get("relationship"):
                result.append(e)
                continue
            if count[e["source"]] < max_edges:
                result.append(e)
                count[e["source"]] += 1
        return result

    # ------------------------------------------------------------------
    # NORMALIZE DATAFRAME (CRITICAL FIX)
    # ------------------------------------------------------------------

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dataframe by converting ID columns to strings.
        This is CRITICAL for cross-table matching to work.
        """
        if df is None or df.empty:
            return df
        
        df_copy = df.copy()
        
        for col in df_copy.columns:
            col_lower = col.lower()
            # Check if column looks like an ID column
            is_id_column = any(id_pattern in col_lower for id_pattern in 
                              ['_id', 'id', '_key', '_code', '_no', '_num'])
            
            if is_id_column:
                # Convert to string and clean
                df_copy[col] = df_copy[col].astype(str).str.strip()
                # Remove '.0' from numeric strings if present (e.g., "1.0" -> "1")
                df_copy[col] = df_copy[col].str.replace(r'\.0$', '', regex=True)
                # Replace 'nan' with actual NaN
                df_copy[col] = df_copy[col].replace('nan', pd.NA)
                
        return df_copy

    # ------------------------------------------------------------------
    # CORE VALIDATION
    # ------------------------------------------------------------------

    def _validate_edge(self,
                   rel_agent: RelationshipAgent,
                   df1: pd.DataFrame, df2: pd.DataFrame,
                   col1: str, col2: str) -> tuple:
    
        try:
            s1_raw = df1[col1].dropna()
            s2_raw = df2[col2].dropna()
        except KeyError:
            return False, 0.0, None

        # FILTER 1 — joinability (data-driven)
        if not rel_agent.is_joinable_column(col1, s1_raw):
            return False, 0.0, None
        if not rel_agent.is_joinable_column(col2, s2_raw):
            return False, 0.0, None

        # FILTER 2 — directional confidence (adaptive)
        conf_max, cov1, cov2 = rel_agent.validate_df_directional(df1, df2, col1, col2)
        if not rel_agent.should_accept_confidence(conf_max, cov1, cov2):
            return False, conf_max, None

        # FILTER 3 — cardinality (data-driven)
        relationship = rel_agent.detect_relationship_df(df1, df2, col1, col2)
        if relationship is None:
            return False, conf_max, None

        # No additional filters - trust the data
        return True, conf_max, relationship
    # ------------------------------------------------------------------
    # LAYOUT ENGINE
    # ------------------------------------------------------------------

    def _apply_layout(self, nodes: list, edges: list, is_folder: bool) -> tuple:
        TABLE_Y        = 150 if is_folder else -50
        COLUMN_Y_START = 250 if is_folder else 80

        table_nodes = [n for n in nodes if n["type"] == "table"]
        table_count = max(len(table_nodes), 1)
        TABLE_X_GAP = max(500, 2000 // table_count)

        table_x: dict = {}
        center_offset = 0 if is_folder else (table_count - 1) * TABLE_X_GAP / 2
        for idx, n in enumerate(table_nodes):
            table_x[n["id"]] = idx * TABLE_X_GAP - center_offset

        col_groups: dict = {}
        for n in nodes:
            if n["type"] == "column":
                table_id = n["id"].split(":")[0]
                col_groups.setdefault(table_id, []).append(n["id"])

        max_cols  = max((len(v) for v in col_groups.values()), default=1)
        COL_Y_GAP = max(70, min(120, 600 // max_cols))

        aligned_y: dict = {}
        cur_y = COLUMN_Y_START
        seen_pairs: set = set()
        for e in edges:
            if not e.get("relationship"):
                continue
            key = tuple(sorted([e["source"], e["target"]]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                aligned_y[e["source"]] = cur_y
                aligned_y[e["target"]] = cur_y
                cur_y += COL_Y_GAP

        pos: dict = {}
        for n in table_nodes:
            pos[n["id"]] = (table_x[n["id"]], TABLE_Y)
        for table_id, cols in col_groups.items():
            bx = table_x.get(table_id, 0)
            for i, col_id in enumerate(sorted(cols)):
                pos[col_id] = (bx, aligned_y.get(col_id, COLUMN_Y_START + i * COL_Y_GAP))

        final_nodes = [
            {
                "id":       n["id"],
                "label":    n["label"],
                "type":     n["type"],
                "x":        float(pos.get(n["id"], (0, 0))[0]),
                "y":        float(pos.get(n["id"], (0, 0))[1]),
                "stats":    n.get("stats", {}),
                "key_type": n.get("key_type", "none"),
            }
            for n in nodes
        ]
        return final_nodes, edges

    # ------------------------------------------------------------------
    # SHARED BUILD CORE
    # ------------------------------------------------------------------

    def _build_core(self,
                    dataframes:    dict,
                    metadata_map:  dict,
                    llm_edges:     list,
                    node_id_fn,
                    label_fn,
                    rel_agent:     RelationshipAgent,
                    matching_agent: MatchingAgent,
                    llm_agent:     LLMAgent) -> tuple:
        """
        Single relationship-detection engine used by both modes.
        """
        key_agent     = KeyDetectionAgent()
        table_columns = {t: list(m.keys()) for t, m in metadata_map.items()}
        key_map       = {t: key_agent.detect_keys(m) for t, m in metadata_map.items()}
        matches       = matching_agent.match(table_columns, key_map)

        print(f"[KG] Found {len(matches)} potential matches from MatchingAgent")
        for m in matches[:5]:
            print(f"[KG] Match: {m['table1']}.{m['col1']} <-> {m['table2']}.{m['col2']} (score: {m['score']})")

        nodes_dict: dict = {}
        edges:      list = []

        def _ensure_table(tname: str):
            nid = node_id_fn(tname)
            if nid not in nodes_dict:
                nodes_dict[nid] = self._make_table_node(nid, label_fn(tname))

        def _ensure_column(tname: str, col: str, stats: dict) -> str:
            col_nid = f"{node_id_fn(tname)}:{col}"
            if col_nid not in nodes_dict:
                nodes_dict[col_nid] = self._make_column_node(col_nid, col, stats)
            has_edge = {"source": node_id_fn(tname), "target": col_nid, "type": "has_column"}
            if has_edge not in edges:
                edges.append(has_edge)
            return col_nid

        def _add_rel_edge(src_nid, tgt_nid, c1, c2, relationship, confidence, reason):
            edges.append({
                "source":        src_nid,
                "target":        tgt_nid,
                "relationship":  relationship,
                "confidence":    confidence,
                "source_column": c1,
                "target_column": c2,
                "reason":        reason,
            })

        def _already_exists(src_nid, tgt_nid) -> bool:
            return any(
                e.get("source") == src_nid and e.get("target") == tgt_nid
                for e in edges if e.get("relationship")
            )

        # ── PASS 1: LLM edges ─────────────────────────────────────────
        print(f"[KG] Processing {len(llm_edges)} LLM-suggested edges")
        for e in llm_edges:
            t1 = e.get("source", "")
            t2 = e.get("target", "")
            c1 = e.get("source_column", "")
            c2 = e.get("target_column", "")

            if t1 not in metadata_map or t2 not in metadata_map:
                continue
            if not c1 or not c2:
                continue
            if str(c1).strip() in {"", "undefined"} or str(c2).strip() in {"", "undefined"}:
                continue

            df1 = dataframes.get(t1)
            df2 = dataframes.get(t2)
            if df1 is None or df2 is None or df1.empty or df2.empty:
                continue

            keep, confidence, relationship = self._validate_edge(rel_agent, df1, df2, c1, c2)
            if not keep:
                continue

            _ensure_table(t1)
            _ensure_table(t2)
            s1 = _ensure_column(t1, c1, metadata_map.get(t1, {}).get(c1, {}))
            s2 = _ensure_column(t2, c2, metadata_map.get(t2, {}).get(c2, {}))

            reason = e.get("reason") or (
                f"{c1} and {c2} share {int(confidence*100)}% overlap "
                f"indicating a {relationship} relationship."
            )
            _add_rel_edge(s1, s2, c1, c2, relationship, confidence, reason)

        # ── PASS 2: MatchingAgent fallback ────────────────────────────
        print(f"[KG] Processing {len(matches)} MatchingAgent candidates")
        for m in matches:
            t1, t2 = m["table1"], m["table2"]
            c1, c2 = m["col1"],   m["col2"]

            df1 = dataframes.get(t1)
            df2 = dataframes.get(t2)
            if df1 is None or df2 is None or df1.empty or df2.empty:
                continue

            keep, confidence, relationship = self._validate_edge(rel_agent, df1, df2, c1, c2)
            if not keep or relationship is None:
                continue

            _ensure_table(t1)
            _ensure_table(t2)
            s1 = _ensure_column(t1, c1, metadata_map.get(t1, {}).get(c1, {}))
            s2 = _ensure_column(t2, c2, metadata_map.get(t2, {}).get(c2, {}))

            if _already_exists(s1, s2):
                continue

            reason = (
                f"{c1} and {c2} share {int(confidence*100)}% value overlap "
                f"indicating a {relationship} relationship."
            )
            _add_rel_edge(s1, s2, c1, c2, relationship, confidence, reason)
            print(f"  ✅ Added relationship: {t1}.{c1} ↔ {t2}.{c2} ({relationship})")

        print(f"[KG] Final edges count: {len(edges)}")
        print(f"[KG] Relationship edges count: {len([e for e in edges if e.get('relationship')])}")

        return nodes_dict, edges

    # ------------------------------------------------------------------
    # SELECT DATASETS MODE
    # ------------------------------------------------------------------

    def build_graph(self, dataset_ids: list, db) -> dict:
        """
        Entry point for the SELECT DATASETS tab.
        """
        file_paths: list = []
        datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()

        for ds in datasets:
            try:
                df = _load_dataframe_for_dataset(db, ds.id)
                if df is None or df.empty:
                    print(f"[KG] Dataset {ds.id} empty — skipped")
                    continue
                temp_path = f"storage/temp_{ds.id}.csv"
                df.to_csv(temp_path, index=False)
                real_name = (
                    os.path.basename(ds.physical_name)
                    if ds.physical_name else f"dataset_{ds.id}"
                )
                file_paths.append((temp_path, real_name))
            except Exception as exc:
                print(f"[KG] Error loading dataset {ds.id}: {exc}")

        if not file_paths:
            return {"nodes": [], "edges": [], "message": "No datasets loaded"}

        # path_map: real_name → temp_path
        path_map: dict = {real: temp for temp, real in file_paths}

        # Human-readable display labels
        display_map: dict = {}
        for ds in datasets:
            real = (
                os.path.basename(ds.physical_name)
                if ds.physical_name else f"dataset_{ds.id}"
            )
            display_map[real] = ds.display_name or real

        # Profile and load DataFrames (keyed by real_name)
        profiler     = ProfilingAgent()
        metadata_map = {}
        dataframes   = {}
        
        for temp_path, real_name in file_paths:
            print(f"[KG] Loading dataset: {real_name}")
            metadata_map[real_name] = profiler.profile(temp_path)
            try:
                df = pd.read_csv(temp_path)
                # 🔥 CRITICAL: Normalize dataframe (convert ID columns to string)
                df = self._normalize_dataframe(df)
                dataframes[real_name] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
                # Print sample of ID columns for debugging
                for col in df.columns:
                    if 'id' in col.lower():
                        print(f"  {col} sample: {list(df[col].dropna().head(3))}")
            except Exception as exc:
                print(f"[KG] Error loading dataframe {real_name}: {exc}")
                dataframes[real_name] = pd.DataFrame()

        # LLM pass
        llm_agent  = LLMAgent()
        llm_edges  = llm_agent.generate_kg(metadata_map).get("edges", [])

        rel_agent      = RelationshipAgent()
        matching_agent = MatchingAgent()

        def node_id_fn(real_name: str) -> str:
            return path_map.get(real_name, real_name)

        def label_fn(real_name: str) -> str:
            return display_map.get(real_name, real_name)

        nodes_dict, edges = self._build_core(
            dataframes=dataframes,
            metadata_map=metadata_map,
            llm_edges=llm_edges,
            node_id_fn=node_id_fn,
            label_fn=label_fn,
            rel_agent=rel_agent,
            matching_agent=matching_agent,
            llm_agent=llm_agent,
        )

        edges = self._dedup_edges(edges)

        real_edges = [e for e in edges if e.get("relationship")]
        if not real_edges:
            nodes = list(nodes_dict.values())
            if nodes:
                final_nodes, _ = self._apply_layout(nodes, [], is_folder=False)
                return {
                    "nodes": final_nodes,
                    "edges": [],
                    "message": "No relationships found between selected datasets"
                }
            return {"nodes": [], "edges": [], "message": "No relationships found"}

        nodes = list(nodes_dict.values())
        final_nodes, edges = self._apply_layout(nodes, edges, is_folder=False)

        # CSV export: replace temp paths with real names
        def _clean_id(val: str) -> str:
            if ":" in val:
                tbl, col = val.split(":", 1)
                real = next((r for r, t in path_map.items() if t == tbl), tbl)
                return f"{real}:{col}"
            real = next((r for r, t in path_map.items() if t == val), val)
            return real

        csv_edges = []
        for e in edges:
            if not e.get("relationship"):
                csv_edges.append(e)
            else:
                csv_edges.append({**e,
                                   "source": _clean_id(e["source"]),
                                   "target": _clean_id(e["target"])})

        print(f"[KG] Final graph: {len(final_nodes)} nodes, {len(edges)} edges")
        return {"nodes": final_nodes, "edges": edges, "csv_edges": csv_edges}

    # ------------------------------------------------------------------
    # SELECT FOLDER MODE
    # ------------------------------------------------------------------

    def build_graph_from_folder(self, folder_name: str) -> dict:
        """
        Entry point for the SELECT FOLDER tab.
        """
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name    = "intern26"

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client    = blob_service_client.get_container_client(container_name)

        prefix = f"dqm/raw/{folder_name}/"
        blobs  = container_client.list_blobs(name_starts_with=prefix)

        local_files:   list = []
        file_name_map: dict = {}

        for blob in blobs:
            if not blob.name.endswith(".csv"):
                continue
            blob_client = container_client.get_blob_client(blob)
            tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            data = blob_client.download_blob().readall()
            tmp.write(data)
            tmp.close()
            real_name = os.path.basename(blob.name)
            local_files.append(tmp.name)
            file_name_map[tmp.name] = real_name

        if not local_files:
            return {"nodes": [], "edges": []}

        profiler     = ProfilingAgent()
        metadata_map = {}
        dataframes   = {}
        
        for tmp_path in local_files:
            real_name = file_name_map[tmp_path]
            print(f"[KG] Loading folder dataset: {real_name}")
            metadata_map[real_name] = profiler.profile(tmp_path) or {}
            try:
                df = pd.read_csv(tmp_path)
                # Normalize dataframe (convert ID columns to string)
                df = self._normalize_dataframe(df)
                dataframes[real_name] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as exc:
                print(f"[KG] Error loading {real_name}: {exc}")
                dataframes[real_name] = pd.DataFrame()

        # LLM pass
        llm_agent  = LLMAgent()
        llm_edges  = llm_agent.generate_kg(metadata_map).get("edges", [])

        rel_agent      = RelationshipAgent()
        matching_agent = MatchingAgent()

        def node_id_fn(real_name: str) -> str:
            return real_name

        def label_fn(real_name: str) -> str:
            return real_name

        nodes_dict, edges = self._build_core(
            dataframes=dataframes,
            metadata_map=metadata_map,
            llm_edges=llm_edges,
            node_id_fn=node_id_fn,
            label_fn=label_fn,
            rel_agent=rel_agent,
            matching_agent=matching_agent,
            llm_agent=llm_agent,
        )

        edges = self._dedup_edges(edges)
        edges = self._limit_edges_per_column(edges, max_edges=3)

        nodes = list(nodes_dict.values())
        final_nodes, edges = self._apply_layout(nodes, edges, is_folder=True)

        print(f"[KG] Folder mode final graph: {len(final_nodes)} nodes, {len(edges)} edges")
        return {"nodes": final_nodes, "edges": edges}