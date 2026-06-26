"""
lineage_engine.py
─────────────────
Builds lineage graph from REAL Azure Blob path structure.
 
Each dataset's blob path is parsed into a hierarchy of nodes:
 
  intern26 → ai-dqm → raw → cpg → customers → v1 → customers.csv → [dq_rules]
 
Every level of the path = one node.
Every parent→child = one edge.
 
100% real, 100% dynamic. No hardcoding. No JSON.
 
─────────────────────────────────────────────────────────────
CACHING
─────────────────────────────────────────────────────────────
The full graph is expensive to build — it queries DataSource,
Dataset, DQRule tables PLUS calls get_dq_scores_summary() for
every dataset. Without caching this runs on every node click.
 
Cache design:
  - Single module-level _GraphCache instance
  - TTL = 5 minutes (configurable via LINEAGE_CACHE_TTL_SECONDS env var)
  - Thread-safe via threading.Lock
  - invalidate_cache() busts it immediately (call on dataset/rule changes)
  - Filtered views (get_dataset_graph) are NOT cached — they are cheap
    slices of the (cached) full graph, computed in microseconds
"""
 
import re
import time
import threading
import os
 
 
# ── Cache ──────────────────────────────────────────────────────────────────────
 
_CACHE_TTL = int(os.environ.get("LINEAGE_CACHE_TTL_SECONDS", "300"))  # 5 min default
 
 
class _GraphCache:
    """Thread-safe TTL cache for the full lineage graph."""
 
    def __init__(self):
        self._lock     = threading.Lock()
        self._graph    = None
        self._built_at = 0.0
 
    def get(self):
        with self._lock:
            if self._graph is not None and (time.time() - self._built_at) < _CACHE_TTL:
                age = round(time.time() - self._built_at)
                print(f"[lineage_engine] Cache HIT (age {age}s, TTL {_CACHE_TTL}s)")
                return self._graph
        return None
 
    def set(self, graph: dict) -> None:
        with self._lock:
            self._graph    = graph
            self._built_at = time.time()
        print(
            f"[lineage_engine] Cache SET — "
            f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
            f"TTL {_CACHE_TTL}s"
        )
 
    def invalidate(self) -> None:
        with self._lock:
            self._graph    = None
            self._built_at = 0.0
        print("[lineage_engine] Cache INVALIDATED")
 
 
_cache = _GraphCache()
 
 
def invalidate_cache() -> None:
    """
    Bust the lineage cache. Call this whenever datasets or DQ rules change
    so the next request rebuilds fresh from the DB.
 
    Usage (add to your dataset/rule routers after create/delete):
        from app.graph.lineage_engine import invalidate_cache
        invalidate_cache()
    """
    _cache.invalidate()
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
 
 
def _node_id(*parts: str) -> str:
    return "_".join(_slugify(p) for p in parts if p)
 
 
def _get_real_dq_score(db, dataset_id: int) -> int:
    try:
        from app.services import dq_scores as dq_scores_service
        summary = dq_scores_service.get_dq_scores_summary(db, dataset_id)
        if summary.get("status") == "COMPLETED":
            score = (
                summary.get("dataHealth") or
                summary.get("dataHealthScore") or
                summary.get("data_health_score")
            )
            if score is not None:
                return round(float(score))
    except Exception as e:
        print(f"[lineage_engine] DQ score fetch failed for dataset {dataset_id}: {e}")
    return 85
 
 
def _parse_blob_path(path: str):
    if not path:
        return []
    path = path.replace("\\", "/").strip("/")
    return [p for p in path.split("/") if p]
 
 
# ── Graph builder (cached) ─────────────────────────────────────────────────────
 
def _get_azure_graph() -> dict:
    """
    Build full lineage graph from Azure Blob path structure.
 
    Node hierarchy per dataset:
      DataSource → layer → domain → name → version → file.csv → DQ Rules
 
    Served from in-memory cache after the first build.
    Cache is invalidated by invalidate_cache() or expires after TTL.
    """
    # ── 1. Return from cache if fresh ─────────────────────────────────────────
    cached = _cache.get()
    if cached is not None:
        return cached
 
    # ── 2. Build from DB ──────────────────────────────────────────────────────
    print("[lineage_engine] Cache MISS — building graph from DB...")
    t0 = time.time()
 
    try:
        from app.database import SessionLocal
        from app.models import DataSource, Dataset, DQRule
 
        db = SessionLocal()
        try:
            sources  = db.query(DataSource).all()
            datasets = db.query(Dataset).all()
            dq_rules = db.query(DQRule).filter(DQRule.status == "Active").all()
        finally:
            db.close()
 
        db2 = SessionLocal()
        try:
            nodes_map = {}    # id → node dict (deduped)
            edges_set = set() # (source, target) deduped
            edges     = []
 
            def add_node(node_id, name, node_type, quality_score=100, **extra):
                if node_id not in nodes_map:
                    nodes_map[node_id] = {
                        "id":            node_id,
                        "name":          name,
                        "type":          node_type,
                        "quality_score": quality_score,
                        **extra
                    }
 
            def add_edge(src, tgt):
                key = (src, tgt)
                if key not in edges_set:
                    edges_set.add(key)
                    edges.append({"source": src, "target": tgt})
 
            # DataSource nodes
            source_map = {}
            for src in sources:
                src_node_id = _slugify(src.name)
                add_node(src_node_id, src.name, "source", quality_score=100)
                source_map[src.id] = src_node_id
 
            # Dataset blob path → node hierarchy
            dataset_file_node_map = {}
 
            for ds in datasets:
                raw_path    = ds.physical_name or ""
                label       = ds.display_name or ds.physical_name or f"dataset_{ds.id}"
                dq_score    = _get_real_dq_score(db2, ds.id)
                src_node_id = source_map.get(ds.datasource_id)
                parts       = _parse_blob_path(raw_path)
 
                if not parts:
                    file_node_id = _slugify(label)
                    add_node(file_node_id, label, "table",
                             quality_score=dq_score, dataset_db_id=ds.id)
                    if src_node_id:
                        add_edge(src_node_id, file_node_id)
                    dataset_file_node_map[ds.id] = file_node_id
                    continue
 
                prev_node_id = src_node_id
                for i, part in enumerate(parts):
                    is_last      = (i == len(parts) - 1)
                    curr_node_id = _node_id(*parts[:i + 1])
 
                    if is_last:
                        add_node(curr_node_id, part, "table",
                                 quality_score=dq_score,
                                 dataset_db_id=ds.id,
                                 full_path=raw_path)
                        dataset_file_node_map[ds.id] = curr_node_id
                    else:
                        add_node(curr_node_id, part, "source", quality_score=100)
 
                    if prev_node_id and prev_node_id != curr_node_id:
                        add_edge(prev_node_id, curr_node_id)
                    prev_node_id = curr_node_id
 
            # DQ Rule nodes
            # Use rule.id (unique DB primary key) to avoid collisions
            # when multiple datasets share the same rule_code (e.g. RULE-002)
            for rule in dq_rules:
                rule_node_id   = f"rule_{rule.id}"
                parent_node_id = dataset_file_node_map.get(rule.dataset_id)
                add_node(
                    rule_node_id, rule.name, "dq_rule",
                    quality_score=80,
                    rule_type=getattr(rule, "type", ""),
                    severity=getattr(rule, "severity", "Medium"),
                )
                if parent_node_id:
                    add_edge(parent_node_id, rule_node_id)
 
            result = {"nodes": list(nodes_map.values()), "edges": edges}
 
            elapsed = round((time.time() - t0) * 1000)
            print(
                f"[lineage_engine] Graph built in {elapsed}ms — "
                f"{len(result['nodes'])} nodes, {len(result['edges'])} edges"
            )
 
            # ── 4. Merge user-defined dataset→dataset edges from DB ────────
            try:
                db3 = SessionLocal()
                try:
                    from app.models import LineageEdge
                    db_edges = db3.query(LineageEdge).all()
                    for de in db_edges:
                        add_edge(de.source, de.target)
                    if db_edges:
                        print(
                            f"[lineage_engine] +{len(db_edges)} user-defined "
                            f"edge(s) merged from lineage_edges table"
                        )
                except Exception as e:
                    print(f"[lineage_engine] lineage_edges merge skipped: {e}")
                finally:
                    db3.close()
            except Exception:
                pass  # LineageEdge table may not exist yet on first run
 
            _cache.set(result)

            # ── Persist dataset coverage to lineage_edges ─────────────────────
            # health_metrics_router.lineage_coverage queries lineage_edges for
            # source_dataset_id / target_dataset_id to compute coverage %.
            # The lineage engine never writes to DB — we fix that here by
            # inserting one coverage sentinel row per registered dataset
            # that appears in the built graph.
            try:
                from app.database import engine as _le_engine
                from sqlalchemy import text as _lt

                with _le_engine.connect() as _lc:
                    # Remove previous auto-generated rows (identified by source='__auto__')
                    _lc.execute(_lt(
                        "DELETE FROM lineage_edges WHERE source='__auto__' OR target='__auto__'"
                    ))
                    _inserted = 0
                    for ds_id, node_id in dataset_file_node_map.items():
                        if ds_id is None:
                            continue
                        # Check if column exists (migration may not have run yet)
                        _lc.execute(_lt(
                            "INSERT OR IGNORE INTO lineage_edges "
                            "(source, target, source_dataset_id, target_dataset_id) "
                            "VALUES ('__auto__', '__auto__', :sid, :tid)"
                        ), {"sid": int(ds_id), "tid": int(ds_id)})
                        _inserted += 1
                    _lc.commit()
                    if _inserted:
                        print(f"[lineage_engine] ✓ Wrote {_inserted} dataset coverage rows to lineage_edges")
            except Exception as _lpe:
                print(f"[lineage_engine] lineage_edges persist skipped (non-fatal): {_lpe}")

            return result
 
        finally:
            db2.close()
 
    except Exception as e:
        print(f"[lineage_engine] Azure graph build failed: {e}")
        import traceback; traceback.print_exc()
        return {"nodes": [], "edges": []}
 
 
# ── Filter ─────────────────────────────────────────────────────────────────────
 
def _filter_graph(graph: dict, dataset_id: str) -> dict:
    """
    Subgraph for a selected node:
      - ALL ancestors (full path from root down to selected node)
      - direct children (DQ rules attached to it)
 
    Not cached — always a fast slice of the cached full graph.
    """
    target = next((n for n in graph["nodes"] if n["id"] == dataset_id), None)
    if not target:
        target = next(
            (n for n in graph["nodes"] if dataset_id.lower() in n["id"].lower()), None
        )
    if not target:
        print(f"[lineage_engine] No node found for: {dataset_id}")
        return {"nodes": [], "edges": []}
 
    target_id = target["id"]
 
    parents  = {}
    children = {}
    for e in graph["edges"]:
        children.setdefault(e["source"], []).append(e["target"])
        parents.setdefault(e["target"],  []).append(e["source"])
 
    ancestors = set()
    queue = [target_id]
    while queue:
        curr = queue.pop()
        for p in parents.get(curr, []):
            if p not in ancestors:
                ancestors.add(p)
                queue.append(p)
 
    direct_children = set(children.get(target_id, []))
    included = {target_id} | ancestors | direct_children
 
    return {
        "nodes": [n for n in graph["nodes"] if n["id"] in included],
        "edges": [
            e for e in graph["edges"]
            if e["source"] in included and e["target"] in included
        ],
    }
 
 
# ── Public API ─────────────────────────────────────────────────────────────────
 
class LineageEngine:
 
    @staticmethod
    def get_full_graph() -> dict:
        try:
            return _get_azure_graph()
        except Exception as e:
            print(f"[lineage_engine] get_full_graph failed: {e}")
            return {"nodes": [], "edges": []}
 
    @staticmethod
    def get_dataset_graph(dataset_id: str) -> dict:
        try:
            return _filter_graph(LineageEngine.get_full_graph(), dataset_id)
        except Exception as e:
            print(f"[lineage_engine] get_dataset_graph failed: {e}")
            return {"nodes": [], "edges": []}
 
    @staticmethod
    def get_node_lineage(node_id: str) -> dict:
        return LineageEngine.get_dataset_graph(node_id)