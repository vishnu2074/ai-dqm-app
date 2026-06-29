"""
python-backend/app/graph/kg_engine.py

FIXES:
  1. LLM uses direct HTTP requests against Azure AI Foundry endpoint format
     (/models/{model}/chat/completions) — fixes "Connection error" from SDK
  2. Domain inference reads the full blob physical_path, not just display name
     — fixes both CPG customers showing in CPG when one is BFSI
  3. Rule-based fallback when LLM unavailable (detects FK patterns + shared names)
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests as _http
from sqlalchemy.orm import Session

from app.models import (
    ColumnProfile, Dataset, DataSource,
    KnowledgeGraphEdge, ProfilingRun,
)

# ─── constants ────────────────────────────────────────────────────────────────

MIN_SHARED_TOKENS = 1
LLM_SAMPLE_SIZE   = 6
SKIP_TYPES        = {"BOOLEAN", "BOOL"}
FK_PATTERN = re.compile(
    r"(_id|_key|_code|_ref|_fk|_no|_num|id$|key$|code$|_uuid|_pk)$",
    re.IGNORECASE,
)
SAMPLE_FRAC = 0.05
SAMPLE_MAX  = 300

_AZURE_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_AZURE_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
_AZURE_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")


# ─── domain inference (path-based) ───────────────────────────────────────────

def _infer_domain(name: str, physical_path: str = "") -> str:
    """
    Read the domain from the blob path first (most reliable).
    Path structure: ai-dqm/raw/<domain>/...
    Falls back to name-based matching only for known non-ambiguous terms.
    """
    if physical_path:
        parts = physical_path.replace("\\", "/").lower().split("/")
        for part in parts:
            if part in ("bfsi", "banking", "finance", "fintech"):
                return "bfsi"
            if part in ("cpg", "retail", "consumer", "fmcg"):
                return "cpg"
            if part in ("hls", "health", "healthcare", "hospital", "medical"):
                return "hls"

    n = (name or "").lower()
    # BFSI-specific (check before generic "customer")
    if any(x in n for x in ("bfsi", "bank", "account", "transaction", "loan", "card", "credit")):
        return "bfsi"
    # HLS-specific
    if any(x in n for x in ("hls", "patient", "admission", "diagnos", "prescription", "lab_result")):
        return "hls"
    # CPG-specific
    if any(x in n for x in ("cpg", "product", "order", "supplier", "inventory", "sku")):
        return "cpg"
    return "other"


# ─── LLM via direct HTTP (Azure AI Foundry) ───────────────────────────────────

def _llm_chat(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """
    Azure AI Foundry endpoint (serverless): {endpoint}/models/chat/completions
    Model is specified in the request body, NOT in the URL path.
    The URL format /models/{model}/chat/completions causes 404 — the correct
    serverless API format puts the model in the JSON payload.

    Your endpoint: https://dilip-mm4oi19h-eastus2.services.ai.azure.com/models/chat/completions
    """
    if not _AZURE_KEY or not _AZURE_ENDPOINT:
        return None

    # Azure AI Foundry serverless: /models/chat/completions (model in body)
    # Strip any accidental trailing path segments from the env var
    ep = _AZURE_ENDPOINT
    for _sfx in ["/chat/completions", "/models"]:
        while ep.endswith(_sfx):
            ep = ep[:-len(_sfx)].rstrip("/")
    url = f"{ep}/models/chat/completions"

    try:
        resp = _http.post(
            url,
            headers={"Content-Type": "application/json", "api-key": _AZURE_KEY},
            json={
                "model": _AZURE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            params={"api-version": _AZURE_VERSION},
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except _http.exceptions.ConnectionError as e:
        print(f"[kg_engine] LLM connection error: {e}")
    except _http.exceptions.Timeout:
        print(f"[kg_engine] LLM request timed out")
    except Exception as e:
        print(f"[kg_engine] LLM call error: {e}")
    return None


# ─── helpers ──────────────────────────────────────────────────────────────────

def _type_family(data_type: str) -> str:
    t = (data_type or "").upper()
    if any(x in t for x in ("INT", "FLOAT", "NUMERIC", "DECIMAL", "DOUBLE", "REAL")):
        return "numeric"
    if any(x in t for x in ("VARCHAR", "TEXT", "STRING", "CHAR")):
        return "string"
    if any(x in t for x in ("DATE", "TIME", "TIMESTAMP")):
        return "datetime"
    if "BOOL" in t:
        return "boolean"
    return "other"


def _tokenize_name(name: str) -> set:
    return set(re.split(r"[_\-\s\.]+", name.lower())) - {"", "id", "the", "a"}


def _load_column_samples(db: Session, dataset_id: int, column_names: List[str]) -> Dict[str, List[str]]:
    result = {c: [] for c in column_names}
    if not column_names:
        return result
    try:
        dataset    = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        datasource = db.query(DataSource).filter(DataSource.id == dataset.datasource_id).first() if dataset else None
        if not dataset or not datasource:
            return result
        ds_type = (datasource.type or "").upper()
        if ds_type == "AZURE_BLOB":
            try:
                from azure.storage.blob import BlobServiceClient
                from app.services.datasources import decrypt_password
                conn_str  = decrypt_password(datasource.connection_string)
                container = datasource.container_name
                client    = BlobServiceClient.from_connection_string(conn_str)
                blob      = client.get_blob_client(container=container, blob=dataset.physical_name)
                data      = blob.download_blob().readall()
                df = pd.read_csv(io.BytesIO(data), usecols=[c for c in column_names if c], low_memory=False)
                n      = min(SAMPLE_MAX, max(1, int(len(df) * SAMPLE_FRAC)))
                sample = df.sample(n=min(n, len(df)), random_state=42) if len(df) > 0 else df
                for col in column_names:
                    if col in sample.columns:
                        vals = sample[col].dropna().head(LLM_SAMPLE_SIZE).astype(str).tolist()
                        result[col] = [v for v in vals if v not in ("nan", "None", "")]
            except Exception as e:
                print(f"[kg_engine] Azure sample load failed for dataset {dataset_id}: {e}")
        elif ds_type == "POSTGRESQL":
            try:
                import psycopg2
                from app.services.datasources import decrypt_password
                password = decrypt_password(datasource.encrypted_password) if datasource.encrypted_password else None
                conn = psycopg2.connect(host=datasource.host, port=datasource.port,
                    database=datasource.database, user=datasource.username, password=password)
                safe_cols  = ", ".join(f'"{c.replace(chr(34), "")}"' for c in column_names)
                safe_table = dataset.physical_name.replace('"', '""')
                df = pd.read_sql_query(
                    f'SELECT {safe_cols} FROM "{safe_table}" TABLESAMPLE BERNOULLI(5) LIMIT {SAMPLE_MAX}', conn)
                conn.close()
                for col in column_names:
                    if col in df.columns:
                        vals = df[col].dropna().head(LLM_SAMPLE_SIZE).astype(str).tolist()
                        result[col] = [v for v in vals if v not in ("nan", "None", "")]
            except Exception as e:
                print(f"[kg_engine] PG sample load failed for dataset {dataset_id}: {e}")
    except Exception as e:
        print(f"[kg_engine] _load_column_samples outer error: {e}")
    return result


# ─── column index ─────────────────────────────────────────────────────────────

def _build_column_index(db: Session, dataset_ids: List[int]) -> List[Dict[str, Any]]:
    result = []
    for ds_id in dataset_ids:
        run = (db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == ds_id, ProfilingRun.status == "COMPLETED")
            .order_by(ProfilingRun.id.desc()).first())
        if not run:
            continue
        dataset = db.query(Dataset).filter(Dataset.id == ds_id).first()
        if not dataset:
            continue
        physical = dataset.physical_name or ""
        ds_name  = (dataset.display_name or physical or f"dataset_{ds_id}")
        ds_name  = ds_name.split("/")[-1].split("\\")[-1].replace(".csv","").replace(".xlsx","")
        profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        for p in profiles:
            tf = _type_family(p.data_type or "")
            if tf == "boolean":
                continue
            result.append({
                "dataset_id":     ds_id,
                "dataset_name":   ds_name,
                "physical_path":  physical,
                "run_id":         run.id,
                "rows":           run.rows_processed or 0,
                "column_name":    p.column_name or "",
                "data_type":      p.data_type or "VARCHAR",
                "type_family":    tf,
                "uniqueness":     round(p.uniqueness or 0.0, 4),
                "completeness":   round(p.completeness or 0.0, 4),
                "null_count":     p.null_count or 0,
                "distinct_count": p.distinct_count or 0,
                "health_score":   p.health_score or 0.0,
                "is_key":         ((p.uniqueness or 0) >= 0.85 or bool(FK_PATTERN.search(p.column_name or ""))),
                "tokens":         _tokenize_name(p.column_name or ""),
            })
    return result


# ─── dataset-pair pre-filter ──────────────────────────────────────────────────

def _dataset_pairs_to_compare(columns: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    by_ds: Dict[int, List[Dict]] = {}
    for col in columns:
        by_ds.setdefault(col["dataset_id"], []).append(col)
    ds_ids = list(by_ds.keys())
    pairs  = []
    for i in range(len(ds_ids)):
        for j in range(i + 1, len(ds_ids)):
            ds_a, ds_b = ds_ids[i], ds_ids[j]
            cols_a, cols_b = by_ds[ds_a], by_ds[ds_b]
            tokens_a = set().union(*(c["tokens"] for c in cols_a))
            tokens_b = set().union(*(c["tokens"] for c in cols_b))
            shared   = tokens_a & tokens_b
            has_keys = any(c["is_key"] for c in cols_a) and any(c["is_key"] for c in cols_b)
            if len(shared) >= MIN_SHARED_TOKENS or has_keys:
                pairs.append((ds_a, ds_b))
    return pairs


# ─── rule-based fallback ──────────────────────────────────────────────────────

def _rule_based_detect(
    cols_a: List[Dict], cols_b: List[Dict],
    ds_a_id: int, ds_a_name: str,
    ds_b_id: int, ds_b_name: str,
) -> List[Dict]:
    """Detect FK/semantic relationships without LLM using naming conventions."""
    results = []
    seen    = set()

    def _norm(n: str) -> str:
        return re.sub(r"[_\-\s]+", "_", n.lower()).strip("_")

    names_a = {_norm(c["column_name"]): c for c in cols_a}
    names_b = {_norm(c["column_name"]): c for c in cols_b}

    GENERIC = {"region", "status", "gender", "country", "category", "type", "flag", "code"}

    # 1. Exact normalised name match
    for name, col_a in names_a.items():
        if name in names_b:
            col_b = names_b[name]
            if name in GENERIC and col_a["uniqueness"] < 0.05 and col_b["uniqueness"] < 0.05:
                continue
            key = (col_a["column_name"], col_b["column_name"])
            if key not in seen:
                seen.add(key)
                is_fk = bool(FK_PATTERN.search(col_a["column_name"]))
                results.append({
                    "source_dataset_id":    ds_a_id, "source_dataset_name": ds_a_name,
                    "source_column":        col_a["column_name"], "source_type": col_a["data_type"],
                    "source_uniqueness":    col_a["uniqueness"],
                    "target_dataset_id":    ds_b_id, "target_dataset_name": ds_b_name,
                    "target_column":        col_b["column_name"], "target_type": col_b["data_type"],
                    "target_uniqueness":    col_b["uniqueness"],
                    "relationship_type":    "foreign_key" if is_fk else "semantic",
                    "cardinality":          "1:N" if is_fk else "N:M",
                    "confidence":           0.85 if is_fk else 0.70,
                    "method":               "rule",
                    "llm_explanation":      f'"{col_a["column_name"]}" appears in both {ds_a_name} and {ds_b_name} — same column name and compatible types suggest a shared entity.',
                })

    # 2. FK stem match: customer_id in A ↔ customer_id in B
    for col_a in cols_a:
        if not FK_PATTERN.search(col_a["column_name"]):
            continue
        stem_a = re.sub(r"(_id|_key|_code|_ref|_fk)$", "", col_a["column_name"].lower())
        for col_b in cols_b:
            stem_b = re.sub(r"(_id|_key|_code|_ref|_fk)$", "", col_b["column_name"].lower())
            if stem_a and stem_b and (stem_a in stem_b or stem_b in stem_a):
                key = (col_a["column_name"], col_b["column_name"])
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "source_dataset_id":    ds_a_id, "source_dataset_name": ds_a_name,
                        "source_column":        col_a["column_name"], "source_type": col_a["data_type"],
                        "source_uniqueness":    col_a["uniqueness"],
                        "target_dataset_id":    ds_b_id, "target_dataset_name": ds_b_name,
                        "target_column":        col_b["column_name"], "target_type": col_b["data_type"],
                        "target_uniqueness":    col_b["uniqueness"],
                        "relationship_type":    "foreign_key",
                        "cardinality":          "1:N",
                        "confidence":           0.75,
                        "method":               "rule",
                        "llm_explanation":      f'"{col_a["column_name"]}" in {ds_a_name} appears to be a foreign key referencing "{col_b["column_name"]}" in {ds_b_name} based on naming pattern.',
                    })
    return results


# ─── LLM batched detection ────────────────────────────────────────────────────

def _build_column_summary(col: Dict, samples: List[str]) -> str:
    sample_str = ", ".join(f'"{v}"' for v in samples[:LLM_SAMPLE_SIZE]) if samples else "no samples"
    return (f"  - {col['column_name']} [{col['data_type']}] "
            f"uniqueness={col['uniqueness']:.0%} completeness={col['completeness']:.0%} "
            f"{'[KEY] ' if col['is_key'] else ''}samples: {sample_str}")


def _llm_detect_pairs(
    db: Session,
    ds_a_id: int, ds_a_name: str, cols_a: List[Dict],
    ds_b_id: int, ds_b_name: str, cols_b: List[Dict],
) -> List[Dict[str, Any]]:
    key_cols_a = [c for c in cols_a if c["is_key"] or c["type_family"] in ("string","numeric")][:20]
    key_cols_b = [c for c in cols_b if c["is_key"] or c["type_family"] in ("string","numeric")][:20]

    samples_a = _load_column_samples(db, ds_a_id, [c["column_name"] for c in key_cols_a])
    samples_b = _load_column_samples(db, ds_b_id, [c["column_name"] for c in key_cols_b])

    summary_a = "\n".join(_build_column_summary(c, samples_a.get(c["column_name"], [])) for c in key_cols_a)
    summary_b = "\n".join(_build_column_summary(c, samples_b.get(c["column_name"], [])) for c in key_cols_b)

    prompt = f"""You are a senior data engineer analyzing two datasets to find column-level relationships.

DATASET A: "{ds_a_name}" ({cols_a[0]['rows'] if cols_a else 0:,} rows)
{summary_a}

DATASET B: "{ds_b_name}" ({cols_b[0]['rows'] if cols_b else 0:,} rows)
{summary_b}

Identify ALL pairs (one column from A, one from B) that have a join/FK relationship or represent the same real-world entity.

Return ONLY valid JSON, no markdown fences:
{{"relationships": [{{"column_a": "...", "column_b": "...", "relationship_type": "foreign_key|semantic|derived", "cardinality": "1:1|1:N|N:M", "confidence": 0.0-1.0, "explanation": "..."}}]}}

If none exist return {{"relationships": []}}"""

    raw = _llm_chat(prompt, max_tokens=800)

    if raw is None:
        print(f"[kg_engine] LLM unavailable for {ds_a_name} ↔ {ds_b_name}, using rule-based fallback")
        return _rule_based_detect(cols_a, cols_b, ds_a_id, ds_a_name, ds_b_id, ds_b_name)

    try:
        raw    = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        rels   = parsed.get("relationships", [])
        valid_a = {c["column_name"] for c in cols_a}
        valid_b = {c["column_name"] for c in cols_b}
        confirmed = []
        for r in rels:
            col_a_name = r.get("column_a", "")
            col_b_name = r.get("column_b", "")
            if col_a_name not in valid_a or col_b_name not in valid_b:
                continue
            if float(r.get("confidence", 0)) < 0.4:
                continue
            col_a_meta = next(c for c in cols_a if c["column_name"] == col_a_name)
            col_b_meta = next(c for c in cols_b if c["column_name"] == col_b_name)
            confirmed.append({
                "source_dataset_id":    ds_a_id, "source_dataset_name": ds_a_name,
                "source_column":        col_a_name, "source_type": col_a_meta["data_type"],
                "source_uniqueness":    col_a_meta["uniqueness"],
                "target_dataset_id":    ds_b_id, "target_dataset_name": ds_b_name,
                "target_column":        col_b_name, "target_type": col_b_meta["data_type"],
                "target_uniqueness":    col_b_meta["uniqueness"],
                "relationship_type":    r.get("relationship_type", "semantic"),
                "cardinality":          r.get("cardinality", "N:M"),
                "confidence":           float(r.get("confidence", 0.5)),
                "method":               "llm",
                "llm_explanation":      r.get("explanation", ""),
            })
        return confirmed
    except Exception as e:
        print(f"[kg_engine] LLM parse failed for {ds_a_name} ↔ {ds_b_name}: {e}, using rule-based")
        return _rule_based_detect(cols_a, cols_b, ds_a_id, ds_a_name, ds_b_id, ds_b_name)


# ─── persist ──────────────────────────────────────────────────────────────────

def _save_edges(db: Session, relationships: List[Dict]) -> int:
    saved = 0
    for r in relationships:
        existing = (db.query(KnowledgeGraphEdge)
            .filter(
                KnowledgeGraphEdge.source_dataset_id == r["source_dataset_id"],
                KnowledgeGraphEdge.source_column     == r["source_column"],
                KnowledgeGraphEdge.target_dataset_id == r["target_dataset_id"],
                KnowledgeGraphEdge.target_column     == r["target_column"],
                KnowledgeGraphEdge.invalidated       == False,
            ).first())
        if existing:
            if r["confidence"] > existing.confidence:
                existing.confidence      = r["confidence"]
                existing.llm_explanation = r.get("llm_explanation")
                existing.cardinality     = r.get("cardinality")
            continue
        edge = KnowledgeGraphEdge(
            source_dataset_id   = r["source_dataset_id"],
            source_column       = r["source_column"],
            source_dataset_name = r["source_dataset_name"],
            target_dataset_id   = r["target_dataset_id"],
            target_column       = r["target_column"],
            target_dataset_name = r["target_dataset_name"],
            relationship_type   = r.get("relationship_type", "semantic"),
            cardinality         = r.get("cardinality", "N:M"),
            name_similarity     = None, value_overlap = None,
            confidence          = r["confidence"],
            method              = r.get("method", "rule"),
            llm_explanation     = r.get("llm_explanation"),
            invalidated         = False,
        )
        db.add(edge)
        saved += 1
    db.commit()
    return saved


# ─── graph builder ────────────────────────────────────────────────────────────

def build_column_graph(db: Session, dataset_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    if dataset_ids is None:
        runs = db.query(ProfilingRun.dataset_id).filter(ProfilingRun.status == "COMPLETED").distinct().all()
        dataset_ids = [r[0] for r in runs]

    columns  = _build_column_index(db, dataset_ids)
    nodes, edges, node_ids = [], [], set()

    by_ds: Dict[int, List[Dict]] = {}
    for col in columns:
        by_ds.setdefault(col["dataset_id"], []).append(col)

    for ds_id, ds_cols in by_ds.items():
        ds_name       = ds_cols[0]["dataset_name"]
        physical_path = ds_cols[0].get("physical_path", "")
        scores        = [c["health_score"] for c in ds_cols if c["health_score"]]
        avg_health    = round(sum(scores)/len(scores), 1) if scores else 0.0
        domain        = _infer_domain(ds_name, physical_path)

        ds_node_id = f"ds_{ds_id}"
        if ds_node_id not in node_ids:
            nodes.append({"id": ds_node_id, "label": ds_name, "type": "dataset",
                          "dataset_id": ds_id, "health_score": avg_health,
                          "column_count": len(ds_cols), "domain": domain})
            node_ids.add(ds_node_id)

        for col in ds_cols:
            col_node_id = f"col_{ds_id}_{col['column_name']}"
            if col_node_id not in node_ids:
                nodes.append({"id": col_node_id, "label": col["column_name"], "type": "column",
                              "dataset_id": ds_id, "dataset_name": ds_name, "domain": domain,
                              "data_type": col["data_type"], "type_family": col["type_family"],
                              "uniqueness": col["uniqueness"], "completeness": col["completeness"],
                              "health_score": col["health_score"], "is_key": col["is_key"],
                              "null_count": col["null_count"], "distinct_count": col["distinct_count"]})
                node_ids.add(col_node_id)
            edges.append({"id": f"contains_{ds_node_id}_{col_node_id}",
                          "source": ds_node_id, "target": col_node_id, "edge_type": "contains"})

    kg_edges = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.invalidated == False).all()
    for e in kg_edges:
        src = f"col_{e.source_dataset_id}_{e.source_column}"
        tgt = f"col_{e.target_dataset_id}_{e.target_column}"
        if src not in node_ids or tgt not in node_ids:
            continue
        edges.append({"id": f"rel_{e.id}", "source": src, "target": tgt,
                      "edge_type": "relationship", "relationship_type": e.relationship_type,
                      "cardinality": e.cardinality, "confidence": e.confidence,
                      "method": e.method, "llm_explanation": e.llm_explanation,
                      "source_dataset_id": e.source_dataset_id, "target_dataset_id": e.target_dataset_id})

    return {"nodes": nodes, "edges": edges, "stats": {
        "datasets": len(by_ds),
        "columns": len([n for n in nodes if n["type"] == "column"]),
        "relationship_edges": len([e for e in edges if e["edge_type"] == "relationship"]),
    }}


# ─── public: detect relationships ─────────────────────────────────────────────

def detect_relationships(db: Session, dataset_ids: Optional[List[int]] = None, force_rerun: bool = False) -> Dict[str, Any]:
    t0 = time.time()

    if dataset_ids is None:
        runs = db.query(ProfilingRun.dataset_id).filter(ProfilingRun.status == "COMPLETED").distinct().all()
        dataset_ids = [r[0] for r in runs]

    if len(dataset_ids) < 2:
        return {"status": "INSUFFICIENT_DATA", "message": "Need at least 2 profiled datasets.", "edges_saved": 0}

    if force_rerun:
        db.query(KnowledgeGraphEdge).filter(
            KnowledgeGraphEdge.source_dataset_id.in_(dataset_ids)
        ).update({"invalidated": True}, synchronize_session=False)
        db.commit()

    if not force_rerun:
        existing = db.query(KnowledgeGraphEdge).filter(
            KnowledgeGraphEdge.source_dataset_id.in_(dataset_ids),
            KnowledgeGraphEdge.invalidated == False,
        ).count()
        if existing > 0:
            return {"status": "CACHED", "message": f"Using {existing} cached relationships.",
                    "edges_saved": existing, "elapsed_ms": 0, "llm_calls": 0}

    columns = _build_column_index(db, dataset_ids)
    if not columns:
        return {"status": "NO_PROFILES", "message": "No column profiles found. Run DQ Scoring first.", "edges_saved": 0}

    pairs = _dataset_pairs_to_compare(columns)
    if not pairs:
        return {"status": "NO_PAIRS", "message": "No overlapping column patterns found.", "edges_saved": 0, "llm_calls": 0}

    by_ds: Dict[int, List[Dict]] = {}
    for col in columns:
        by_ds.setdefault(col["dataset_id"], []).append(col)
    ds_names = {col["dataset_id"]: col["dataset_name"] for col in columns}

    all_relationships, llm_calls = [], 0
    for ds_a_id, ds_b_id in pairs:
        cols_a = by_ds.get(ds_a_id, [])
        cols_b = by_ds.get(ds_b_id, [])
        if not cols_a or not cols_b:
            continue
        rels = _llm_detect_pairs(db,
            ds_a_id, ds_names.get(ds_a_id, f"ds_{ds_a_id}"), cols_a,
            ds_b_id, ds_names.get(ds_b_id, f"ds_{ds_b_id}"), cols_b)
        all_relationships.extend(rels)
        llm_calls += 1

    edges_saved = _save_edges(db, all_relationships)
    elapsed     = round((time.time() - t0) * 1000)

    # Determine if LLM was actually used
    used_llm = any(r.get("method") == "llm" for r in all_relationships)

    return {
        "status":              "OK",
        "elapsed_ms":          elapsed,
        "datasets_analysed":   len(dataset_ids),
        "pairs_compared":      len(pairs),
        "llm_calls":           llm_calls,
        "relationships_found": len(all_relationships),
        "edges_saved":         edges_saved,
        "message":             f"Detected {len(all_relationships)} relationships using {'LLM' if used_llm else 'rule-based'} analysis.",
    }


# ─── public helpers ───────────────────────────────────────────────────────────

def get_relationships_for_dataset(db: Session, dataset_id: int) -> List[Dict[str, Any]]:
    edges = (db.query(KnowledgeGraphEdge)
        .filter(
            (KnowledgeGraphEdge.source_dataset_id == dataset_id) |
            (KnowledgeGraphEdge.target_dataset_id == dataset_id),
            KnowledgeGraphEdge.invalidated == False,
        ).order_by(KnowledgeGraphEdge.confidence.desc()).all())
    return [{
        "id": e.id,
        "source_dataset_id": e.source_dataset_id, "source_dataset_name": e.source_dataset_name,
        "source_column": e.source_column,
        "target_dataset_id": e.target_dataset_id, "target_dataset_name": e.target_dataset_name,
        "target_column": e.target_column,
        "relationship_type": e.relationship_type, "cardinality": e.cardinality,
        "confidence": e.confidence, "method": e.method, "llm_explanation": e.llm_explanation,
        "detected_at": e.detected_at.isoformat() if e.detected_at else None,
    } for e in edges]


def get_all_kg_edges(db: Session) -> List[Dict[str, Any]]:
    edges = (db.query(KnowledgeGraphEdge)
        .filter(KnowledgeGraphEdge.invalidated == False)
        .order_by(KnowledgeGraphEdge.confidence.desc()).all())
    return [{
        "id": e.id,
        "source_dataset_id": e.source_dataset_id, "source_dataset_name": e.source_dataset_name,
        "source_column": e.source_column,
        "target_dataset_id": e.target_dataset_id, "target_dataset_name": e.target_dataset_name,
        "target_column": e.target_column,
        "relationship_type": e.relationship_type, "cardinality": e.cardinality,
        "confidence": e.confidence, "method": e.method, "llm_explanation": e.llm_explanation,
    } for e in edges]


def invalidate_dataset_edges(db: Session, dataset_id: int) -> int:
    count = (db.query(KnowledgeGraphEdge)
        .filter(
            (KnowledgeGraphEdge.source_dataset_id == dataset_id) |
            (KnowledgeGraphEdge.target_dataset_id == dataset_id))
        .update({"invalidated": True}, synchronize_session=False))
    db.commit()
    return count