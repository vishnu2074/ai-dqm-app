# python-backend/app/services/ai_recommendations.py
"""
AI Rule Recommendations.

FIXES:
  - Reads dataset directly from Azure Blob (no local file cache needed)
  - LLM calls use Azure AI Foundry HTTP endpoint directly (not openai SDK)
  - Falls back to rule-based recommendations if LLM unavailable
"""
from __future__ import annotations

import io
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import requests as _http
from sqlalchemy.orm import Session

from app.models import Dataset, DataSource, ColumnProfile, ProfilingRun
from app.services.dq_engine import _apply_rule

_AZURE_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_AZURE_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
_AZURE_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")


def _llm_chat(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """Direct HTTP call to Azure AI Foundry — bypasses SDK routing issues."""
    if not _AZURE_KEY or not _AZURE_ENDPOINT:
        return None
    url = f"{_AZURE_ENDPOINT}/models/{_AZURE_MODEL}/chat/completions"
    try:
        resp = _http.post(
            url,
            headers={"Content-Type": "application/json", "api-key": _AZURE_KEY},
            json={"messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": max_tokens},
            params={"api-version": _AZURE_VERSION},
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ai_recommendations] LLM error: {e}")
        return None


def _load_df_from_blob(db: Session, dataset_id: int) -> Optional[pd.DataFrame]:
    """Load dataset directly from Azure Blob — no local file cache required."""
    try:
        dataset    = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        datasource = db.query(DataSource).filter(DataSource.id == dataset.datasource_id).first() if dataset else None
        if not dataset or not datasource:
            return None

        ds_type = (datasource.type or "").upper()

        if ds_type == "AZURE_BLOB":
            from azure.storage.blob import BlobServiceClient
            from app.services.datasources import decrypt_password
            conn_str  = decrypt_password(datasource.connection_string)
            container = datasource.container_name
            client    = BlobServiceClient.from_connection_string(conn_str)
            blob      = client.get_blob_client(container=container, blob=dataset.physical_name)
            data      = blob.download_blob().readall()
            return pd.read_csv(io.BytesIO(data), low_memory=False)

        elif ds_type == "POSTGRESQL":
            import psycopg2
            from app.services.datasources import decrypt_password
            password = decrypt_password(datasource.encrypted_password) if datasource.encrypted_password else None
            conn = psycopg2.connect(
                host=datasource.host, port=datasource.port,
                database=datasource.database, user=datasource.username, password=password)
            safe_table = dataset.physical_name.replace('"', '""')
            df = pd.read_sql_query(f'SELECT * FROM "{safe_table}" LIMIT 5000', conn)
            conn.close()
            return df

        else:
            # Try the original local-file path as last resort
            from app.services.dq_engine import _get_version, _resolve_to_local_file, _read_dataset
            version    = _get_version(db, dataset_id, version_id=None)
            local_path = _resolve_to_local_file(db, dataset, version.file_path)
            return _read_dataset(local_path)

    except Exception as e:
        print(f"[ai_recommendations] Could not load df for dataset {dataset_id}: {e}")
        return None


def _get_column_profiles(db: Session, dataset_id: int) -> List[Dict]:
    """Get column profiles from DB — used to build the LLM prompt without needing file."""
    run = (db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(ProfilingRun.id.desc()).first())
    if not run:
        return []
    profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
    return [{
        "column":       p.column_name,
        "type":         p.data_type or "VARCHAR",
        "completeness": round((p.completeness or 0) * 100, 1),
        "uniqueness":   round((p.uniqueness or 0) * 100, 1),
        "null_count":   p.null_count or 0,
        "distinct":     p.distinct_count or 0,
        "health":       round(p.health_score or 0, 1),
    } for p in profiles]


def _rule_based_recommendations(profiles: List[Dict], dataset_name: str) -> List[Dict]:
    """
    Generate DQ rule recommendations from column profiles without LLM.
    Uses heuristics based on column names, types, and profile statistics.
    """
    rules = []
    seen  = set()

    for p in profiles:
        col  = p["column"]
        ctype = (p["type"] or "").upper()
        comp  = p["completeness"]
        uniq  = p["uniqueness"]
        name  = col.lower()

        # 1. Uniqueness rule for ID columns
        if re.search(r"(_id|_key|_uuid|_pk)$", name, re.I) and uniq < 95:
            key = f"unique_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Uniqueness Check",
                    "type":      "Uniqueness",
                    "column":    col,
                    "condition": f"{col}.is_unique",
                    "severity":  "High",
                    "reasoning": f'"{col}" appears to be an identifier column but has {100-uniq:.1f}% duplicate values.',
                    "confidence": 0.90,
                })

        # 2. Not-null rule for high-importance columns
        if comp < 95 and re.search(r"(_id|_name|_email|_date|_amount|_status)$|^(id|name|email|date|amount|status)", name, re.I):
            key = f"notnull_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Not Null Check",
                    "type":      "Completeness",
                    "column":    col,
                    "condition": f"{col}.notna()",
                    "severity":  "Medium",
                    "reasoning": f'"{col}" has {100-comp:.1f}% null values but appears to be a required field.',
                    "confidence": 0.85,
                })

        # 3. Email format validation
        if "email" in name:
            key = f"email_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Email Format",
                    "type":      "Validity",
                    "column":    col,
                    "condition": f'{col}.str.contains("@", na=False)',
                    "severity":  "Medium",
                    "reasoning": f'"{col}" is an email column — validate format contains @ symbol.',
                    "confidence": 0.88,
                })

        # 4. Negative value check for amount/price/quantity columns
        if re.search(r"(amount|price|cost|salary|income|balance|quantity|count)", name, re.I):
            if any(x in ctype for x in ("INT", "FLOAT", "NUMERIC", "DECIMAL")):
                key = f"nonneg_{col}"
                if key not in seen:
                    seen.add(key)
                    rules.append({
                        "name":      f"{col} Non-Negative Check",
                        "type":      "Validity",
                        "column":    col,
                        "condition": f"{col} >= 0",
                        "severity":  "High",
                        "reasoning": f'"{col}" should not have negative values for financial/quantity data.',
                        "confidence": 0.87,
                    })

        # 5. Date in past check
        if re.search(r"(_date|_at|birth|dob)", name, re.I) and any(x in ctype for x in ("DATE", "TIME")):
            key = f"date_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Valid Date Check",
                    "type":      "Validity",
                    "column":    col,
                    "condition": f"pd.to_datetime({col}, errors='coerce').notna()",
                    "severity":  "Low",
                    "reasoning": f'"{col}" is a date column — validate all values are parseable dates.',
                    "confidence": 0.82,
                })

        # 6. Status/category allowed values (low distinct count string columns)
        if any(x in ctype for x in ("VARCHAR", "TEXT", "STRING")) and 2 <= p["distinct"] <= 10 and uniq < 10:
            if re.search(r"(status|type|category|tier|level|gender|flag)", name, re.I):
                key = f"enum_{col}"
                if key not in seen:
                    seen.add(key)
                    rules.append({
                        "name":      f"{col} Allowed Values",
                        "type":      "Validity",
                        "column":    col,
                        "condition": f"{col}.notna()",
                        "severity":  "Medium",
                        "reasoning": f'"{col}" has only {p["distinct"]} distinct values — validate against an allowed values list.',
                        "confidence": 0.80,
                    })

    return rules[:10]  # Return top 10


def _llm_recommendations(profiles: List[Dict], dataset_name: str) -> Optional[List[Dict]]:
    """Generate DQ rules via LLM using column profile metadata."""
    if not profiles:
        return None

    col_desc = "\n".join(
        f"  {p['column']} ({p['type']}) — completeness: {p['completeness']}%, "
        f"uniqueness: {p['uniqueness']}%, nulls: {p['null_count']}, distinct: {p['distinct']}"
        for p in profiles[:20]
    )

    prompt = f"""You are a data quality expert. Analyze the following dataset column profiles and recommend 5-8 specific, actionable DQ validation rules.

Dataset: "{dataset_name}"
Columns:
{col_desc}

For each rule, provide:
- name: short descriptive rule name
- type: one of Completeness, Uniqueness, Validity, Consistency, Accuracy
- column: exact column name from the list above
- condition: a simple pandas-style condition string (e.g. "{'{'}col{'}'} >= 0", "{'{'}col{'}'}.notna()", "{'{'}col{'}'}.str.contains('@')")
- severity: Critical, High, Medium, or Low
- reasoning: one sentence explaining why this rule matters
- confidence: 0.0-1.0

Return ONLY valid JSON, no markdown:
{{"rules": [{{"name": "...", "type": "...", "column": "...", "condition": "...", "severity": "...", "reasoning": "...", "confidence": 0.85}}]}}"""

    raw = _llm_chat(prompt)
    if raw is None:
        return None

    try:
        raw    = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        return parsed.get("rules", [])
    except Exception as e:
        print(f"[ai_recommendations] LLM parse error: {e}")
        return None


# ── Public functions ───────────────────────────────────────────────────────────

def generate_ai_recommendations(db: Session, dataset_id: int, k: int = 8) -> Dict[str, Any]:
    """
    Returns {status, rules} where each rule has the shape the frontend expects.
    Uses LLM if available, falls back to rule-based heuristics.
    """
    # Get dataset name
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"status": "error", "rules": [], "message": "Dataset not found."}

    ds_name = (dataset.display_name or dataset.physical_name or "").split("/")[-1].replace(".csv", "")

    # Get column profiles from DB (no file I/O needed)
    profiles = _get_column_profiles(db, dataset_id)
    if not profiles:
        return {"status": "success", "rules": [], "message": "No column profiles found. Run DQ Scoring first."}

    # Try LLM first, fall back to rule-based
    raw_rules = _llm_recommendations(profiles, ds_name)
    method    = "llm"

    if not raw_rules:
        print(f"[ai_recommendations] LLM unavailable, using rule-based for dataset {dataset_id}")
        raw_rules = _rule_based_recommendations(profiles, ds_name)
        method    = "rule_based"

    valid_cols = {p["column"] for p in profiles}
    rules_out  = []

    for i, r in enumerate(raw_rules[:k]):
        # Only include rules for columns that actually exist
        col = r.get("column", "")
        if col and col not in valid_cols:
            continue
        rules_out.append({
            "id":             f"AI-REC-{i+1:03d}",
            "name":           r.get("name", ""),
            "type":           r.get("type", "Validity"),
            "column":         col,
            "condition":      r.get("condition", ""),
            "expectedImpact": "Medium",
            "confidence":     float(r.get("confidence", 0.80)) * 100,
            "reasoning":      r.get("reasoning", ""),
            "severity":       r.get("severity", "Medium"),
            "status":         "Pending",
        })

    return {"status": "success", "rules": rules_out, "method": method}


def simulate_ai_rule(db: Session, dataset_id: int, rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a candidate rule against the dataset and returns passRate + violations.
    Reads from Azure Blob directly — no local cache required.
    """
    df = _load_df_from_blob(db, dataset_id)

    if df is None or df.empty:
        return {"status": "success", "passRate": 0.0, "violations": 0, "totalRows": 0,
                "message": "Could not load dataset for simulation."}

    total_rows = len(df)
    dummy = SimpleNamespace(
        type      = str(rule.get("type")      or "Validity"),
        column    = str(rule.get("column")    or ""),
        condition = str(rule.get("condition") or ""),
    )

    try:
        mask, err = _apply_rule(df, dummy)
        if err:
            return {"status": "error", "message": err,
                    "passRate": 0.0, "violations": total_rows, "totalRows": total_rows}
        pass_rate  = round(float(mask.mean()) * 100, 2)
        violations = int((~mask).sum())
        return {"status": "success", "passRate": pass_rate, "violations": violations, "totalRows": total_rows}
    except Exception as e:
        return {"status": "error", "message": str(e),
                "passRate": 0.0, "violations": total_rows, "totalRows": total_rows}