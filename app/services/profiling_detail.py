"""
python-backend/app/services/profiling_detail.py

CHANGES (v2.0 — Health Observatory Integration):
  1. _ai_dataset_description() — FIXED to use correct Azure AI Foundry auth
     (Bearer token + /v1/chat/completions endpoint)
  2. Each column now has BOTH:
       columnType  (NEW):  specific SQL/pandas type — INTEGER, FLOAT, VARCHAR, DATE,
                           EMAIL, ENUM, UUID, TIMESTAMP, DECIMAL, TEXT etc.
       typeClass   (unchanged): generic bucket — NUMERIC | CATEGORICAL | DATETIME | STRING
       semanticType (unchanged): LLM PascalCase label — CustomerID, EmailAddress etc.
  3. description field added to payload — consumed by Data Description tab
  4. NEW: Sensitivity classification per column (PII, Financial, Health, etc.)
  5. NEW: AI summary saved to profiling_runs.ai_summary column
  6. NEW: Sensitivity labels saved to column_profiles.sensitivity_label column
  7. NEW: AI descriptions saved to column_profiles.ai_description column
"""
from __future__ import annotations

import re
import math
import os
import json
try:
    import httpx
except ImportError:
    httpx = None  # optional HTTP client
import requests as _http
from datetime import timedelta, timezone
from typing import Any

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.models import Dataset, ProfilingRun
from app.services.dq_scores import (
    _load_dataframe_for_dataset,
    _fmt_ist,
    run_dq_scoring,
)

IST = timezone(timedelta(hours=5, minutes=30))

TOP_K              = 10
RARE_THRESHOLD_PCT = 1.0
OUTLIER_IQR_FACTOR = 1.5
OUTLIER_Z          = 3.0

_DATE_COL_RE = re.compile(
    r"(date|time|dt|day|month|year|created|updated|timestamp|_at|_on|since|until"
    r"|start|end|birth|dob|expir)", re.I)
_EMAIL_RE    = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE    = re.compile(r"^[\+\-\(\)\s\d]{7,20}$")
_UUID_RE     = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_US_DATE_RE  = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
_BOOL_RE     = re.compile(r"^(true|false|yes|no|1|0|y|n)$", re.I)
_ZIP_RE      = re.compile(r"^\d{5}(-\d{4})?$")

_ENTITY_GROUPS = {
    "Customer":    re.compile(r"customer|client|user|member|person|buyer|contact", re.I),
    "Transaction": re.compile(r"order|transaction|invoice|payment|purchase|sale|txn", re.I),
    "Product":     re.compile(r"product|item|sku|article|catalog|good", re.I),
    "Temporal":    re.compile(r"date|time|timestamp|created|updated|modified|_at$|_on$", re.I),
    "Financial":   re.compile(r"amount|price|cost|revenue|income|salary|fee|tax|total|balance", re.I),
    "Location":    re.compile(r"city|country|state|zip|postal|address|region|lat|lon|geo", re.I),
}

_PATTERN_CHECKS = [
    ("ISO_DATE",     re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("ISO_DATETIME", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")),
    ("US_DATE",      re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")),
    ("EMAIL",        _EMAIL_RE),
    ("PHONE",        _PHONE_RE),
    ("UUID",         _UUID_RE),
    ("INTEGER_ID",   re.compile(r"^\d+$")),
    ("BOOLEAN",      _BOOL_RE),
    ("ZIP_US",       _ZIP_RE),
    ("ALPHANUMERIC", re.compile(r"^[a-zA-Z0-9]+$")),
    ("FREE_TEXT",    re.compile(r"^.{20,}$")),
]

_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")


# ─── Sensitivity Classification ──────────────────────────────────────────────

_SENSITIVE_PATTERNS = {
    "PII": re.compile(
        r"(email|phone|mobile|cell|ssn|social.*security|passport|national.*id|"
        r"drivers.*license|dob|birth.*date|first.*name|last.*name|full.*name)", re.I
    ),
    "Financial": re.compile(
        r"(salary|income|bank.*account|credit.*card|balance|amount|price|cost|"
        r"revenue|payment|purchase|transaction)", re.I
    ),
    "Health": re.compile(
        r"(patient|diagnosis|medical|health|prescription|treatment|disease|symptom)", re.I
    ),
    "Authentication": re.compile(
        r"(password|secret|token|api.*key|credential|auth)", re.I
    ),
    "Location": re.compile(
        r"(address|latitude|longitude|gps|coordinates|zip.*code|postal|city|country)", re.I
    ),
}


def _classify_sensitivity(col: str, type_class: str, sample_values: list) -> str:
    """Classify column sensitivity based on name patterns and sample values."""
    # Check column name patterns first
    for label, pattern in _SENSITIVE_PATTERNS.items():
        if pattern.search(col):
            return label

    # Check sample values for sensitive patterns
    for val in sample_values[:20]:
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ('nan', 'null', 'none', ''):
            continue
        if _EMAIL_RE.match(val_str):
            return "PII"
        if _PHONE_RE.match(val_str):
            return "PII"
        if _UUID_RE.match(val_str) and any(x in col.lower() for x in ('id', 'key', 'token')):
            return "Authentication"

    return "Public"


# ─── DB Save Helpers ─────────────────────────────────────────────────────────

def _save_ai_summary_to_db(run_id: int, summary: str):
    """Save AI-generated dataset description to profiling_runs.ai_summary column."""
    if not summary:
        return
    try:
        from app.database import engine
        from sqlalchemy import text as sa_text
        with engine.connect() as conn:
            conn.execute(
                sa_text("UPDATE profiling_runs SET ai_summary = :summary WHERE id = :id"),
                {"summary": summary, "id": run_id}
            )
            conn.commit()
        print(f"[profiling] ✓ Saved AI summary for run {run_id} ({len(summary)} chars)")
    except Exception as e:
        print(f"[profiling] ✗ Failed to save AI summary: {e}")


def _save_sensitivity_to_db(profiling_run_id: int, column_name: str, sensitivity: str):
    """Save sensitivity label to column_profiles table for the given run."""
    try:
        from app.database import engine
        from sqlalchemy import text as sa_text
        with engine.connect() as conn:
            conn.execute(
                sa_text("""
                    UPDATE column_profiles
                    SET sensitivity_label = :label
                    WHERE profiling_run_id = :run_id AND column_name = :col_name
                """),
                {"label": sensitivity, "run_id": profiling_run_id, "col_name": column_name}
            )
            conn.commit()
    except Exception as e:
        print(f"[profiling] ✗ Failed to save sensitivity for {column_name}: {e}")


def _save_ai_description_to_db(profiling_run_id: int, column_name: str, description: str):
    """Save AI-generated column description to column_profiles table."""
    if not description:
        return
    try:
        from app.database import engine
        from sqlalchemy import text as sa_text
        with engine.connect() as conn:
            conn.execute(
                sa_text("""
                    UPDATE column_profiles
                    SET ai_description = :desc
                    WHERE profiling_run_id = :run_id AND column_name = :col_name
                """),
                {"desc": description, "run_id": profiling_run_id, "col_name": column_name}
            )
            conn.commit()
    except Exception as e:
        print(f"[profiling] ✗ Failed to save AI description for {column_name}: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(val) -> Any:
    if val is None: return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
    except (TypeError, ValueError): pass
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.floating): return float(val)
    return val


def _classify(col: str, series: pd.Series) -> str:
    """Generic typeClass: NUMERIC | CATEGORICAL | DATETIME | STRING"""
    if pd.api.types.is_bool_dtype(series):           return "CATEGORICAL"
    if pd.api.types.is_numeric_dtype(series):        return "NUMERIC"
    if pd.api.types.is_datetime64_any_dtype(series): return "DATETIME"
    if _DATE_COL_RE.search(col):
        sample = series.dropna().astype(str).head(50)
        if len(sample) > 0 and sum(1 for v in sample if _ISO_DATE_RE.match(v) or _US_DATE_RE.match(v)) / len(sample) >= 0.5:
            return "DATETIME"
    non_null = series.dropna()
    if series.dtype == object and len(non_null) > 0:
        if series.nunique(dropna=True) / max(len(non_null), 1) < 0.15 or series.nunique(dropna=True) <= 20:
            return "CATEGORICAL"
    return "STRING"


def _infer_column_type(col: str, series: pd.Series, type_class: str) -> str:
    """
    Specific SQL/pandas-level type — more precise than typeClass.
    Used in the Data Description table as 'Type' column.

    Examples:
      typeClass=NUMERIC   → INTEGER, FLOAT, DECIMAL, BOOLEAN
      typeClass=STRING    → VARCHAR, TEXT, UUID, EMAIL, PHONE, INTEGER_STR
      typeClass=DATETIME  → DATE, DATETIME, TIMESTAMP, TIME
      typeClass=CATEGORICAL → BOOLEAN, ENUM, CATEGORICAL
    """
    if type_class == "NUMERIC":
        if pd.api.types.is_bool_dtype(series):    return "BOOLEAN"
        if pd.api.types.is_integer_dtype(series): return "INTEGER"
        if pd.api.types.is_float_dtype(series):
            col_l = col.lower()
            if any(x in col_l for x in ("amount","price","cost","salary","fee","tax","balance","rate","percent")):
                return "DECIMAL"
            return "FLOAT"
        return "NUMERIC"

    if type_class == "DATETIME":
        col_l = col.lower()
        if "timestamp" in col_l or "_at" in col_l: return "TIMESTAMP"
        if "time" in col_l and "date" not in col_l: return "TIME"
        return "DATE"

    if type_class == "CATEGORICAL":
        if pd.api.types.is_bool_dtype(series): return "BOOLEAN"
        n_unique = series.nunique(dropna=True)
        if n_unique <= 2:  return "BOOLEAN"
        if n_unique <= 10: return "ENUM"
        return "CATEGORICAL"

    if type_class == "STRING":
        non_null = series.dropna().astype(str)
        sample   = non_null.head(100)
        n        = len(sample)
        if n > 5:
            if sum(1 for v in sample if _EMAIL_RE.match(v)) / n > 0.7: return "EMAIL"
            if sum(1 for v in sample if _PHONE_RE.match(v)) / n > 0.7: return "PHONE"
            if sum(1 for v in sample if _UUID_RE.match(v))  / n > 0.7: return "UUID"
            if sum(1 for v in sample if re.match(r"^\d+$", v)) / n > 0.8: return "INTEGER_STR"
        avg_len = non_null.str.len().mean() if len(non_null) > 0 else 0
        if avg_len > 80: return "TEXT"
        return "VARCHAR"

    return type_class


def _histogram(series: pd.Series, bins: int = 20) -> list:
    try:
        nn = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(nn) < 2: return []
        counts, edges = np.histogram(nn, bins=bins)
        return [{"bin_start": round(float(edges[i]),4), "bin_end": round(float(edges[i+1]),4), "count": int(counts[i])}
                for i in range(len(counts))]
    except Exception: return []


def _to_shape(val: str) -> str:
    out, prev, run = [], None, 0
    for ch in val[:20]:
        c = "N" if ch.isdigit() else ("A" if ch.isalpha() else ch)
        if c == prev: run += 1
        else:
            if prev: out.append(f"{prev}{run}" if run > 1 else prev)
            prev, run = c, 1
    if prev: out.append(f"{prev}{run}" if run > 1 else prev)
    return "".join(out)


def _detect_patterns(non_null: pd.Series) -> list:
    sample = non_null.astype(str).head(2000)
    counts: dict[str, int] = {}
    for val in sample:
        matched = False
        for name, rx in _PATTERN_CHECKS:
            if rx.match(val):
                counts[name] = counts.get(name, 0) + 1
                matched = True; break
        if not matched:
            counts["OTHER"] = counts.get("OTHER", 0) + 1
    total = len(sample)
    if total == 0: return []
    return sorted([{"pattern":k,"count":v,"pct":round(v/total*100,2)} for k,v in counts.items()], key=lambda x:-x["count"])


def _severity(pct: float) -> str:
    if pct >= 10: return "CRITICAL"
    if pct >= 5:  return "HIGH"
    if pct >= 1:  return "MEDIUM"
    return "LOW"


def _numeric_outliers(num: pd.Series) -> list:
    results = []
    if len(num) < 5: return results
    q1, q3 = float(np.percentile(num,25)), float(np.percentile(num,75))
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1-OUTLIER_IQR_FACTOR*iqr, q3+OUTLIER_IQR_FACTOR*iqr
        mask   = (num < lo) | (num > hi)
        cnt    = int(mask.sum())
        if cnt:
            pct = round(cnt/len(num)*100, 2)
            results.append({"type":"IQR","count":cnt,"pct":pct,"lowerBound":round(lo,4),"upperBound":round(hi,4),
                "examples":[round(float(v),4) for v in sorted(num[mask].values)[:5]],"severity":_severity(pct)})
    mean, std = float(num.mean()), float(num.std())
    if std > 0:
        mask = ((num-mean)/std).abs() > OUTLIER_Z
        cnt  = int(mask.sum())
        if cnt:
            pct = round(cnt/len(num)*100, 2)
            results.append({"type":"Z-Score","count":cnt,"pct":pct,
                "lowerBound":round(mean-OUTLIER_Z*std,4),"upperBound":round(mean+OUTLIER_Z*std,4),
                "examples":[round(float(v),4) for v in sorted(num[mask].values)[:5]],"severity":_severity(pct)})
    return results


def _cat_outliers(vc: pd.Series, total: int) -> list:
    if total == 0 or len(vc) == 0: return []
    rare = vc[vc < total*RARE_THRESHOLD_PCT/100]
    if len(rare) == 0: return []
    cnt = int(rare.sum()); pct = round(cnt/total*100, 2)
    return [{"type":"Rare Categories","count":cnt,"pct":pct,
             "examples":[str(v) for v in rare.index[:5].tolist()],"severity":_severity(pct),
             "detail":f"{len(rare)} categories each appearing < {RARE_THRESHOLD_PCT}% of rows"}]


def _date_outliers(dates: pd.Series, today) -> list:
    results = []
    try:
        future = dates[dates > today]
        if len(future):
            pct = round(len(future)/len(dates)*100, 2)
            results.append({"type":"Future Dates","count":int(len(future)),"pct":pct,
                "examples":[str(v.date()) for v in sorted(future)[:5]],"severity":_severity(pct)})
    except Exception: pass
    try:
        anc = pd.Timestamp("1900-01-01",tz=dates.dt.tz) if dates.dt.tz else pd.Timestamp("1900-01-01")
        ancient = dates[dates < anc]
        if len(ancient):
            results.append({"type":"Ancient Dates","count":int(len(ancient)),
                "pct":round(len(ancient)/len(dates)*100,2),
                "examples":[str(v.date()) for v in sorted(ancient)[:5]],"severity":"HIGH"})
    except Exception: pass
    try:
        gaps = dates.sort_values().diff().dt.days.dropna()
        if len(gaps) >= 5:
            med = float(gaps.median())
            if med > 0:
                big = gaps[gaps > 10*med]
                if len(big):
                    results.append({"type":"Temporal Gaps","count":int(len(big)),
                        "pct":round(len(big)/len(gaps)*100,2),
                        "examples":[f"{v:.0f} days" for v in sorted(big.values,reverse=True)[:5]],
                        "severity":"MEDIUM"})
    except Exception: pass
    return results


# ─── AI: semantic labels (Anthropic API) ─────────────────────────────────────

def _ai_semantic_labels(columns_info: list[dict]) -> dict[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key: return {}
    col_descs = []
    for c in columns_info:
        samples_str = ", ".join(str(v) for v in c["sample_values"][:10]) if c["sample_values"] else "N/A"
        col_descs.append(f'- Column: "{c["name"]}" | DataType: {c["type_class"]} | Samples: [{samples_str}] | NullPct: {c["null_pct"]:.1f}% | DistinctPct: {c["distinct_pct"]:.1f}%')
    prompt = f"""You are a data cataloging expert. For each column, assign a single precise semantic label.
Rules: PascalCase 1-3 words. Be specific (CustomerID not just ID). Never purely generic labels.
Respond ONLY with valid JSON: {{"column_name": "SemanticLabel", ...}}

Columns:
{chr(10).join(col_descs)}"""
    try:
        resp = httpx.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-haiku-4-5-20251001","max_tokens":512,"messages":[{"role":"user","content":prompt}]},
            timeout=30.0)
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        if raw.startswith("```"): raw = re.sub(r"^```[a-z]*\n?","",raw); raw = re.sub(r"\n?```$","",raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[profiling] AI semantic labeling failed (non-fatal): {e}")
        return {}


# ─── AI: dataset description (Azure AI Foundry — FIXED) ──────────────────────

def _ai_dataset_description(ds_name: str, total_rows: int, columns: list[dict], summary: dict) -> str:
    """
    LLM-generated natural language description of the dataset.
    FIXED: Uses get_llm_client() from app.main with correct Bearer auth.
    Falls back to raw HTTP with correct auth, then to structured template.
    """
    # ── Priority 1: Use centralized LLM client (correct auth) ────────────────
    try:
        from app.main import get_llm_client
        client = get_llm_client()

        if client:
            col_lines = []
            for c in columns[:20]:
                sem   = c.get("entity",{}).get("semanticType","")
                ctype = c.get("columnType", c.get("typeClass",""))
                col_lines.append(
                    f"  {c['columnName']} ({ctype}, semantic={sem}, "
                    f"nulls={c.get('nullPct',0):.1f}%, unique={c.get('distinctPct',0):.0f}%)"
                )

            prompt = (
                f"You are a data catalog expert writing a detailed description for business and technical users.\n"
                f'Dataset: "{ds_name}"\n'
                f"Rows: {total_rows:,} | Columns: {len(columns)}\n"
                f"Completeness: {summary.get('avgCompleteness',0):.1f}% | Uniqueness: {summary.get('avgUniqueness',0):.1f}%\n"
                f"Types: {summary.get('numericColumns',0)} numeric, {summary.get('categoricalColumns',0)} categorical, "
                f"{summary.get('datetimeColumns',0)} datetime, {summary.get('stringColumns',0)} string\n"
                f"Columns:\n{chr(10).join(col_lines)}\n\n"
                f"Write a professional dataset description covering:\n"
                f"- What this dataset represents and its business domain (2-3 sentences)\n"
                f"- Key entities or concepts captured (1-2 sentences)\n"
                f"- Data structure highlights — identifiers, measures, time coverage (2 sentences)\n"
                f"- Data quality characteristics — completeness, uniqueness (1-2 sentences)\n"
                f"- Suggested use cases (2-3 specific bullet points starting with •)\n"
                f"Be specific to this dataset. No markdown headers. Plain paragraphs and bullets only."
            )

            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct"))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )

            raw = response.choices[0].message.content
            if raw and len(raw.strip()) > 80:
                print(f"[profiling] ✓ AI description generated via LLM client ({len(raw)} chars)")
                return raw.strip()
    except Exception as e:
        print(f"[profiling] LLM client description failed (non-fatal): {e}")

    # ── Priority 2: Fallback to raw HTTP with CORRECT auth (Bearer token) ────
    if _KEY and _ENDPOINT:
        try:
            col_lines = []
            for c in columns[:20]:
                sem   = c.get("entity",{}).get("semanticType","")
                ctype = c.get("columnType", c.get("typeClass",""))
                col_lines.append(
                    f"  {c['columnName']} ({ctype}, semantic={sem}, "
                    f"nulls={c.get('nullPct',0):.1f}%, unique={c.get('distinctPct',0):.0f}%)"
                )

            prompt = (
                f"You are a data catalog expert. Describe dataset '{ds_name}' "
                f"({total_rows:,} rows, {len(columns)} columns). "
                f"Cover: business domain, key entities, data structure, quality, use cases. "
                f"Columns: {', '.join(c['columnName'] for c in columns[:10])}. "
                f"Plain paragraphs only, no markdown."
            )

            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct"))

            # FIXED: Use /v1/chat/completions with Bearer auth (Azure AI Foundry format)
            r = _http.post(
                f"{_ENDPOINT}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_KEY}",
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 600,
                },
                timeout=35,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            if raw and len(raw.strip()) > 80:
                print(f"[profiling] ✓ AI description generated via HTTP ({len(raw)} chars)")
                return raw.strip()
        except Exception as e:
            print(f"[profiling] HTTP AI description failed (non-fatal): {e}")

    # ── Priority 3: Fallback to template ─────────────────────────────────────
    return _fallback_description(ds_name, total_rows, columns, summary)


def _fallback_description(ds_name: str, total_rows: int, columns: list[dict], summary: dict) -> str:
    num_c = summary.get("numericColumns",0); cat_c = summary.get("categoricalColumns",0)
    dt_c  = summary.get("datetimeColumns",0); str_c = summary.get("stringColumns",0)
    comp  = summary.get("avgCompleteness",0); uniq  = summary.get("avgUniqueness",0)
    n_cols = len(columns)
    all_cols = " ".join(c["columnName"].lower() for c in columns)
    domain = "general-purpose"
    if any(x in all_cols for x in ("customer","order","product","transaction")): domain = "transactional / customer data"
    elif any(x in all_cols for x in ("patient","admission","diagnosis","lab")):  domain = "healthcare / clinical"
    elif any(x in all_cols for x in ("account","loan","balance","card","credit")): domain = "financial / banking"
    key_cols = ", ".join(c["columnName"] for c in columns[:5])
    return (
        f"This dataset, '{ds_name}', contains {total_rows:,} records across {n_cols} columns "
        f"representing {domain}. The schema includes {num_c} numeric measure(s), {cat_c} categorical "
        f"attribute(s), {dt_c} date/time field(s), and {str_c} text/string column(s).\n\n"
        f"Overall data completeness is {comp:.1f}% with an average column uniqueness of {uniq:.1f}%. "
        f"Key columns include: {key_cols}.\n\n"
        f"• Use for analytical reporting and KPI dashboards\n"
        f"• Integrate with related datasets via shared identifier columns\n"
        f"• Apply DQ rules to enforce data validity and completeness standards"
    )


# ─── Entity & pattern helpers ─────────────────────────────────────────────────

def _entity_profile(col: str, ctype: str, series: pd.Series) -> dict:
    col_l    = col.lower()
    non_null = series.dropna().astype(str)
    sample   = non_null.head(100)
    n        = len(sample)
    sem_type, role = "Text", "Attribute"
    if n > 10:
        if sum(1 for v in sample if _EMAIL_RE.match(v)) / n > 0.7:   sem_type, role = "Email", "Attribute"
        elif sum(1 for v in sample if _PHONE_RE.match(v)) / n > 0.7: sem_type, role = "Phone", "Attribute"
        elif sum(1 for v in sample if _UUID_RE.match(v)) / n > 0.7:  sem_type, role = "UUID", "Identifier"
        elif sum(1 for v in sample if _BOOL_RE.match(v)) / n > 0.8:  sem_type, role = "Flag", "Attribute"
    if sem_type == "Text":
        if ctype == "DATETIME":   sem_type, role = "Date", "Timestamp"
        elif re.search(r"(_id|_key|_code|num|number|_no)$", col_l):
            ur = series.nunique(dropna=True) / max(len(series.dropna()), 1)
            sem_type, role = "ID", ("Identifier" if ur > 0.9 else "Attribute")
        elif re.search(r"(amount|price|cost|revenue|salary|income|fee|tax|total|balance)", col_l):
            sem_type, role = "Amount", "Measure"
        elif re.search(r"(flag|is_|has_|active|enabled|status)", col_l): sem_type, role = "Flag", "Attribute"
        elif ctype == "NUMERIC":      sem_type, role = "Numeric", "Measure"
        elif ctype == "CATEGORICAL":  sem_type, role = "Code", "Attribute"
    nn = series.dropna()
    ur = series.nunique(dropna=True) / max(len(nn), 1)
    pk = 0.95 if ur>0.99 and series.isna().sum()==0 else 0.75 if ur>0.95 else 0.5 if ur>0.85 else 0.0
    fk = 0.8 if re.search(r"(_id|_key|_code|_ref|_fk)$", col_l) and 0.1 < ur < 0.9 else 0.0
    group = next((g for g,p in _ENTITY_GROUPS.items() if p.search(col)), "Other")
    return {"semanticType":sem_type,"role":role,"entityGroup":group,
            "pkScore":round(pk,2),"fkScore":round(fk,2),"isNullable":bool(series.isna().any())}


def _pattern_profile(non_null: pd.Series) -> dict:
    if len(non_null) == 0:
        return {"dominantPattern":None,"consistencyPct":None,"patternsBreakdown":[],
                "mixedFormatFlag":False,"invalidCount":0,"invalidPct":0.0,"shapeSummary":[]}
    breakdown = _detect_patterns(non_null)
    dominant  = breakdown[0] if breakdown else None
    invalid   = next((b["count"] for b in breakdown if b["pattern"]=="OTHER"), 0)
    mixed     = len(breakdown)>2 and sum(b["count"] for b in breakdown[1:])/max(len(non_null),1)>0.05
    shapes: dict[str,list] = {}
    for val in non_null.astype(str).head(500):
        sh = _to_shape(val); shapes.setdefault(sh,[]).append(val)
    shape_summary = sorted([{"shape":sh,"count":len(ex),"example":ex[0]} for sh,ex in shapes.items()],
                            key=lambda x:-x["count"])[:5]
    return {"dominantPattern":dominant["pattern"] if dominant else None,
            "consistencyPct":dominant["pct"] if dominant else None,
            "patternsBreakdown":breakdown,"mixedFormatFlag":mixed,
            "invalidCount":invalid,"invalidPct":round(invalid/max(len(non_null),1)*100,2),
            "shapeSummary":shape_summary}


# ─── Per-column profiling ─────────────────────────────────────────────────────

def _profile_column(col: str, ctype: str, series: pd.Series, total_rows: int) -> dict:
    null_count = int(series.isna().sum())
    null_pct   = round(null_count/total_rows*100, 2) if total_rows else 0.0
    non_null   = series.dropna()
    distinct   = int(series.nunique(dropna=True))
    count      = total_rows - null_count
    base: dict = {
        "columnName":     col,
        "typeClass":      ctype,                              # generic bucket
        "columnType":     _infer_column_type(col, series, ctype),  # specific SQL type
        "count":          count,
        "nullCount":      null_count, "nullPct": null_pct,
        "distinctCount":  distinct,
        "distinctPct":    _safe(distinct/max(count,1)*100),
        "duplicateCount": max(0, count-distinct),
        "outliers":       [],
    }
    if ctype == "NUMERIC":
        num = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(num) >= 2:
            q25,q75 = float(np.percentile(num,25)), float(np.percentile(num,75))
            base.update({"min":_safe(num.min()),"max":_safe(num.max()),"mean":_safe(num.mean()),
                "median":_safe(num.median()),"std":_safe(num.std()),"variance":_safe(num.var()),
                "sum":_safe(num.sum()),"range":_safe(float(num.max())-float(num.min())),
                "p05":_safe(np.percentile(num,5)),"p25":_safe(q25),
                "p50":_safe(np.percentile(num,50)),"p75":_safe(q75),
                "p95":_safe(np.percentile(num,95)),"p99":_safe(np.percentile(num,99)),
                "iqr":_safe(q75-q25),"skewness":_safe(float(num.skew())),
                "kurtosis":_safe(float(num.kurtosis())),"histogram":_histogram(num)})
            base["outliers"] = _numeric_outliers(num)
    elif ctype == "CATEGORICAL":
        vc = non_null.value_counts(); total_cat = len(non_null)
        probs = vc/total_cat if total_cat>0 and len(vc)>0 else pd.Series(dtype=float)
        base.update({"entropy":_safe(-float((probs*np.log2(probs+1e-12)).sum())) if len(probs) else None,
            "mostFrequentValue":str(vc.index[0]) if len(vc) else None,
            "mostFrequentCount":int(vc.iloc[0]) if len(vc) else None,
            "rareValueCount":int((vc<total_cat*RARE_THRESHOLD_PCT/100).sum()),
            "topKValues":[{"value":str(v),"count":int(c),"pct":round(c/max(total_cat,1)*100,2)}
                          for v,c in zip(vc.head(TOP_K).index,vc.head(TOP_K).values)]})
        base["outliers"] = _cat_outliers(vc, total_cat)
    elif ctype == "DATETIME":
        try:
            dates = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dates) >= 2:
                gaps  = dates.sort_values().diff().dt.days.dropna()
                today = pd.Timestamp.now(tz=dates.dt.tz) if dates.dt.tz else pd.Timestamp.now()
                base.update({"minDate":str(dates.min().date()),"maxDate":str(dates.max().date()),
                    "dateRangeDays":int((dates.max()-dates.min()).days),
                    "avgGapDays":_safe(gaps.mean()),"maxGapDays":_safe(gaps.max()),
                    "futureDateCount":int((dates>today).sum()),"nullDateCount":null_count})
                base["outliers"] = _date_outliers(dates, today)
        except Exception: pass
    elif ctype == "STRING":
        lengths = non_null.astype(str).str.len()
        base.update({"minLength":_safe(lengths.min()) if len(lengths) else None,
            "maxLength":_safe(lengths.max()) if len(lengths) else None,
            "avgLength":_safe(lengths.mean()) if len(lengths) else None,
            "lengthStd":_safe(lengths.std()) if len(lengths) else None,
            "regexPatterns":_detect_patterns(non_null)})
    base["entity"]  = _entity_profile(col, ctype, series)
    base["pattern"] = _pattern_profile(non_null)

    # ── NEW: Add sensitivity classification ────────────────────────────────────
    sample_vals = non_null.head(20).tolist() if len(non_null) > 0 else []
    base["sensitivity"] = _classify_sensitivity(col, ctype, sample_vals)

    return base


def _dataset_summary(columns: list, total_rows: int) -> dict:
    n = max(len(columns), 1)
    return {
        "totalRows":          total_rows,
        "totalColumns":       len(columns),
        "numericColumns":     sum(1 for c in columns if c["typeClass"]=="NUMERIC"),
        "categoricalColumns": sum(1 for c in columns if c["typeClass"]=="CATEGORICAL"),
        "datetimeColumns":    sum(1 for c in columns if c["typeClass"]=="DATETIME"),
        "stringColumns":      sum(1 for c in columns if c["typeClass"]=="STRING"),
        "avgCompleteness":    round(sum(100-(c.get("nullPct")or 0) for c in columns)/n, 2),
        "avgUniqueness":      round(sum(c.get("distinctPct")or 0 for c in columns)/n, 2),
        "pctColsWithNulls":   round(sum(1 for c in columns if (c.get("nullCount")or 0)>0)/n*100, 1),
    }


# ─── Build profile payload ────────────────────────────────────────────────────

def _build_profile_payload(run: ProfilingRun, df: pd.DataFrame, total_runs: int, ds_name: str) -> dict:
    total_rows = len(df)
    columns    = [_profile_column(col, _classify(col, df[col]), df[col], total_rows) for col in df.columns]

    # AI semantic labels — batch Anthropic call
    columns_info = []
    for col_data in columns:
        col_name = col_data["columnName"]
        non_null = df[col_name].dropna()
        columns_info.append({"name":col_name,"type_class":col_data["typeClass"],
            "sample_values":[str(v) for v in non_null.head(10).tolist()],
            "null_pct":col_data["nullPct"],"distinct_pct":col_data.get("distinctPct") or 0.0})
    ai_labels = _ai_semantic_labels(columns_info)
    for col_data in columns:
        col_name = col_data["columnName"]
        if col_name in ai_labels and ai_labels[col_name]:
            col_data["entity"]["semanticType"] = ai_labels[col_name]

    summary     = _dataset_summary(columns, total_rows)
    description = _ai_dataset_description(ds_name, total_rows, columns, summary)

    return {
        "status":         "ok",
        "runId":          run.id,
        "runTimestamp":   _fmt_ist(run.timestamp),
        "runType":        "FULL" if run.is_full_scan else "INCREMENTAL",
        "deltaRows":      run.delta_rows or 0,
        "totalRuns":      total_runs,
        "datasetName":    ds_name,
        "totalRows":      total_rows,
        "totalColumns":   len(df.columns),
        "columns":        columns,
        "datasetSummary": summary,
        "description":    description,
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def run_detail_profiling(db: Session, dataset_id: int) -> dict:
    run = run_dq_scoring(db, dataset_id)
    df  = _load_dataframe_for_dataset(db, dataset_id)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    raw = (dataset.display_name or dataset.physical_name or "") if dataset else ""
    ds_name = raw.split("/")[-1].split("\\")[-1]
    total_runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id==dataset_id, ProfilingRun.status=="COMPLETED").count()

    payload = _build_profile_payload(run, df, total_runs, ds_name)

    # ── Save AI summary to profiling_runs table ───────────────────────────────
    description = payload.get("description", "")
    if description and len(description) > 50:
        _save_ai_summary_to_db(run.id, description)

    # ── Save sensitivity labels + AI descriptions to column_profiles ──────────
    for col_data in payload.get("columns", []):
        col_name = col_data.get("columnName")
        if not col_name:
            continue

        # Save sensitivity label
        sensitivity = col_data.get("sensitivity", "Public")
        _save_sensitivity_to_db(run.id, col_name, sensitivity)

        # Generate and save per-column AI description
        entity = col_data.get("entity", {})
        sem_type = entity.get("semanticType", col_data.get("columnType", "Unknown"))
        type_class = col_data.get("typeClass", "Unknown")
        null_pct = col_data.get("nullPct", 0)
        distinct_pct = col_data.get("distinctPct", 0)

        col_desc = (
            f"Column '{col_name}' is of type {type_class} (semantic: {sem_type}). "
            f"Completeness: {100 - null_pct:.1f}%, Uniqueness: {distinct_pct:.0f}%. "
        )
        if null_pct > 0:
            col_desc += f"Contains {col_data.get('nullCount', 0)} null values. "
        if col_data.get("distinctCount", 0) > 0:
            col_desc += f"Has {col_data.get('distinctCount', 0)} distinct values. "

        _save_ai_description_to_db(run.id, col_name, col_desc)

    return payload


def get_detail_profile(db: Session, dataset_id: int) -> dict:
    run = (db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id==dataset_id, ProfilingRun.status=="COMPLETED")
        .order_by(ProfilingRun.id.desc()).first())
    if not run: return {"status": "NO_DATA"}
    df  = _load_dataframe_for_dataset(db, dataset_id)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    raw = (dataset.display_name or dataset.physical_name or "") if dataset else ""
    ds_name = raw.split("/")[-1].split("\\")[-1]
    total_runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id==dataset_id, ProfilingRun.status=="COMPLETED").count()

    payload = _build_profile_payload(run, df, total_runs, ds_name)

    # ── Save AI summary if not already saved ──────────────────────────────────
    description = payload.get("description", "")
    if description and len(description) > 50:
        # Check if already saved
        if not getattr(run, 'ai_summary', None):
            _save_ai_summary_to_db(run.id, description)

    # ── Save sensitivity labels + AI descriptions ─────────────────────────────
    for col_data in payload.get("columns", []):
        col_name = col_data.get("columnName")
        if not col_name:
            continue

        sensitivity = col_data.get("sensitivity", "Public")
        _save_sensitivity_to_db(run.id, col_name, sensitivity)

        entity = col_data.get("entity", {})
        sem_type = entity.get("semanticType", col_data.get("columnType", "Unknown"))
        type_class = col_data.get("typeClass", "Unknown")
        null_pct = col_data.get("nullPct", 0)
        distinct_pct = col_data.get("distinctPct", 0)

        col_desc = (
            f"Column '{col_name}' is of type {type_class} (semantic: {sem_type}). "
            f"Completeness: {100 - null_pct:.1f}%, Uniqueness: {distinct_pct:.0f}%. "
        )
        if null_pct > 0:
            col_desc += f"Contains {col_data.get('nullCount', 0)} null values. "
        if col_data.get("distinctCount", 0) > 0:
            col_desc += f"Has {col_data.get('distinctCount', 0)} distinct values. "

        _save_ai_description_to_db(run.id, col_name, col_desc)

    return payload