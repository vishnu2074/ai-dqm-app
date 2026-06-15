"""
python-backend/app/services/ai_rule_generator.py

FIXED: Uses direct HTTP requests to Azure AI Foundry endpoint.
NO openai SDK import — that caused "Missing credentials" errors because
the SDK looks for AZURE_OPENAI_AD_TOKEN which we don't have.

Azure AI Foundry endpoint format:
  {endpoint}/models/{model}/chat/completions?api-version={version}

This is different from Azure OpenAI SDK which routes to:
  {endpoint}/openai/deployments/{model}/chat/completions
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import time as _time
import pandas as pd
import requests as _http
from app.services.llm_tracker import track_llm_call

_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")
_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")


def _llm(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """Direct HTTP call to Azure AI Foundry — no openai SDK."""
    if not _KEY or not _ENDPOINT:
        return None
    url = f"{_ENDPOINT}/chat/completions"
    _t0 = _time.time()
    try:
        r = _http.post(
            url,
            headers={"Content-Type": "application/json", "api-key": _KEY},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=45,
        )
        r.raise_for_status()
        body = r.json()
        content_out = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        track_llm_call(
            feature="dq_rules", model=_MODEL,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(_time.time() - _t0) * 1000,
            success=True, input_length=len(prompt), output_length=len(content_out or ""),
        )
        return content_out
    except _http.exceptions.ConnectionError as e:
        track_llm_call(feature="dq_rules", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type="ConnectionError", input_length=len(prompt))
        print(f"[ai_rule_generator] Connection error: {e}")
        return None
    except _http.exceptions.Timeout:
        track_llm_call(feature="dq_rules", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type="Timeout", input_length=len(prompt))
        print(f"[ai_rule_generator] Request timed out")
        return None
    except Exception as e:
        track_llm_call(feature="dq_rules", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000, success=False,
            error_type=type(e).__name__, input_length=len(prompt))
        print(f"[ai_rule_generator] LLM error: {e}")
        return None


def _rule_based_rules(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate DQ rules from dataframe statistics without LLM.
    Used when LLM is unavailable.
    """
    rules = []
    seen  = set()

    for col in df.columns:
        col_lower = col.lower()
        dtype_str = str(df[col].dtype)
        is_numeric = "int" in dtype_str or "float" in dtype_str
        null_pct   = df[col].isna().mean()
        uniq_pct   = df[col].nunique() / max(len(df), 1)

        # Uniqueness for ID columns
        if re.search(r"(_id|_key|_uuid|_pk)$", col, re.I) and uniq_pct < 0.95:
            key = f"unique_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Uniqueness Check",
                    "type":      "Uniqueness",
                    "column":    col,
                    "condition": f"{col}.is_unique",
                    "severity":  "High",
                    "reasoning": f'"{col}" is an identifier with {(1-uniq_pct)*100:.1f}% duplicate rate.',
                    "confidence": 0.90,
                })

        # Not-null for important columns
        if null_pct > 0.01 and re.search(
            r"(_id|_name|_email|_date|_amount|_status)$|^(id|name|email|date|amount|status)", col, re.I):
            key = f"notnull_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Not Null",
                    "type":      "Completeness",
                    "column":    col,
                    "condition": f"{col}.notna()",
                    "severity":  "Medium",
                    "reasoning": f'"{col}" has {null_pct*100:.1f}% null values.',
                    "confidence": 0.85,
                })

        # Email format
        if "email" in col_lower:
            key = f"email_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Email Format",
                    "type":      "Validity",
                    "column":    col,
                    "condition": f'{col}.str.contains("@", na=False)',
                    "severity":  "Medium",
                    "reasoning": f'Email column should contain @ symbol.',
                    "confidence": 0.88,
                })

        # Non-negative for financial/quantity columns
        if is_numeric and re.search(r"(amount|price|cost|salary|income|balance|quantity|count)", col, re.I):
            if df[col].dropna().lt(0).any():
                key = f"nonneg_{col}"
                if key not in seen:
                    seen.add(key)
                    rules.append({
                        "name":      f"{col} Non-Negative",
                        "type":      "Validity",
                        "column":    col,
                        "condition": f"{col} >= 0",
                        "severity":  "High",
                        "reasoning": f'"{col}" contains negative values ({df[col].lt(0).sum()} rows).',
                        "confidence": 0.87,
                    })

        # Allowed values for low-cardinality categoricals
        n_distinct = df[col].nunique()
        if (not is_numeric and 2 <= n_distinct <= 8 and
            re.search(r"(status|type|category|tier|level|gender|flag)", col, re.I)):
            allowed = [str(v) for v in df[col].dropna().unique().tolist()]
            key = f"enum_{col}"
            if key not in seen:
                seen.add(key)
                rules.append({
                    "name":      f"{col} Allowed Values",
                    "type":      "Validity",
                    "column":    col,
                    "condition": f"{col}.notna()",
                    "severity":  "Medium",
                    "reasoning": f'"{col}" has {n_distinct} distinct values: {", ".join(allowed[:5])}.',
                    "confidence": 0.80,
                })

    return rules[:8]


def generate_ai_rules(df: pd.DataFrame, k: int = 8) -> List[Dict[str, Any]]:
    """
    Generate DQ rules for a dataframe.
    Uses LLM if available, falls back to rule-based heuristics.
    Called by ai_recommendations.py.
    """
    # Build column profile for LLM prompt
    col_profiles = []
    for col in df.columns[:25]:  # cap at 25 cols to keep prompt size reasonable
        dtype_str  = str(df[col].dtype)
        null_pct   = round(df[col].isna().mean() * 100, 1)
        uniq_pct   = round(df[col].nunique() / max(len(df), 1) * 100, 1)
        n_distinct = df[col].nunique()
        col_profiles.append(
            f"  {col} ({dtype_str}) — nulls: {null_pct}%, uniqueness: {uniq_pct}%, distinct: {n_distinct}"
        )

    col_list = "\n".join(col_profiles)

    prompt = f"""You are a data quality expert. Analyze these column profiles and recommend {k} specific DQ validation rules.

Dataset: {len(df):,} rows × {len(df.columns)} columns
Columns:
{col_list}

For each rule provide:
- name: short descriptive name
- type: Completeness | Uniqueness | Validity | Consistency | Accuracy
- column: exact column name from the list
- condition: simple pandas condition (e.g. "{'{'}col{'}'} >= 0", "{'{'}col{'}'}.notna()", "{'{'}col{'}'}.str.contains('@')")
- severity: Critical | High | Medium | Low
- reasoning: one sentence why this matters
- confidence: 0.0-1.0

Return ONLY valid JSON, no markdown:
{{"rules": [{{"name":"...","type":"...","column":"...","condition":"...","severity":"...","reasoning":"...","confidence":0.85}}]}}"""

    raw = _llm(prompt, max_tokens=1500)

    if raw:
        try:
            raw    = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            rules  = parsed.get("rules", [])
            if rules:
                # Validate column names
                valid_cols = set(df.columns)
                return [r for r in rules if r.get("column","") in valid_cols][:k]
        except Exception as e:
            print(f"[ai_rule_generator] JSON parse error: {e}")

    # LLM unavailable or failed — use rule-based
    print("[ai_rule_generator] LLM unavailable, using rule-based fallback")
    return _rule_based_rules(df)[:k]