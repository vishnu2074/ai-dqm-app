# policy_suggestions_routes.py — Uses REAL data from database, no hardcoding
 
from __future__ import annotations
 
import hashlib

import json

import logging

import os

import uuid

from collections import defaultdict

from datetime import datetime, timezone, timedelta

from typing import Any, Dict, List, Optional, Set, Tuple
 
import time as _time
import requests as _http

from fastapi import APIRouter, Depends, HTTPException, Request

from sqlalchemy import desc, func, text

from sqlalchemy.orm import Session
 
from app.database import get_db

from app.models import Dataset, QualityCheck, ProfilingRun, DQRule, ColumnProfile
 
logger = logging.getLogger(__name__)
 
policy_suggestions_router = APIRouter()
 
# ─── Azure AI Foundry Configuration ──────────────────────────────────────────
 
_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")

_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")

_MODEL    = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")

_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

from app.services.llm_tracker import track_llm_call, status_code_suffix
 
 
def _llm(prompt: str, max_tokens: int = 1500) -> Optional[str]:

    """Call Azure AI Foundry LLM for policy generation."""

    if not _KEY or not _ENDPOINT:

        logger.warning("[policy_suggestions] LLM skipped — credentials not set")

        return None

    url = f"{_ENDPOINT}/models/chat/completions"

    headers = {"Content-Type": "application/json", "api-key": _KEY}

    payload = {

        "model": _MODEL,

        "messages": [{"role": "user", "content": prompt}],

        "temperature": 0.3,

        "max_tokens": max_tokens,

    }

    _t0 = _time.time()

    try:

        r = _http.post(url, headers=headers, json=payload,

                       params={"api-version": _VERSION}, timeout=40)

        r.raise_for_status()

        body = r.json()

        out = body["choices"][0]["message"]["content"]

        usage = body.get("usage", {})

        track_llm_call(
            feature="policy", model=_MODEL,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(_time.time() - _t0) * 1000,
            success=True, input_length=len(prompt), output_length=len(out or ""),
        )

        return out

    except Exception as e:

        track_llm_call(
            feature="policy", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000,
            success=False, error_type=status_code_suffix(e), input_length=len(prompt),
        )

        logger.error(f"[policy_suggestions] LLM error: {e}")

        return None
 
 
# ─── State Management ─────────────────────────────────────────────────────────
 
_dismissed: Set[str] = set()

_adopted: Set[str] = set()
 
_suggestions_cache: List[Dict] = []

_cache_expires_at: Optional[datetime] = None

_cache_data_hash: Optional[str] = None

_CACHE_TTL_SECONDS = 300  # 5 minutes — longer TTL avoids constant regeneration
 
 
# ─── Check Type Mappings ──────────────────────────────────────────────────────
 
_CHECK_TO_POLICY_TYPE: Dict[str, str] = {

    "ALL_NULLS": "Quality", "HIGH_NULL_RATE": "Quality",

    "CONSTANT_COLUMN": "Quality", "NEAR_CONSTANT_COLUMN": "Quality",

    "STATISTICAL_OUTLIERS": "Quality", "ZERO_DOMINATED_COLUMN": "Quality",

    "HIGH_DUPLICATE_RATE": "Compliance", "DUPLICATE_ROWS": "Compliance",

    "DUPLICATE_TIMESTAMPS": "Compliance", "INVALID_EMAIL_FORMAT": "Privacy",

    "SUSPICIOUS_WHITESPACE": "Quality", "MIXED_CASE_INCONSISTENCY": "Quality",

    "UNEXPECTED_NEGATIVE_VALUES": "Quality", "FIXED_LENGTH_VIOLATION": "Compliance",

    "UNPARSEABLE_DATES": "Quality", "FUTURE_DATES": "Quality",

    "ANCIENT_DATES": "Quality", "STALE_DATA": "Retention",

    "TEMPORAL_GAPS": "Quality", "WEEKEND_BUSINESS_DATES": "Compliance",

    "EMPTY_DATASET": "Governance", "SUSPICIOUSLY_FEW_ROWS": "Governance",

}
 
_CHECK_TO_RECOMMENDED_ACTION: Dict[str, str] = {

    "ALL_NULLS": "create_dq_rule:NOT NULL validation",

    "HIGH_NULL_RATE": "create_dq_rule:null rate threshold rule",

    "CONSTANT_COLUMN": "apply_validation:constant column detection",

    "NEAR_CONSTANT_COLUMN": "apply_validation:near-constant column alert",

    "STATISTICAL_OUTLIERS": "create_dq_rule:outlier detection",

    "ZERO_DOMINATED_COLUMN": "create_dq_rule:zero rate threshold",

    "HIGH_DUPLICATE_RATE": "create_dq_rule:uniqueness check",

    "DUPLICATE_ROWS": "create_dq_rule:deduplication check",

    "DUPLICATE_TIMESTAMPS": "create_dq_rule:timestamp uniqueness",

    "INVALID_EMAIL_FORMAT": "create_dq_rule:email validation",

    "STALE_DATA": "apply_validation:freshness SLA",

    "EMPTY_DATASET": "apply_validation:row count threshold",

}
 
_CHECK_LABELS: Dict[str, str] = {

    "ALL_NULLS": "All-Null Columns", "HIGH_NULL_RATE": "High Null Rate",

    "CONSTANT_COLUMN": "Constant Columns", "STATISTICAL_OUTLIERS": "Statistical Outliers",

    "HIGH_DUPLICATE_RATE": "High Duplicate Rate", "DUPLICATE_ROWS": "Duplicate Rows",

    "INVALID_EMAIL_FORMAT": "Invalid Email Format", "STALE_DATA": "Stale Data",

    "EMPTY_DATASET": "Empty Datasets",

}
 
 
# ─── Data Collection ──────────────────────────────────────────────────────────
 
def _get_anomalies_from_db(db: Session) -> List[Dict]:

    """Collect anomalies from temporal_checks and quality_checks tables."""

    anomalies = []
 
    try:

        result = db.execute(text("""

            SELECT

                tc.id, tc.check_type, tc.severity, tc.status,

                tc.description, tc.violation_count, tc.column_name,

                tc.created_at, pr.dataset_id,

                COALESCE(d.display_name, d.physical_name, 'Unknown') as dataset_name

            FROM temporal_checks tc

            LEFT JOIN profiling_runs pr ON tc.profiling_run_id = pr.id

            LEFT JOIN datasets d ON pr.dataset_id = d.id

            WHERE tc.status IN ('open', 'investigating', 'active')

            ORDER BY tc.created_at DESC

            LIMIT 100

        """))

        for row in result:

            anomalies.append({

                "id": row[0], "check_type": row[1], "severity": row[2],

                "status": row[3], "description": row[4], "violation_count": row[5] or 0,

                "column_name": row[6] or "unknown", "created_at": row[7],

                "dataset_id": row[8], "dataset_name": row[9] or "Unknown Dataset"

            })

        logger.info(f"[suggestions] Found {len(anomalies)} anomalies from temporal_checks")

    except Exception as e:

        logger.warning(f"Could not read temporal_checks: {e}")
 
    # Supplement from quality_checks if we have fewer than 10

    if len(anomalies) < 10:

        try:

            result = db.execute(text("""

                SELECT

                    qc.id, qc.check_type, qc.severity, qc.status,

                    qc.description, qc.violation_count, qc.column_name,

                    qc.created_at, qc.dataset_id,

                    COALESCE(d.display_name, d.physical_name, 'Unknown') as dataset_name

                FROM quality_checks qc

                LEFT JOIN datasets d ON qc.dataset_id = d.id

                WHERE qc.status IN ('open', 'investigating', 'active', 'failed')

                ORDER BY qc.created_at DESC

                LIMIT 100

            """))

            existing_ids = {a["id"] for a in anomalies}

            for row in result:

                if row[0] not in existing_ids:

                    anomalies.append({

                        "id": row[0], "check_type": row[1], "severity": row[2],

                        "status": row[3], "description": row[4], "violation_count": row[5] or 0,

                        "column_name": row[6] or "unknown", "created_at": row[7],

                        "dataset_id": row[8], "dataset_name": row[9] or "Unknown Dataset"

                    })

            logger.info(f"[suggestions] Found {len(anomalies)} total anomalies after quality_checks supplement")

        except Exception as e:

            logger.warning(f"Could not read quality_checks: {e}")
 
    return anomalies
 
 
def _get_dataset_quality_scores(db: Session) -> List[Dict]:

    """Get dataset quality scores for identifying low-quality datasets."""

    scores = []

    try:

        result = db.execute(text("""

            SELECT

                d.id, COALESCE(d.display_name, d.physical_name, 'Unknown') as name,

                AVG(dqs.overall_score) as avg_score,

                COUNT(dqs.id) as run_count,

                MAX(dqs.calculated_at) as last_run

            FROM datasets d

            LEFT JOIN dataset_quality_scores dqs ON d.id = dqs.dataset_id

            GROUP BY d.id, d.display_name, d.physical_name

            HAVING avg_score IS NOT NULL

            ORDER BY avg_score ASC

            LIMIT 20

        """))

        for row in result:

            scores.append({

                "dataset_id": row[0], "dataset_name": row[1],

                "avg_quality_score": float(row[2] or 0), "run_count": row[3] or 0

            })

    except Exception as e:

        logger.warning(f"Could not read quality scores: {e}")

    return scores
 
 
def _get_failing_dq_rules(db: Session) -> List[Dict]:

    """Get DQ rules that are failing frequently."""

    rules = []

    try:

        result = db.execute(text("""

            SELECT

                dr.id, dr.name, dr.rule_type, dr.severity,

                COUNT(drq.id) as failure_count,

                MAX(drq.executed_at) as last_failure

            FROM dq_rules dr

            JOIN dq_rule_results drq ON dr.id = drq.rule_id

            WHERE drq.status = 'failed'

            GROUP BY dr.id, dr.name, dr.rule_type, dr.severity

            HAVING COUNT(drq.id) >= 2

            ORDER BY failure_count DESC

            LIMIT 20

        """))

        for row in result:

            rules.append({

                "rule_id": row[0], "rule_name": row[1], "rule_type": row[2],

                "severity": row[3], "failure_count": row[4] or 0

            })

    except Exception as e:

        logger.warning(f"Could not read failing rules: {e}")

    return rules
 
 
def _get_column_profiling_issues(db: Session) -> List[Dict]:

    """Get column profiling issues (high null rates, low health scores)."""

    issues = []

    try:

        result = db.execute(text("""

            SELECT

                cp.column_name, cp.data_type,

                cp.null_count, cp.null_percentage,

                cp.distinct_count, cp.health_score,

                COALESCE(d.display_name, d.physical_name, 'Unknown') as dataset_name,

                cp.profiling_run_id

            FROM column_profiles cp

            JOIN profiling_runs pr ON cp.profiling_run_id = pr.id

            JOIN (

                SELECT dataset_id, MAX(id) as max_run_id

                FROM profiling_runs

                GROUP BY dataset_id

            ) latest ON pr.dataset_id = latest.dataset_id AND pr.id = latest.max_run_id

            LEFT JOIN datasets d ON pr.dataset_id = d.id

            WHERE cp.null_percentage > 15 OR cp.health_score < 60

            ORDER BY cp.null_percentage DESC

            LIMIT 50

        """))

        for row in result:

            issues.append({

                "column_name": row[0], "data_type": row[1],

                "null_count": row[2] or 0, "null_percentage": float(row[3] or 0),

                "health_score": float(row[5] or 0) if row[5] is not None else 0,

                "dataset_name": row[6] or "Unknown"

            })

        logger.info(f"[suggestions] Found {len(issues)} column profiling issues")

    except Exception as e:

        logger.warning(f"Could not read column profiles: {e}")

    return issues
 
 
def _compute_data_hash(db: Session) -> str:

    """

    Compute a stable hash of current data state for cache invalidation.

    Includes counts from all relevant tables so any meaningful change

    triggers a regeneration.

    """

    parts = []

    queries = [

        "SELECT COUNT(*) FROM temporal_checks WHERE status IN ('open', 'investigating')",

        "SELECT COUNT(*) FROM quality_checks WHERE status IN ('open', 'investigating', 'active', 'failed')",

        "SELECT COUNT(*) FROM column_profiles WHERE null_percentage > 15",

        "SELECT COUNT(*) FROM column_profiles WHERE health_score < 60",

        "SELECT COALESCE(SUM(violation_count), 0) FROM temporal_checks WHERE status IN ('open', 'investigating')",

    ]

    for q in queries:

        try:

            val = db.execute(text(q)).scalar() or 0

            parts.append(str(val))

        except Exception:

            parts.append("0")

    return hashlib.md5(":".join(parts).encode()).hexdigest()
 
 
# ─── Clustering ───────────────────────────────────────────────────────────────
 
def _cluster_anomalies(anomalies: List[Dict]) -> List[Dict]:

    """Group similar anomalies together by check_type."""

    clusters: Dict[str, Dict] = defaultdict(lambda: {

        "count": 0, "critical": 0, "high": 0, "medium": 0,

        "datasets": set(), "columns": set(), "violations": 0,

        "examples": []

    })
 
    for a in anomalies:

        check_type = (a.get("check_type") or "UNKNOWN").upper()

        severity   = (a.get("severity")   or "medium").upper()

        key        = check_type
 
        clusters[key]["count"]      += 1

        clusters[key]["violations"] += a.get("violation_count", 0)
 
        if severity == "CRITICAL":

            clusters[key]["critical"] += 1

        elif severity == "HIGH":

            clusters[key]["high"] += 1

        else:

            clusters[key]["medium"] += 1
 
        ds = a.get("dataset_name")

        col = a.get("column_name")

        if ds:

            clusters[key]["datasets"].add(ds)

        if col and col != "unknown":

            clusters[key]["columns"].add(col)
 
        if len(clusters[key]["examples"]) < 2 and a.get("description"):

            clusters[key]["examples"].append(a["description"])
 
    result = []

    for check_type, cl in clusters.items():

        label = _CHECK_LABELS.get(check_type, check_type.replace("_", " ").title())

        score = (cl["critical"] * 4 + cl["high"] * 2 + cl["medium"] * 1 + cl["count"] * 0.5)

        result.append({

            "check_type": check_type,

            "label": label,

            "count": cl["count"],

            "critical": cl["critical"],

            "high": cl["high"],

            "violations": cl["violations"],

            "dataset_count": len(cl["datasets"]),

            "datasets": list(cl["datasets"])[:5],

            "columns": list(cl["columns"])[:5],

            "examples": cl["examples"],

            "score": score

        })
 
    result.sort(key=lambda x: (-x["critical"], -x["score"]))

    return result[:8]
 
 
# ─── Prompt Builder ───────────────────────────────────────────────────────────
 
def _build_prompt_from_data(

    anomalies: List[Dict],

    quality_scores: List[Dict],

    failing_rules: List[Dict],

    profiling_issues: List[Dict],

    clusters: List[Dict]

) -> str:

    """Build a comprehensive prompt using actual data from all sources."""

    parts: List[str] = []
 
    if anomalies:

        parts.append(f"\n## CURRENT ANOMALIES ({len(anomalies)} total)\n")

        for a in anomalies[:15]:

            parts.append(

                f"- {a.get('check_type', 'Unknown')} | Severity: {a.get('severity', 'medium')} | "

                f"Dataset: {a.get('dataset_name', 'Unknown')} | Column: {a.get('column_name', 'N/A')} | "

                f"Violations: {a.get('violation_count', 0)}"

            )
 
    if clusters:

        parts.append(f"\n## DETECTED PATTERNS ({len(clusters)} clusters)\n")

        for i, c in enumerate(clusters[:5]):

            parts.append(

                f"{i+1}. **{c['label']}** - {c['count']} occurrences, {c['critical']} critical, {c['high']} high\n"

                f"   Affected datasets: {', '.join(c['datasets'][:3])}\n"

                f"   Affected columns: {', '.join(c['columns'][:3])}\n"

                f"   Total violations: {c['violations']:,} rows"

            )
 
    if quality_scores:

        low_quality = [q for q in quality_scores if q.get('avg_quality_score', 100) < 70]

        if low_quality:

            parts.append(f"\n## LOW QUALITY DATASETS ({len(low_quality)} datasets with score <70%)\n")

            for q in low_quality[:5]:

                parts.append(

                    f"- {q['dataset_name']}: {q['avg_quality_score']:.1f}% average quality score "

                    f"(based on {q.get('run_count', 0)} runs)"

                )
 
    if failing_rules:

        parts.append(f"\n## FREQUENTLY FAILING DQ RULES ({len(failing_rules)} rules)\n")

        for r in failing_rules[:5]:

            parts.append(

                f"- Rule '{r['rule_name']}' failed {r['failure_count']} times | "

                f"Type: {r['rule_type']} | Severity: {r['severity']}"

            )
 
    if profiling_issues:

        parts.append(f"\n## COLUMN QUALITY ISSUES ({len(profiling_issues)} columns)\n")

        for p in profiling_issues[:8]:

            parts.append(

                f"- Column '{p['column_name']}' in dataset '{p['dataset_name']}': "

                f"{p['null_percentage']:.1f}% null values, health score: {p['health_score']:.0f}%"

            )
 
    # Count distinct issue categories so the LLM knows how many policies to generate

    n_clusters     = len(clusters)

    n_low_quality  = len([q for q in quality_scores if q.get('avg_quality_score', 100) < 70])

    n_failing      = len(failing_rules)

    n_null_issues  = len([p for p in profiling_issues if p.get('null_percentage', 0) > 20])

    issue_category_count = sum([

        1 if n_clusters > 0 else 0,

        1 if n_low_quality > 0 else 0,

        1 if n_failing > 0 else 0,

        1 if n_null_issues > 0 else 0,

    ])

    target_policies = max(3, min(5, issue_category_count + 1))
 
    parts.append(f"""

## INSTRUCTIONS

Based on the data quality issues detected above, generate exactly **{target_policies} LONG-TERM GOVERNANCE POLICIES** that would prevent these issues from recurring.
 
Each policy must address a DISTINCT issue type — do not repeat the same theme.

For each policy provide:

- **name**: A concise, professional policy name (4-8 words)

- **description**: What this policy enforces and how it prevents recurrence

- **reason**: Why this policy is needed (reference specific data from above)

- **policy_type**: One of: Privacy, Retention, Quality, Compliance, Governance

- **priority**: High, Medium, or Low

- **enforcement_strategy**: How to enforce this (pipeline gate, DQ rule, monitoring, etc.)
 
Respond with ONLY a valid JSON array — no markdown fences, no preamble, no trailing text.

Example format:

[

  {{

    "name": "Null Value Prevention Policy",

    "description": "Enforce NOT NULL constraints on critical business columns",

    "reason": "45% null rate detected in customer_id column across 3 datasets",

    "policy_type": "Quality",

    "priority": "High",

    "enforcement_strategy": "Pipeline validation before ingestion"

  }}

]
 
Generate exactly {target_policies} policies now:

""")
 
    return "\n".join(parts)
 
 
# ─── Suggestion Builder ───────────────────────────────────────────────────────
 
def _build_suggestion_from_policy(

    policy: Dict,

    index: int,

    anomalies: List[Dict],

    profiling_issues: List[Dict],

    clusters: List[Dict],

) -> Dict:

    """Convert a single LLM-generated policy dict into the full suggestion shape."""

    pattern_type = policy.get("policy_type", "Quality")

    priority     = policy.get("priority", "Medium")
 
    # Gather affected datasets from anomalies

    affected_datasets = list({

        a.get("dataset_name") for a in anomalies if a.get("dataset_name")

    })[:5]

    affected_dataset_count = len(affected_datasets)
 
    # Gather affected columns from profiling issues

    affected_columns = [

        p.get("column_name") for p in profiling_issues[:3] if p.get("column_name")

    ]
 
    # Best-matching cluster for metadata

    matching_cluster = clusters[index % len(clusters)] if clusters else {}
 
    total_violations = sum(a.get("violation_count", 0) for a in anomalies)

    pattern_score    = len(anomalies) * 10 + len(profiling_issues) + index
 
    return {

        "id":                     f"ai_sug_{uuid.uuid4().hex[:8]}",

        "name":                   policy.get("name", f"Data {pattern_type} Policy"),

        "description":            policy.get("description", "Enforce data quality standards"),

        "policy_type":            pattern_type,

        "priority":               priority,

        "reason":                 policy.get("reason", "Detected from data quality patterns"),

        "triggered_by":           f"{len(anomalies)} anomalies, {len(profiling_issues)} column issues",

        "pattern_summary":        (

            matching_cluster.get("label", pattern_type) + " pattern detected"

            if matching_cluster else f"{pattern_type} issues detected"

        ),

        "affected_datasets_count": affected_dataset_count,

        "affected_datasets":       affected_datasets,

        "affected_columns":        affected_columns,

        "trend":                   "active",

        "policy_scope":            "global" if len(anomalies) > 10 else "dataset-level",

        "recommended_action":      policy.get("enforcement_strategy", "create_dq_rule"),

        "enforcement_strategy":    policy.get("enforcement_strategy", "Automated data quality rule"),

        "occurrence_count":        len(anomalies),

        "total_violations":        total_violations,

        "pattern_score":           pattern_score,

        "pattern_key":             f"ai_{pattern_type.lower()}_{index}",

    }
 
 
def _generate_suggestions_from_data(db: Session) -> List[Dict]:

    """Generate policy suggestions from actual database data."""
 
    anomalies        = _get_anomalies_from_db(db)

    quality_scores   = _get_dataset_quality_scores(db)

    failing_rules    = _get_failing_dq_rules(db)

    profiling_issues = _get_column_profiling_issues(db)
 
    if not anomalies and not quality_scores and not failing_rules and not profiling_issues:

        logger.warning("[suggestions] No data found in any source — returning empty list")

        return []
 
    clusters = _cluster_anomalies(anomalies)
 
    logger.info(

        f"[suggestions] Building prompt — "

        f"anomalies={len(anomalies)}, quality_scores={len(quality_scores)}, "

        f"failing_rules={len(failing_rules)}, profiling_issues={len(profiling_issues)}, "

        f"clusters={len(clusters)}"

    )
 
    prompt       = _build_prompt_from_data(anomalies, quality_scores, failing_rules, profiling_issues, clusters)

    llm_response = _llm(prompt, max_tokens=2000)
 
    if not llm_response:

        logger.warning("[suggestions] LLM unavailable — using fallback generator")

        return _generate_fallback_suggestions(clusters, profiling_issues, failing_rules)
 
    # ── Parse LLM JSON ────────────────────────────────────────────────────────

    cleaned = llm_response.strip()

    # Strip any markdown fences the model may have added despite instructions

    for fence in ("```json", "```"):

        if cleaned.startswith(fence):

            cleaned = cleaned[len(fence):]

    if cleaned.endswith("```"):

        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()
 
    # Find the outermost JSON array in case the model prepended text

    array_start = cleaned.find("[")

    array_end   = cleaned.rfind("]")

    if array_start != -1 and array_end != -1 and array_end > array_start:

        cleaned = cleaned[array_start: array_end + 1]
 
    try:

        policies = json.loads(cleaned)

        if not isinstance(policies, list):

            raise ValueError("LLM did not return a JSON array")
 
        suggestions = []

        for i, policy in enumerate(policies):

            if not isinstance(policy, dict):

                continue

            suggestion = _build_suggestion_from_policy(

                policy, i, anomalies, profiling_issues, clusters

            )

            suggestions.append(suggestion)
 
        logger.info(f"[suggestions] LLM generated {len(suggestions)} suggestions")
 
        # If LLM returned fewer than expected, pad with fallback suggestions

        if len(suggestions) < 2:

            logger.warning("[suggestions] LLM returned too few — supplementing with fallback")

            fallback = _generate_fallback_suggestions(clusters, profiling_issues, failing_rules)

            # Avoid exact name duplicates

            existing_names = {s["name"] for s in suggestions}

            for fb in fallback:

                if fb["name"] not in existing_names:

                    suggestions.append(fb)

                    existing_names.add(fb["name"])
 
        return suggestions
 
    except (json.JSONDecodeError, ValueError) as e:

        logger.error(f"[suggestions] Failed to parse LLM JSON: {e} — raw: {cleaned[:300]}")

        return _generate_fallback_suggestions(clusters, profiling_issues, failing_rules)
 
 
def _generate_fallback_suggestions(

    clusters: List[Dict],

    profiling_issues: List[Dict],

    failing_rules: List[Dict],

) -> List[Dict]:

    """Generate suggestions from clusters without LLM (fallback)."""

    suggestions: List[Dict] = []

    seen_names: Set[str] = set()
 
    # One suggestion per cluster

    for i, cluster in enumerate(clusters[:5]):

        priority = "High" if cluster["critical"] > 0 else "Medium" if cluster["high"] > 0 else "Low"

        name = f"{cluster['label']} Prevention Policy"

        if name in seen_names:

            name = f"{cluster['label']} Governance Policy {i+1}"

        seen_names.add(name)
 
        suggestions.append({

            "id":                     f"cluster_sug_{uuid.uuid4().hex[:8]}",

            "name":                   name,

            "description":            f"Prevent {cluster['label'].lower()} from recurring across all datasets",

            "policy_type":            _CHECK_TO_POLICY_TYPE.get(cluster['check_type'], "Quality"),

            "priority":               priority,

            "reason":                 (

                f"{cluster['count']} occurrences of {cluster['label'].lower()} detected, "

                f"including {cluster['critical']} critical issues"

            ),

            "triggered_by":           f"{cluster['count']} anomalies across {cluster['dataset_count']} dataset(s)",

            "pattern_summary":        f"{cluster['label']} affecting {len(cluster.get('columns', []))} columns",

            "affected_datasets_count": cluster['dataset_count'],

            "affected_datasets":      cluster.get('datasets', [])[:3],

            "affected_columns":       cluster.get('columns', [])[:3],

            "trend":                  "active",

            "policy_scope":           "global" if cluster['dataset_count'] > 2 else "dataset-level",

            "recommended_action":     _CHECK_TO_RECOMMENDED_ACTION.get(

                                          cluster['check_type'], "create_dq_rule:data quality check"

                                      ),

            "enforcement_strategy":   "Automated data quality rule on ingestion",

            "occurrence_count":       cluster['count'],

            "total_violations":       cluster['violations'],

            "pattern_score":          cluster['score'],

            "pattern_key":            f"cluster_{cluster['check_type']}",

        })
 
    # Add a completeness policy from profiling issues if we have data for it

    null_issues = [p for p in profiling_issues if p.get('null_percentage', 0) > 20]

    if null_issues and len(suggestions) < 5:

        name = "Data Completeness Enforcement Policy"

        if name not in seen_names:

            seen_names.add(name)

            suggestions.append({

                "id":                     f"profile_sug_{uuid.uuid4().hex[:8]}",

                "name":                   name,

                "description":            "Ensure critical columns meet minimum completeness thresholds before data is accepted into the platform",

                "policy_type":            "Quality",

                "priority":               "High",

                "reason":                 f"{len(null_issues)} columns have >20% null values across datasets",

                "triggered_by":           f"{len(null_issues)} columns with high null rates",

                "pattern_summary":        "High null rates detected across multiple datasets",

                "affected_datasets_count": len({p.get('dataset_name') for p in null_issues}),

                "affected_datasets":      list({p.get('dataset_name') for p in null_issues if p.get('dataset_name')})[:3],

                "affected_columns":       [p.get('column_name') for p in null_issues[:3] if p.get('column_name')],

                "trend":                  "active",

                "policy_scope":           "global",

                "recommended_action":     "create_dq_rule:null rate threshold rule",

                "enforcement_strategy":   "Completeness validation gate on data ingestion",

                "occurrence_count":       len(null_issues),

                "total_violations":       sum(p.get('null_count', 0) for p in null_issues[:5]),

                "pattern_score":          len(null_issues) * 10,

                "pattern_key":            "high_null_rate_profile",

            })
 
    # Add a rule-failure policy if we have failing rules

    if failing_rules and len(suggestions) < 5:

        name = "Persistent Rule Failure Remediation Policy"

        if name not in seen_names:

            seen_names.add(name)

            top_rule = failing_rules[0]

            suggestions.append({

                "id":                     f"rule_sug_{uuid.uuid4().hex[:8]}",

                "name":                   name,

                "description":            "Mandate investigation and remediation of DQ rules that fail repeatedly within a rolling window",

                "policy_type":            "Governance",

                "priority":               "Medium",

                "reason":                 (

                    f"{len(failing_rules)} DQ rules are failing repeatedly — "

                    f"e.g. '{top_rule['rule_name']}' failed {top_rule['failure_count']} times"

                ),

                "triggered_by":           f"{len(failing_rules)} frequently-failing DQ rules",

                "pattern_summary":        "Recurring DQ rule failures indicate systemic data issues",

                "affected_datasets_count": 0,

                "affected_datasets":      [],

                "affected_columns":       [],

                "trend":                  "recurring",

                "policy_scope":           "global",

                "recommended_action":     "apply_validation:rule failure escalation workflow",

                "enforcement_strategy":   "Automated escalation after 3 consecutive failures",

                "occurrence_count":       len(failing_rules),

                "total_violations":       sum(r.get('failure_count', 0) for r in failing_rules),

                "pattern_score":          sum(r.get('failure_count', 0) for r in failing_rules) * 5,

                "pattern_key":            "failing_dq_rules",

            })
 
    return suggestions
 
 
# ─── API Endpoints ────────────────────────────────────────────────────────────
 
@policy_suggestions_router.get("/governance/policy-suggestions")

def get_policy_suggestions(db: Session = Depends(get_db)):

    """

    Return AI-generated governance policy suggestions.
 
    Cache strategy:

    - Suggestions are cached for _CACHE_TTL_SECONDS.

    - Cache is also invalidated when the underlying data hash changes.

    - Dismissed/adopted suggestions are filtered out from the response

      but the underlying cache is preserved so a reset-dismissed call

      can restore them without a full regeneration.

    """

    global _suggestions_cache, _cache_expires_at, _cache_data_hash
 
    now          = datetime.now(timezone.utc)

    current_hash = _compute_data_hash(db)
 
    # Invalidate cache only when data has actually changed

    data_changed = (current_hash != _cache_data_hash)

    cache_expired = (_cache_expires_at is None or now >= _cache_expires_at)
 
    if data_changed:

        logger.info(f"[suggestions] Data hash changed ({_cache_data_hash} → {current_hash}) — regenerating")

        _suggestions_cache = []

        _cache_expires_at  = None

        _cache_data_hash   = current_hash
 
    if _suggestions_cache and not cache_expired:

        visible = [

            s for s in _suggestions_cache

            if s["id"] not in _dismissed and s["id"] not in _adopted

        ]

        logger.info(f"[suggestions] Cache HIT — {len(_suggestions_cache)} cached, {len(visible)} visible")

        return visible
 
    # Generate fresh suggestions

    logger.info("[suggestions] Cache MISS — generating from database")

    try:

        suggestions = _generate_suggestions_from_data(db)

    except Exception as e:

        logger.error(f"[suggestions] Generation failed: {e}", exc_info=True)

        suggestions = []
 
    _suggestions_cache = suggestions

    _cache_data_hash   = current_hash

    _cache_expires_at  = now + timedelta(seconds=_CACHE_TTL_SECONDS)
 
    visible = [

        s for s in suggestions

        if s["id"] not in _dismissed and s["id"] not in _adopted

    ]

    logger.info(f"[suggestions] Returning {len(visible)} of {len(suggestions)} suggestions")

    return visible
 
 
@policy_suggestions_router.post("/governance/policy-suggestions/{suggestion_id}/dismiss")

def dismiss_suggestion(suggestion_id: str):

    """Mark a suggestion as dismissed (hidden from the feed)."""

    _dismissed.add(suggestion_id)

    return {"status": "dismissed", "id": suggestion_id}
 
 
@policy_suggestions_router.post("/governance/policy-suggestions/{suggestion_id}/adopt")

def adopt_suggestion(suggestion_id: str, request: Request, db: Session = Depends(get_db)):

    """Adopt a suggestion and create an audit log entry."""

    _adopted.add(suggestion_id)

    cached = next((s for s in _suggestions_cache if s["id"] == suggestion_id), None)
 
    try:

        from app.routers.governance_routes import _audit, _usr, _ip

        _audit(

            db, "Policy Suggestion Adopted", "Suggestion",

            cached.get("name", "Unknown") if cached else "Unknown",

            f"Adopted AI-suggested policy: {cached.get('name', '') if cached else ''}",

            "info", _usr(request), _ip(request)

        )

    except Exception:

        pass
 
    return {

        "status": "adopted",

        "id": suggestion_id,

        "recommended_action": cached.get("recommended_action") if cached else None,

    }
 
 
@policy_suggestions_router.delete("/governance/policy-suggestions/{suggestion_id}/adopt")

def unadopt_suggestion(suggestion_id: str):

    """Remove adoption status."""

    _adopted.discard(suggestion_id)

    return {"status": "unadopted", "id": suggestion_id}
 
 
@policy_suggestions_router.delete("/governance/policy-suggestions/dismissed")

def reset_dismissed():

    """Reset all dismissed suggestions (restores them to the feed)."""

    global _dismissed

    _dismissed = set()

    return {"status": "reset"}
 
 
@policy_suggestions_router.post("/governance/policy-suggestions/refresh")

def force_refresh():

    """Force a full cache refresh on next GET (does NOT clear dismiss/adopt state)."""

    global _suggestions_cache, _cache_expires_at, _cache_data_hash

    _suggestions_cache = []

    _cache_expires_at  = None

    _cache_data_hash   = None

    logger.info("[suggestions] Cache manually cleared — will regenerate on next GET")

    return {"status": "cache_cleared"}
 
 
@policy_suggestions_router.get("/governance/policy-suggestions/debug")

def debug_suggestions(db: Session = Depends(get_db)):

    """Debug endpoint — shows what data is available for suggestion generation."""

    anomalies        = _get_anomalies_from_db(db)

    quality_scores   = _get_dataset_quality_scores(db)

    failing_rules    = _get_failing_dq_rules(db)

    profiling_issues = _get_column_profiling_issues(db)

    clusters         = _cluster_anomalies(anomalies)

    current_hash     = _compute_data_hash(db)
 
    return {

        "data_hash":               current_hash,

        "cached_hash":             _cache_data_hash,

        "cache_expires_at":        _cache_expires_at.isoformat() if _cache_expires_at else None,

        "cached_suggestion_count": len(_suggestions_cache),

        "dismissed_count":         len(_dismissed),

        "adopted_count":           len(_adopted),

        "anomalies_count":         len(anomalies),

        "quality_scores_count":    len(quality_scores),

        "failing_rules_count":     len(failing_rules),

        "profiling_issues_count":  len(profiling_issues),

        "clusters_count":          len(clusters),

        "low_quality_datasets": [

            q for q in quality_scores if q.get('avg_quality_score', 100) < 70

        ][:5],

        "top_anomalies": [

            {"type": a.get("check_type"), "severity": a.get("severity"), "dataset": a.get("dataset_name")}

            for a in anomalies[:5]

        ],

        "clusters": [

            {"type": c["check_type"], "count": c["count"], "critical": c["critical"], "score": c["score"]}

            for c in clusters[:8]

        ],

    }