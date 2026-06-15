"""
app/routers/llm_metrics_router.py

Serves /api/llm-metrics — aggregated LLM observability data sourced from
the local `llm_calls` table (written by app.services.llm_tracker on every
LLM call across the platform).

This endpoint is the single source of truth for the standalone Health
Observatory dashboard's "LLM Performance" tab.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, median

from fastapi import APIRouter
from sqlalchemy import text

from app.database import engine

router = APIRouter(prefix="/api", tags=["llm-metrics"])

# Per-million-token pricing — Azure AI Foundry Llama 3.3 70B (approx, USD)
# Adjust if your contracted rate differs.
_PRICE_PER_M_INPUT  = 0.71
_PRICE_PER_M_OUTPUT = 0.71

FEATURES = ["profiling", "dq_rules", "anomaly", "kg", "agent", "policy"]
FEATURE_LABELS = {
    "profiling": "Profiling AI",
    "dq_rules":  "DQ Rule Generation",
    "anomaly":   "Anomaly Root Cause",
    "kg":        "Knowledge Graph",
    "agent":     "AI Chat Agent",
    "policy":    "Policy Suggestions",
}


def _rows(sql: str, params: dict | None = None) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]


def _table_exists(conn, name: str) -> bool:
    r = conn.execute(text(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
    ), {"n": name}).fetchone()
    return r is not None


@router.get("/llm-metrics")
def get_llm_metrics():
    with engine.connect() as conn:
        if not _table_exists(conn, "llm_calls"):
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "available": False,
                "message": "llm_calls table not found — restart backend to trigger bootstrap",
            }

    now    = datetime.now(timezone.utc)
    cutoff_24h = (now - timedelta(hours=24)).isoformat()
    cutoff_7d  = (now - timedelta(days=7)).isoformat()
    cutoff_30d = (now - timedelta(days=30)).isoformat()

    # ── Overall summary (30d) ─────────────────────────────────────────────────
    all_rows = _rows(
        "SELECT * FROM llm_calls WHERE timestamp >= :cutoff ORDER BY timestamp DESC",
        {"cutoff": cutoff_30d},
    )

    total_calls   = len(all_rows)
    success_rows  = [r for r in all_rows if r["success"]]
    error_rows    = [r for r in all_rows if not r["success"]]
    success_count = len(success_rows)
    error_count   = len(error_rows)

    latencies = [r["latency_ms"] for r in all_rows if r["latency_ms"] is not None]
    avg_latency = round(mean(latencies), 1) if latencies else None
    p95_latency = None
    if latencies:
        sorted_lat = sorted(latencies)
        idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95_latency = round(sorted_lat[idx], 1)

    total_prompt_tokens     = sum(r["prompt_tokens"]     or 0 for r in all_rows)
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in all_rows)
    total_tokens            = total_prompt_tokens + total_completion_tokens

    estimated_cost = round(
        (total_prompt_tokens / 1_000_000) * _PRICE_PER_M_INPUT +
        (total_completion_tokens / 1_000_000) * _PRICE_PER_M_OUTPUT,
        4,
    )

    success_rate = round(success_count / total_calls * 100, 2) if total_calls else None
    error_rate   = round(error_count   / total_calls * 100, 2) if total_calls else None

    output_lengths = [r["output_length"] for r in success_rows if r["output_length"]]
    avg_output_len = round(mean(output_lengths)) if output_lengths else None

    # ── Calls per day (last 7d) ───────────────────────────────────────────────
    rows_7d = [r for r in all_rows if r["timestamp"] >= cutoff_7d]
    calls_per_day = round(len(rows_7d) / 7, 2)

    # ── Hourly time series (last 24h) ─────────────────────────────────────────
    rows_24h = [r for r in all_rows if r["timestamp"] >= cutoff_24h]
    hourly_buckets: dict[str, dict] = {}
    for i in range(24):
        bucket_time = (now - timedelta(hours=23 - i)).strftime("%Y-%m-%dT%H:00")
        hourly_buckets[bucket_time] = {
            "hour": bucket_time, "calls": 0, "errors": 0,
            "tokens": 0, "latencies": [],
        }
    for r in rows_24h:
        try:
            ts = r["timestamp"][:13] + ":00"  # truncate to hour
            if ts in hourly_buckets:
                hourly_buckets[ts]["calls"] += 1
                if not r["success"]:
                    hourly_buckets[ts]["errors"] += 1
                hourly_buckets[ts]["tokens"] += (r["total_tokens"] or 0)
                if r["latency_ms"] is not None:
                    hourly_buckets[ts]["latencies"].append(r["latency_ms"])
        except Exception:
            continue

    timeseries = []
    for bucket_time, b in hourly_buckets.items():
        timeseries.append({
            "hour":       bucket_time,
            "calls":      b["calls"],
            "errors":     b["errors"],
            "tokens":     b["tokens"],
            "avg_latency_ms": round(mean(b["latencies"]), 1) if b["latencies"] else None,
        })

    # ── Per-feature breakdown ─────────────────────────────────────────────────
    feature_breakdown = []
    for feat in FEATURES:
        feat_rows = [r for r in all_rows if r["feature"] == feat]
        if not feat_rows:
            feature_breakdown.append({
                "feature": feat, "label": FEATURE_LABELS.get(feat, feat),
                "calls": 0, "errors": 0, "error_rate": None,
                "avg_latency_ms": None, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
                "avg_output_length": None,
            })
            continue

        feat_success = [r for r in feat_rows if r["success"]]
        feat_errors  = [r for r in feat_rows if not r["success"]]
        feat_lat     = [r["latency_ms"] for r in feat_rows if r["latency_ms"] is not None]
        feat_out_len = [r["output_length"] for r in feat_success if r["output_length"]]

        feature_breakdown.append({
            "feature": feat, "label": FEATURE_LABELS.get(feat, feat),
            "calls": len(feat_rows), "errors": len(feat_errors),
            "error_rate": round(len(feat_errors) / len(feat_rows) * 100, 2),
            "avg_latency_ms": round(mean(feat_lat), 1) if feat_lat else None,
            "prompt_tokens":     sum(r["prompt_tokens"]     or 0 for r in feat_rows),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in feat_rows),
            "total_tokens":      sum(r["total_tokens"]      or 0 for r in feat_rows),
            "avg_output_length": round(mean(feat_out_len)) if feat_out_len else None,
        })

    # ── Error type breakdown ──────────────────────────────────────────────────
    error_types: dict[str, int] = {}
    for r in error_rows:
        et = r["error_type"] or "Unknown"
        error_types[et] = error_types.get(et, 0) + 1

    # ── Recent activity feed (last 25 calls) ──────────────────────────────────
    recent = []
    for r in all_rows[:25]:
        recent.append({
            "timestamp":   r["timestamp"],
            "feature":     r["feature"],
            "label":       FEATURE_LABELS.get(r["feature"], r["feature"]),
            "model":       r["model"],
            "latency_ms":  r["latency_ms"],
            "total_tokens": r["total_tokens"],
            "success":     bool(r["success"]),
            "error_type":  r["error_type"],
        })

    return {
        "generated_at": now.isoformat(),
        "available": True,
        "window": "30d",
        "summary": {
            "total_calls": total_calls,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate_pct": success_rate,
            "error_rate_pct": error_rate,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "calls_per_day_7d": calls_per_day,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "avg_output_length_chars": avg_output_len,
        },
        "timeseries_24h": timeseries,
        "feature_breakdown": feature_breakdown,
        "error_types": error_types,
        "recent_activity": recent,
        "pricing_note": (
            f"Cost estimated at ${_PRICE_PER_M_INPUT}/1M input tokens, "
            f"${_PRICE_PER_M_OUTPUT}/1M output tokens (Azure AI Foundry Llama 3.3 70B approx rate)"
        ),
    }
