"""
azure_metrics_collector.py
──────────────────────────
Tracks REAL token usage, latency, and cost from every LLM call made
through your Azure AI Foundry endpoint.

HOW IT WORKS:
  Every time your app calls the LLM (chat.completions.create), the
  OpenAI SDK returns a response.usage object with:
    - prompt_tokens
    - completion_tokens
    - total_tokens

  This module intercepts those values via a patched LLM client wrapper
  and writes them to a local DB table (llm_usage_log) so the health
  dashboard can show live, accumulating stats.

WHAT YOU NEED:
  The same env vars your app already uses — nothing new:
    AZURE_OPENAI_ENDPOINT  (e.g. https://dilip-mm4oi19h-eastus2.services.ai.azure.com)
    AZURE_OPENAI_API_KEY   (your Foundry API key)
    AZURE_OPENAI_MODEL     (e.g. Llama-3.3-70B-Instruct — optional, auto-detected)

NO AZURE MONITOR, NO APP REGISTRATION, NO SERVICE PRINCIPAL NEEDED.
"""

import os
import sqlite3
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("ai_dqm.azure_metrics")

DB_PATH    = os.getenv("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")
ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL", "")

# In-process cache — busted on every new LLM call
_cache: Optional[dict] = None
_cache_ts: Optional[datetime] = None
_CACHE_TTL = 30   # seconds


# ═══════════════════════════════════════════════════════════════════════════
# DB BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════


# ── Published Azure AI Foundry pricing for Llama-3.3-70B-Instruct ────────────
# Source: Azure AI Foundry marketplace, verified June 2026
# $0.71 per 1M input tokens, $0.71 per 1M output tokens
# Override via env vars if you switch to a different model
_PRICE_INPUT_PER_M  = float(os.getenv("LLM_PRICE_INPUT_PER_M",  "0.71"))
_PRICE_OUTPUT_PER_M = float(os.getenv("LLM_PRICE_OUTPUT_PER_M", "0.71"))


def _calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Exact cost using published Azure AI Foundry pricing.
    Formula: (prompt_tokens / 1_000_000 * input_rate)
           + (completion_tokens / 1_000_000 * output_rate)
    Returns USD rounded to 8 decimal places.
    """
    return round(
        (prompt_tokens    / 1_000_000) * _PRICE_INPUT_PER_M +
        (completion_tokens / 1_000_000) * _PRICE_OUTPUT_PER_M,
        8,
    )


def ensure_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at         TEXT NOT NULL,
            model             TEXT,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens      INTEGER NOT NULL DEFAULT 0,
            cost_usd          REAL NOT NULL DEFAULT 0,
            latency_ms        REAL,
            status            TEXT DEFAULT 'success',
            error_message     TEXT,
            caller            TEXT
        )
    """)
    # Migrate existing tables that don't have cost_usd yet
    cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_usage_log)").fetchall()}
    if "cost_usd" not in cols:
        conn.execute("ALTER TABLE llm_usage_log ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0")
        # Back-fill cost for existing rows using the pricing formula
        conn.execute(f"""
            UPDATE llm_usage_log
            SET cost_usd = ROUND(
                (prompt_tokens    / 1000000.0) * {_PRICE_INPUT_PER_M} +
                (completion_tokens / 1000000.0) * {_PRICE_OUTPUT_PER_M},
                8
            )
            WHERE cost_usd = 0 AND (prompt_tokens > 0 OR completion_tokens > 0)
        """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# RECORD A SINGLE LLM CALL
# ═══════════════════════════════════════════════════════════════════════════

def record_call(
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    model: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    caller: Optional[str] = None,
) -> None:
    """
    Persists one LLM call to llm_usage_log including exact cost,
    then busts the dashboard cache so the new call appears immediately.
    """
    global _cache, _cache_ts
    try:
        ensure_table()
        cost_usd = _calc_cost(prompt_tokens, completion_tokens)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO llm_usage_log
              (called_at, model, prompt_tokens, completion_tokens, total_tokens,
               cost_usd, latency_ms, status, error_message, caller)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            model or MODEL_NAME or "unknown",
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            cost_usd,
            round(latency_ms, 2),
            status,
            error_message,
            caller,
        ))
        conn.commit()
        conn.close()
        _cache    = None
        _cache_ts = None
        logger.debug(
            f"LLM call recorded: {prompt_tokens}p + {completion_tokens}c = "
            f"{prompt_tokens + completion_tokens} tokens | "
            f"${cost_usd:.6f} | {latency_ms:.0f}ms"
        )
    except Exception as e:
        logger.error(f"record_call failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PATCHED LLM CLIENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class TrackedOpenAIClient:
    """
    Drop-in replacement for the OpenAI client returned by get_llm_client().
    Every chat.completions.create call is transparently intercepted to
    record token usage and latency. All other behaviour is unchanged.
    """

    def __init__(self, raw_client, caller_label: str = "app"):
        self._client = raw_client
        self.chat    = _ChatProxy(raw_client.chat, caller_label)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, chat, caller):
        self._chat        = chat
        self.completions  = _CompletionsProxy(chat.completions, caller)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _CompletionsProxy:
    def __init__(self, completions, caller):
        self._completions = completions
        self._caller      = caller

    def create(self, **kwargs):
        start     = time.time()
        status    = "success"
        error_msg = None
        response  = None
        is_stream = kwargs.get("stream", False)

        try:
            response = self._completions.create(**kwargs)

            # Streaming responses: wrap the iterator to capture usage from final chunk
            if is_stream:
                return self._wrap_stream(response, start, kwargs)

            return response

        except Exception as e:
            status    = "error"
            error_msg = str(e)[:1000]   # capture full error so we can diagnose
            raise

        finally:
            # Only record here for non-streaming calls.
            # Streaming calls are recorded inside _wrap_stream after iteration completes.
            if not is_stream:
                latency_ms     = (time.time() - start) * 1000
                prompt_tok     = 0
                completion_tok = 0
                model_used     = kwargs.get("model", MODEL_NAME or "unknown")

                if response is not None and hasattr(response, "usage") and response.usage:
                    prompt_tok     = getattr(response.usage, "prompt_tokens",     0) or 0
                    completion_tok = getattr(response.usage, "completion_tokens", 0) or 0
                    if hasattr(response, "model") and response.model:
                        model_used = response.model

                record_call(
                    prompt_tokens     = prompt_tok,
                    completion_tokens = completion_tok,
                    latency_ms        = latency_ms,
                    model             = model_used,
                    status            = status,
                    error_message     = error_msg,
                    caller            = self._caller,
                )

    def _wrap_stream(self, stream_iter, start, kwargs):
        """
        Wraps a streaming response, collecting chunks and extracting
        usage from the final chunk (stream_options={"include_usage": True}).
        Records to DB after the stream is fully consumed.
        """
        prompt_tok     = 0
        completion_tok = 0
        model_used     = kwargs.get("model", MODEL_NAME or "unknown")
        status         = "success"
        error_msg      = None

        try:
            for chunk in stream_iter:
                # Azure AI Foundry sends usage in the last chunk when
                # stream_options={"include_usage": True} is set
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tok     = getattr(chunk.usage, "prompt_tokens",     0) or 0
                    completion_tok = getattr(chunk.usage, "completion_tokens", 0) or 0
                if hasattr(chunk, "model") and chunk.model:
                    model_used = chunk.model
                yield chunk
        except Exception as e:
            status    = "error"
            error_msg = str(e)[:1000]
            raise
        finally:
            latency_ms = (time.time() - start) * 1000
            record_call(
                prompt_tokens     = prompt_tok,
                completion_tokens = completion_tok,
                latency_ms        = latency_ms,
                model             = model_used,
                status            = status,
                error_message     = error_msg,
                caller            = self._caller,
            )

    def __getattr__(self, name):
        return getattr(self._completions, name)


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE STATS FOR THE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def fetch_live(force: bool = False) -> Optional[dict]:
    """
    Returns live aggregated stats from llm_usage_log.
    Cache TTL is 30s but is busted immediately on every new LLM call,
    so the dashboard reflects new usage within one refresh cycle.
    """
    global _cache, _cache_ts

    if not force and _cache is not None and _cache_ts is not None:
        age = (datetime.now(timezone.utc) - _cache_ts).total_seconds()
        if age < _CACHE_TTL:
            return _cache

    window_hours = int(os.getenv("AZURE_MONITOR_WINDOW_HOURS", "24"))
    now          = datetime.now(timezone.utc)
    cutoff       = (now - timedelta(hours=window_hours)).isoformat()
    prev_cutoff  = (now - timedelta(hours=window_hours * 2)).isoformat()

    try:
        ensure_table()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # Current window
        cur = conn.execute("""
            SELECT
                COUNT(*)                                            AS total_calls,
                SUM(CASE WHEN status='success' THEN 1 ELSE 0 END)  AS success_calls,
                SUM(CASE WHEN status='error'   THEN 1 ELSE 0 END)  AS error_calls,
                SUM(prompt_tokens)                                  AS prompt_tokens,
                SUM(completion_tokens)                              AS completion_tokens,
                SUM(total_tokens)                                   AS total_tokens,
                SUM(cost_usd)                                       AS total_cost_usd,
                AVG(latency_ms)                                     AS avg_latency_ms,
                MAX(latency_ms)                                     AS max_latency_ms,
                MIN(latency_ms)                                     AS min_latency_ms
            FROM llm_usage_log
            WHERE called_at >= ?
        """, (cutoff,)).fetchone()

        # Previous window for trend
        prev = conn.execute("""
            SELECT
                COUNT(*)          AS total_calls,
                SUM(total_tokens) AS total_tokens,
                SUM(cost_usd)     AS total_cost_usd
            FROM llm_usage_log
            WHERE called_at >= ? AND called_at < ?
        """, (prev_cutoff, cutoff)).fetchone()

        # Per-model breakdown
        models_rows = conn.execute("""
            SELECT
                model,
                COUNT(*)               AS calls,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(total_tokens)      AS total_tokens,
                SUM(cost_usd)          AS total_cost_usd,
                AVG(latency_ms)        AS avg_latency_ms
            FROM llm_usage_log
            WHERE called_at >= ?
            GROUP BY model
            ORDER BY total_tokens DESC
        """, (cutoff,)).fetchall()

        # Last 10 calls with error messages
        recent_rows = conn.execute("""
            SELECT called_at, model, prompt_tokens, completion_tokens,
                   total_tokens, cost_usd, latency_ms, status, error_message
            FROM llm_usage_log
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

        # All-time totals
        alltime = conn.execute("""
            SELECT
                COUNT(*)               AS total_calls,
                SUM(total_tokens)      AS total_tokens,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(cost_usd)          AS total_cost_usd,
                MIN(called_at)         AS first_call,
                MAX(called_at)         AS last_call
            FROM llm_usage_log
        """).fetchone()

        conn.close()

        total_calls   = cur["total_calls"]    or 0
        success_calls = cur["success_calls"]  or 0
        error_calls   = cur["error_calls"]    or 0
        prompt_tokens = cur["prompt_tokens"]  or 0
        comp_tokens   = cur["completion_tokens"] or 0
        total_tokens  = cur["total_tokens"]   or 0
        window_cost   = cur["total_cost_usd"] or 0.0
        avg_lat       = cur["avg_latency_ms"]
        max_lat       = cur["max_latency_ms"]
        min_lat       = cur["min_latency_ms"]

        prev_calls    = prev["total_calls"]      or 0
        prev_tokens   = prev["total_tokens"]     or 0
        prev_cost     = prev["total_cost_usd"]   or 0.0

        # Trends vs previous window
        token_trend = None
        if prev_tokens and prev_tokens > 0:
            token_trend = round(((total_tokens - prev_tokens) / prev_tokens) * 100, 1)

        call_trend = None
        if prev_calls and prev_calls > 0:
            call_trend = round(((total_calls - prev_calls) / prev_calls) * 100, 1)

        cost_trend = None
        if prev_cost and prev_cost > 0:
            cost_trend = round(((window_cost - prev_cost) / prev_cost) * 100, 1)

        # Cost per call average
        avg_cost_per_call = round(window_cost / total_calls, 8) if total_calls > 0 else 0.0

        # Resolve model name
        resolved_model = MODEL_NAME
        if not resolved_model and models_rows:
            first = dict(models_rows[0])
            if first.get("model") and first["model"] != "unknown":
                resolved_model = first["model"]
        if not resolved_model and alltime["last_call"]:
            row = sqlite3.connect(DB_PATH).execute(
                "SELECT model FROM llm_usage_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                resolved_model = row[0]
        resolved_model = resolved_model or "Azure AI Foundry"

        data = {
            "source":             "llm_usage_log",
            "auth_method":        "api_key_intercept",
            "deployment_name":    resolved_model,
            "endpoint":           ENDPOINT or "not set",
            "window_hours":       window_hours,
            "window_start":       cutoff,
            "window_end":         now.isoformat(),
            "fetched_at":         now.isoformat(),

            "total_requests":     total_calls,
            "success_requests":   success_calls,
            "error_count":        error_calls,
            "throttled_count":    None,
            "blocked_count":      None,
            "server_errors":      None,
            "client_errors":      None,

            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  comp_tokens,
            "total_tokens":       total_tokens,

            "avg_latency_ms":     round(avg_lat, 2) if avg_lat is not None else None,
            "max_latency_ms":     round(max_lat, 2) if max_lat is not None else None,
            "min_latency_ms":     round(min_lat, 2) if min_lat is not None else None,

            # Cost — calculated from real token counts using published pricing
            "cost_usd":              round(window_cost, 6),
            "prev_window_cost_usd":  round(prev_cost, 6),
            "cost_trend_pct":        cost_trend,
            "avg_cost_per_call_usd": avg_cost_per_call,
            "price_input_per_m":     _PRICE_INPUT_PER_M,
            "price_output_per_m":    _PRICE_OUTPUT_PER_M,

            "token_trend_pct":    token_trend,
            "call_trend_pct":     call_trend,
            "prev_window_tokens": prev_tokens,
            "prev_window_calls":  prev_calls,

            "alltime_calls":          alltime["total_calls"]      or 0,
            "alltime_tokens":         alltime["total_tokens"]      or 0,
            "alltime_prompt_tokens":  alltime["prompt_tokens"]     or 0,
            "alltime_comp_tokens":    alltime["completion_tokens"] or 0,
            "alltime_cost_usd":       round(alltime["total_cost_usd"] or 0.0, 6),
            "first_call_at":          alltime["first_call"],
            "last_call_at":           alltime["last_call"],

            "models":             [dict(r) for r in models_rows],
            "recent_calls":       [dict(r) for r in recent_rows],
            "actual_cost":        {"amount": round(window_cost, 6), "currency": "USD"},
            "cache_age_s":        0,
        }

        _cache    = data
        _cache_ts = now
        return data

    except Exception as e:
        logger.error(f"fetch_live failed: {e}", exc_info=True)
        return None


def is_configured() -> bool:
    """True if app already has LLM credentials (AZURE_OPENAI_ENDPOINT + KEY)."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
