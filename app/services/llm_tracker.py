"""
app/services/llm_tracker.py

Central LLM call tracker — dual-write to:
  1. SQLite  llm_calls  table  (primary source for /api/llm-metrics dashboard)
  2. Langfuse Cloud            (optional; enables full trace inspection UI)

Usage in any service:
    from app.services.llm_tracker import track_llm_call, track_error
    import time

    _t0 = time.time()
    response = make_llm_call(prompt)
    usage = response_json.get("usage", {})

    track_llm_call(
        feature   = "dq_rules",          # see FEATURES list below
        model     = "Llama-3.3-70B-Instruct",
        prompt_tokens     = usage.get("prompt_tokens"),
        completion_tokens = usage.get("completion_tokens"),
        latency_ms        = (time.time() - _t0) * 1000,
        success           = True,
        input_length      = len(prompt),
        output_length     = len(response) if response else 0,
        prompt            = prompt[:500],   # optional, for Langfuse trace
        response          = response[:500], # optional
    )

Features (use exactly these strings):
    "profiling"   - dataset description + column semantic labels
    "dq_rules"    - AI DQ rule generation
    "anomaly"     - root-cause analysis
    "kg"          - knowledge graph relationship detection
    "agent"       - AI chat agent responses
    "policy"      - governance policy suggestions
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Optional

# ── Langfuse singleton ────────────────────────────────────────────────────────
_lf        = None
_lf_ready  = False
_lf_lock   = threading.Lock()


def _get_langfuse():
    global _lf, _lf_ready
    if _lf_ready:
        return _lf
    with _lf_lock:
        if _lf_ready:
            return _lf
        pk   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk   = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pk and sk:
            try:
                from langfuse import Langfuse
                _lf = Langfuse(public_key=pk, secret_key=sk, host=host)
                print("[llm_tracker] ✓ Langfuse connected → cloud.langfuse.com")
            except Exception as e:
                print(f"[llm_tracker] Langfuse init failed (non-fatal): {e}")
        else:
            print("[llm_tracker] Langfuse keys not set — local DB tracking only")
        _lf_ready = True
    return _lf


# ── Public API ────────────────────────────────────────────────────────────────

def track_llm_call(
    feature:           str,
    model:             str   = "Llama-3.3-70B-Instruct",
    prompt_tokens:     Optional[int]   = None,
    completion_tokens: Optional[int]   = None,
    latency_ms:        float = 0.0,
    success:           bool  = True,
    error_type:        Optional[str]   = None,
    input_length:      int   = 0,
    output_length:     int   = 0,
    prompt:            str   = "",
    response:          str   = "",
    metadata:          Optional[dict]  = None,
) -> None:
    """
    Record a completed LLM call.  Non-fatal — never raises.
    """
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) or None

    # ── SQLite write (synchronous, fast) ─────────────────────────────────────
    _write_to_db(
        feature, model,
        prompt_tokens, completion_tokens, total_tokens,
        latency_ms, success, error_type, input_length, output_length,
    )

    # ── Langfuse write (async daemon thread — never blocks request) ───────────
    threading.Thread(
        target=_send_to_langfuse,
        args=(feature, model, prompt, response, prompt_tokens,
              completion_tokens, total_tokens, latency_ms, success,
              error_type, metadata),
        daemon=True,
    ).start()


def status_code_suffix(exception: Exception) -> str:
    """
    Enrich a generic exception class name with its HTTP status code when
    available, so the dashboard's error breakdown shows "HTTPError_404"
    instead of just "HTTPError" — makes auth/URL misconfigurations
    immediately visible without digging through Render logs.
    """
    name = type(exception).__name__
    # requests.exceptions.HTTPError → exception.response.status_code
    resp = getattr(exception, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code:
            return f"{name}_{code}"
    # httpx.HTTPStatusError → exception.response.status_code (same shape)
    # openai SDK errors → exception.status_code directly
    code = getattr(exception, "status_code", None)
    if code:
        return f"{name}_{code}"
    return name


def track_error(
    feature:      str,
    model:        str   = "Llama-3.3-70B-Instruct",
    error_type:   str   = "UnknownError",
    latency_ms:   float = 0.0,
    input_length: int   = 0,
) -> None:
    """Convenience wrapper for failed LLM calls."""
    track_llm_call(
        feature=feature, model=model,
        latency_ms=latency_ms, success=False,
        error_type=error_type, input_length=input_length,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _write_to_db(
    feature, model,
    pt, ct, tt,
    latency_ms, success, error_type, input_length, output_length,
) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text
        ts = datetime.now(timezone.utc).isoformat()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO llm_calls
                    (timestamp, feature, model, prompt_tokens, completion_tokens,
                     total_tokens, latency_ms, success, error_type,
                     input_length, output_length)
                VALUES
                    (:ts, :feat, :model, :pt, :ct, :tt, :lat, :ok, :err, :il, :ol)
            """), {
                "ts": ts, "feat": feature, "model": model,
                "pt": pt, "ct": ct, "tt": tt,
                "lat": round(latency_ms, 2),
                "ok": 1 if success else 0,
                "err": error_type,
                "il": input_length, "ol": output_length,
            })
            conn.commit()
    except Exception as e:
        print(f"[llm_tracker] DB write failed (non-fatal): {e}")


def _send_to_langfuse(
    feature, model, prompt, response,
    pt, ct, tt, latency_ms, success, error_type, metadata,
) -> None:
    """
    Langfuse SDK v4 API.
    v2/v3's `lf.trace(...).generation(...)` was removed entirely in v4 —
    the SDK now uses an OpenTelemetry-based observation model:
    `start_observation(as_type="generation")` returns an object you
    `.update()` with output/usage, then `.end()` to close it.
    """
    try:
        lf = _get_langfuse()
        if not lf:
            return
        gen = lf.start_observation(
            name=f"ai_dqm_{feature}",
            as_type="generation",
            model=model,
            model_parameters={"source": "azure_ai_foundry"},
            input=prompt[:3000] if prompt else "",
            metadata={"feature": feature, **(metadata or {})},
        )
        gen.update(
            output=response[:3000] if response else "",
            usage_details={
                "input":  pt or 0,
                "output": ct or 0,
                "total":  tt or 0,
            },
            level="DEFAULT" if success else "ERROR",
            status_message=error_type if not success else None,
            metadata={
                "latency_ms": latency_ms,
                "success":    success,
                "error_type": error_type,
                **(metadata or {}),
            },
        )
        gen.end()
        lf.flush()
    except Exception as e:
        print(f"[llm_tracker] Langfuse send failed (non-fatal): {e}")
