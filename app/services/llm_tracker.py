"""
app/services/llm_tracker.py

Central LLM call tracker — dual-write to:
  1. SQLite  llm_calls  table  (primary source for /api/llm-metrics dashboard)
  2. Langfuse Cloud            (optional; enables full trace inspection UI)

FIXED (this version):
  Langfuse Python SDK v4 REMOVED the old `Langfuse().trace()` / `trace.generation()`
  API entirely. Calling them (as the previous version did) raises AttributeError,
  which was being swallowed by a broad except — so every call silently failed to
  reach Langfuse Cloud, even though the SQLite write worked fine.

  v4 uses an OpenTelemetry-based API:
      with langfuse_client.start_as_current_generation(
          name=..., model=..., input=..., model_parameters=...,
      ) as gen:
          gen.update(output=..., usage_details={...}, metadata={...})

  This version uses that API, wrapped so a missing/old SDK or any internal
  Langfuse error still never breaks the request — SQLite write always succeeds
  independently of Langfuse status.

Usage in any service:
    from app.services.llm_tracker import track_llm_call
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
_lf            = None
_lf_ready      = False
_lf_lock       = threading.Lock()
_lf_sdk_v4     = False   # True if the installed SDK exposes the v4 API we need


def _get_langfuse():
    """
    Lazily create and cache a Langfuse client. Detects whether the installed
    SDK exposes the v4 `start_as_current_generation` API. If it doesn't
    (old SDK, or SDK missing entirely), `_lf` stays usable for nothing and
    `_send_to_langfuse` becomes a guaranteed no-op.
    """
    global _lf, _lf_ready, _lf_sdk_v4
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
                client = Langfuse(public_key=pk, secret_key=sk, host=host)
                # v4 API surface check — this is the method we actually need.
                if hasattr(client, "start_as_current_generation"):
                    _lf = client
                    _lf_sdk_v4 = True
                    print("[llm_tracker] ✓ Langfuse v4 connected → cloud.langfuse.com")
                elif hasattr(client, "trace"):
                    # Old v2/v3 SDK installed — still usable, different code path.
                    _lf = client
                    _lf_sdk_v4 = False
                    print("[llm_tracker] ✓ Langfuse legacy (v2/v3) SDK connected")
                else:
                    print("[llm_tracker] Langfuse client has neither v4 nor legacy API — disabling")
                    _lf = None
            except Exception as e:
                print(f"[llm_tracker] Langfuse init failed (non-fatal): {e}")
                _lf = None
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
    Sends one generation event to Langfuse, using whichever API version
    was detected by _get_langfuse(). Always non-fatal.
    """
    try:
        lf = _get_langfuse()
        if not lf:
            return

        meta = {
            "latency_ms": latency_ms,
            "success":    success,
            "error_type": error_type,
            **(metadata or {}),
        }

        if _lf_sdk_v4:
            # ── Langfuse Python SDK v4 (OpenTelemetry-based) ─────────────────
            # start_as_current_generation is a context manager; on exit it
            # flushes the span. We set output/usage via .update() before exit.
            with lf.start_as_current_generation(
                name=f"{feature}_generation",
                model=model,
                input=prompt[:3000] if prompt else "",
                model_parameters={"source": "azure_ai_foundry"},
                metadata={"ai_dqm_feature": feature, **meta},
            ) as gen:
                gen.update(
                    output=response[:3000] if response else "",
                    usage_details={
                        "input":  pt or 0,
                        "output": ct or 0,
                        "total":  tt or 0,
                    },
                )
            # v4 client batches/flushes on its own background thread, but we
            # explicitly flush here since this call runs in a short-lived
            # daemon thread that may exit before the SDK's own flush timer.
            if hasattr(lf, "flush"):
                lf.flush()
        else:
            # ── Legacy (v2/v3) API — kept for backward compatibility ─────────
            trace = lf.trace(
                name=f"ai_dqm_{feature}",
                metadata={"model": model, **meta},
            )
            trace.generation(
                name=f"{feature}_generation",
                model=model,
                model_parameters={"source": "azure_ai_foundry"},
                input=prompt[:3000] if prompt else "",
                output=response[:3000] if response else "",
                usage={"input": pt or 0, "output": ct or 0, "total": tt or 0},
                metadata=meta,
            )
            if hasattr(lf, "flush"):
                lf.flush()

    except Exception as e:
        print(f"[llm_tracker] Langfuse send failed (non-fatal): {e}")
