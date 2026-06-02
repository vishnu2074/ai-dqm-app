# app/main.py

import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as _FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path

# ── App — created FIRST before any imports that might fail ───────────────────
app = FastAPI(title="AI DQM Backend", version="2.5.0")

# ── CORS ─────────────────────────────────────────────────────────────────────
_HEALTH_DASHBOARD_URL = os.getenv("HEALTH_DASHBOARD_URL", "")
_extra = [_HEALTH_DASHBOARD_URL] if _HEALTH_DASHBOARD_URL else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:5174",
        "http://localhost:8001",
        *_extra,
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup errors collector ──────────────────────────────────────────────────
_STARTUP_ERRORS: list[str] = []

# ── Health endpoints — registered immediately, before anything else ───────────
@app.get("/health", include_in_schema=False)
@app.get("/api/health", include_in_schema=False)
def api_health():
    return {
        "status": "ok",
        "version": "2.5.0",
        "startup_errors": _STARTUP_ERRORS,
        "python": sys.version,
    }

@app.post("/api/backup-db", include_in_schema=False)
def backup_db():
    try:
        from app.database import upload_db_to_blob
        ok = upload_db_to_blob()
        return {"status": "ok" if ok else "skipped", "blob": "ai-dqm/ai_dqm.db"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Database bootstrap ────────────────────────────────────────────────────────
try:
    from sqlalchemy import text
    from app.database import engine, Base, seed_governance_data

    def _bootstrap_sqlite_schema():
        try:
            with engine.connect() as conn:
                # ── Original dq_rules columns ─────────────────────────────────
                exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='dq_rules'")
                ).fetchone()
                if exists:
                    cols = conn.execute(text("PRAGMA table_info(dq_rules)")).fetchall()
                    existing = {row[1] for row in cols}
                    for col, defn in [
                        ("input_mode", "VARCHAR NOT NULL DEFAULT 'manual'"),
                        ("nl_text",    "TEXT"),
                        ("regex_pattern", "TEXT"),
                        ("meta",       "JSON"),
                    ]:
                        if col not in existing:
                            conn.exec_driver_sql(f"ALTER TABLE dq_rules ADD COLUMN {col} {defn}")

                # ── ADDED: profiling_runs timing columns ──────────────────────
                # Required by health_metrics_router to compute:
                #   avg_llm_latency_ms, avg_profiling_runtime_s,
                #   avg_job_duration_ms, api_throughput
                # Without these, all timing metrics remain 0.
                pr_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='profiling_runs'")
                ).fetchone()
                if pr_exists:
                    pr_cols = conn.execute(text("PRAGMA table_info(profiling_runs)")).fetchall()
                    pr_existing = {row[1] for row in pr_cols}
                    for col, defn in [
                        ("started_at",   "TEXT"),        # ISO-8601 UTC timestamp
                        ("completed_at", "TEXT"),        # ISO-8601 UTC timestamp
                        ("duration_ms",  "INTEGER"),     # explicit ms if tracked directly
                    ]:
                        if col not in pr_existing:
                            try:
                                conn.exec_driver_sql(
                                    f"ALTER TABLE profiling_runs ADD COLUMN {col} {defn}"
                                )
                                print(f"[bootstrap] Added column profiling_runs.{col}")
                            except Exception as col_err:
                                # Column may already exist in some environments
                                _STARTUP_ERRORS.append(f"add_col_{col}: {col_err}")

                conn.commit()
        except Exception as e:
            _STARTUP_ERRORS.append(f"schema_bootstrap: {e}")

    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _bootstrap_sqlite_schema()
    seed_governance_data()
    try:
        from app.database import start_periodic_backup
        start_periodic_backup(interval_seconds=300)
    except Exception as e:
        _STARTUP_ERRORS.append(f"periodic_backup: {e}")

    # ── ADDED: Extract real DB path from SQLAlchemy engine URL ───────────────
    # health_metrics_router uses raw sqlite3 + DB_PATH env var.
    # We extract the path here so both layers always point to the same file,
    # regardless of what DATABASE_URL is set to in Render.
    try:
        _db_url = str(engine.url)
        if _db_url.startswith("sqlite:////"):
            # Four slashes = absolute path on Unix: sqlite:////tmp/...
            _db_path = "/" + _db_url[len("sqlite:////"):]
        elif _db_url.startswith("sqlite:///"):
            # Three slashes = relative path
            _db_path = _db_url[len("sqlite:///"):]
        else:
            _db_path = "/tmp/ai-dqm/ai_dqm.db"  # safe fallback

        os.environ["DB_PATH"] = _db_path
        print(f"[startup] DB_PATH set to: {_db_path}")
    except Exception as e:
        # Fallback: use the default path the router already knows about
        os.environ.setdefault("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")
        _STARTUP_ERRORS.append(f"db_path_extract: {e}")

    print("[startup] DB bootstrap complete")

except Exception as e:
    msg = f"DB bootstrap FAILED: {e}"
    _STARTUP_ERRORS.append(msg)
    print(f"[startup] WARNING: {msg}")
    # Still set a default DB_PATH so the health-metrics router can attempt to connect
    os.environ.setdefault("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")


# ── ADDED: LLM client factory (Azure AI Foundry — Llama 3.3 70B) ─────────────
# Replaces any AzureOpenAI client initialization scattered through profiling code.
# Import this wherever you make LLM calls:
#
#   from app.main import get_llm_client
#   client = get_llm_client()
#   if client:
#       response = client.chat.completions.create(
#           model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "Llama-3.3-70B-Instruct"),
#           messages=[{"role": "user", "content": prompt}],
#           max_tokens=500,
#       )
#
# WHY THIS FIX:
#   AzureOpenAI constructs URLs as:
#     {endpoint}/openai/deployments/{model}/chat/completions?api-version=xxx
#   This returns 404 for Llama models on Azure AI Foundry.
#   Foundry uses the OpenAI-compatible path:
#     {endpoint}/v1/chat/completions   (no api-version, Bearer auth)
#
_llm_client_instance = None

def get_llm_client():
    """
    Returns an OpenAI-compatible client for Azure AI Foundry (Llama/non-GPT models).
    Returns None if env vars are not configured — callers must handle None gracefully.
    Thread-safe: client is created once and reused.
    """
    global _llm_client_instance
    if _llm_client_instance is not None:
        return _llm_client_instance

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY",  "")

    if not endpoint or not api_key:
        print("[llm] WARNING: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set — LLM disabled")
        return None

    try:
        from openai import OpenAI
        _llm_client_instance = OpenAI(
            base_url=f"{endpoint}/v1",  # ← Foundry path, NOT /openai/deployments/
            api_key=api_key,            # ← sent as Bearer token automatically
        )
        print(f"[llm] Client initialised → {endpoint}/v1")
        return _llm_client_instance
    except Exception as e:
        _STARTUP_ERRORS.append(f"llm_client_init: {e}")
        print(f"[llm] ERROR initialising client: {e}")
        return None


# Eagerly attempt client creation at startup so any misconfiguration
# is logged immediately rather than silently at first LLM call.
get_llm_client()


# ── Router loader helper ──────────────────────────────────────────────────────
def _load(label: str, fn):
    try:
        fn()
        print(f"[router] ✓ {label}")
    except Exception as e:
        msg = f"{label}: {e}"
        _STARTUP_ERRORS.append(msg)
        print(f"[router] ✗ {msg}")


# ── Routers ───────────────────────────────────────────────────────────────────
def _reg_datasources():
    from app.routers import datasources
    app.include_router(datasources.router, prefix="/datasources", tags=["datasources"])
_load("datasources", _reg_datasources)

def _reg_datasets():
    from app.routers import datasets
    app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
_load("datasets", _reg_datasets)

def _reg_global_context():
    from app.routers import global_context
    app.include_router(global_context.router, prefix="/context", tags=["context"])
_load("global_context", _reg_global_context)

def _reg_lineage():
    from app.routers import lineage
    app.include_router(lineage.router)
_load("lineage", _reg_lineage)

def _reg_impact():
    from app.routers import impact
    app.include_router(impact.router)
_load("impact", _reg_impact)

def _reg_dq_failure():
    from app.routers.dq_failure import router as r
    app.include_router(r)
_load("dq_failure", _reg_dq_failure)

def _reg_lineage_edges():
    from app.routers.lineage_edges import router as r
    app.include_router(r)
_load("lineage_edges", _reg_lineage_edges)

def _reg_governance():
    from app.routers.governance_routes import governance_router
    app.include_router(governance_router, prefix="/governance", tags=["governance"])
_load("governance", _reg_governance)

def _reg_notifications():
    from app.routers.notification_inbox_routes import notification_inbox_router
    app.include_router(notification_inbox_router, tags=["notifications"])
_load("notifications", _reg_notifications)

def _reg_policy():
    from app.routers.policy_suggestions_routes import policy_suggestions_router
    app.include_router(policy_suggestions_router)
_load("policy_suggestions", _reg_policy)

def _reg_dq_scores():
    from app.routers import dq_scores
    app.include_router(dq_scores.router, prefix="/dq-scores", tags=["dq-scores"])
_load("dq_scores", _reg_dq_scores)

def _reg_profiling():
    from app.routers import profiling_detail
    app.include_router(profiling_detail.router, prefix="/profiling-detail", tags=["profiling-detail"])
_load("profiling_detail", _reg_profiling)

def _reg_dq_rules():
    from app.routers import dq_rules
    app.include_router(dq_rules.router)
_load("dq_rules", _reg_dq_rules)

def _reg_dq_engine():
    from app.routers import dq_engine
    app.include_router(dq_engine.router)
_load("dq_engine", _reg_dq_engine)

def _reg_quality_snapshots():
    from app.routers.quality_snapshots import router as r, QualitySnapshot
    from app.database import engine as _engine
    QualitySnapshot.__table__.create(bind=_engine, checkfirst=True)
    app.include_router(r)
_load("quality_snapshots", _reg_quality_snapshots)

def _reg_anomalies():
    from app.routers import anomalies
    app.include_router(anomalies.router)
_load("anomalies", _reg_anomalies)

def _reg_scorecards():
    from app.routers import scorecards
    app.include_router(scorecards.router, prefix="/scorecards", tags=["scorecards"])
_load("scorecards", _reg_scorecards)

def _reg_monitoring():
    from app.routers import monitoring
    app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
_load("monitoring", _reg_monitoring)

def _reg_alerts():
    from app.routers import alerts
    app.include_router(alerts.router)
_load("alerts", _reg_alerts)

def _reg_kg():
    from app.routers import knowledge_graph
    from app.models import KnowledgeGraphEdge
    from app.database import engine as _engine
    KnowledgeGraphEdge.__table__.create(bind=_engine, checkfirst=True)
    app.include_router(knowledge_graph.router)
_load("knowledge_graph", _reg_kg)

def _reg_ai_agent():
    from app.routers import ai_agent
    app.include_router(ai_agent.router)
_load("ai_agent", _reg_ai_agent)

def _reg_overview():
    from app.routers import overview_dashboard
    app.include_router(overview_dashboard.router, prefix="/api")
_load("overview_dashboard", _reg_overview)

# ── Profiling scheduler ───────────────────────────────────────────────────────
def _start_scheduler():
    from app.routers.profiling_detail import start_scheduler
    start_scheduler()
_load("profiling_scheduler", _start_scheduler)

# ── Health Metrics Router ─────────────────────────────────────────────────────
# Registers GET /api/health-metrics
# No prefix — the router already defines the full /api/health-metrics path.
# DB_PATH is set above from the SQLAlchemy engine URL so the router's
# raw sqlite3 connection hits the same database file.
def _reg_health_metrics():
    from app.routers.health_metrics_router import router as hm_router
    app.include_router(hm_router)
_load("health_metrics_router", _reg_health_metrics)


# ── Frontend static files — registered LAST, after ALL API routes ─────────────
_THIS_FILE   = _Path(__file__).resolve()
_APP_DIR     = _THIS_FILE.parent
_SOURCE_ROOT = _APP_DIR.parent

_FRONTEND_DIST = _APP_DIR / "static"
if not (_FRONTEND_DIST / "index.html").exists():
    _FRONTEND_DIST = _SOURCE_ROOT / "Frontend v25" / "dist"

print(f"[startup] Frontend dist:     {_FRONTEND_DIST}")
print(f"[startup] index.html exists: {(_FRONTEND_DIST / 'index.html').exists()}")
print(f"[startup] Startup errors:    {len(_STARTUP_ERRORS)}")
for err in _STARTUP_ERRORS:
    print(f"[startup]   ✗ {err}")

if (_FRONTEND_DIST / "index.html").exists():
    _assets = _FRONTEND_DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")
        print(f"[startup] Mounted /assets → {_assets}")

    @app.get("/favicon.ico", include_in_schema=False)
    def _favicon():
        _f = _FRONTEND_DIST / "favicon.ico"
        return _FileResponse(str(_f)) if _f.exists() else _FileResponse(str(_FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def _serve_spa(full_path: str):
        # IMPORTANT: Never intercept /api/* — let FastAPI 404 naturally for unknown API paths.
        # This guard is a safety net; all real /api/ routes are registered above
        # and FastAPI matches them before this catch-all runs.
        # The ONLY way this guard fires is if someone requests a genuinely
        # non-existent /api/... path — return proper JSON 404 in that case.
        if full_path.startswith("api/") or full_path == "api":
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return _FileResponse(str(_FRONTEND_DIST / "index.html"))

    print("[startup] SPA catch-all registered — frontend active")
else:
    print("[startup] WARNING: index.html not found — API-only mode")

print("[startup] AI DQM Backend ready ✓")