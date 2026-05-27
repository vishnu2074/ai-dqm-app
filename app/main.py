# app/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as _FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path
from sqlalchemy import text

from app.database import engine, Base, seed_governance_data


def _bootstrap_sqlite_schema():
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='dq_rules'")
            ).fetchone()
            if not exists:
                return
            cols = conn.execute(text("PRAGMA table_info(dq_rules)")).fetchall()
            existing = {row[1] for row in cols}
            if "input_mode" not in existing:
                conn.exec_driver_sql(
                    "ALTER TABLE dq_rules ADD COLUMN input_mode VARCHAR NOT NULL DEFAULT 'manual'"
                )
            if "nl_text" not in existing:
                conn.exec_driver_sql("ALTER TABLE dq_rules ADD COLUMN nl_text TEXT")
            if "regex_pattern" not in existing:
                conn.exec_driver_sql("ALTER TABLE dq_rules ADD COLUMN regex_pattern TEXT")
            if "meta" not in existing:
                conn.exec_driver_sql("ALTER TABLE dq_rules ADD COLUMN meta JSON")
            conn.commit()
    except Exception as e:
        print(f"Schema bootstrap warning: {e}")


# ── Create tables and seed ────────────────────────────────────────────────────
try:
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _bootstrap_sqlite_schema()
    seed_governance_data()
    # Start periodic DB backup to Azure Blob (every 5 minutes)
    from app.database import start_periodic_backup
    start_periodic_backup(interval_seconds=300)
except Exception as e:
    print(f"DB bootstrap warning (non-fatal): {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI DQM Backend", version="2.5.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
_DATABRICKS_URL = os.getenv("DATABRICKS_APP_URL", "")

# Health dashboard URLs — add your Render health-dashboard URL here after deploy
_HEALTH_DASHBOARD_URLS = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:8001",
]
_RENDER_HEALTH_DASHBOARD = os.getenv("HEALTH_DASHBOARD_URL", "")
if _RENDER_HEALTH_DASHBOARD:
    _HEALTH_DASHBOARD_URLS.append(_RENDER_HEALTH_DASHBOARD)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        *_HEALTH_DASHBOARD_URLS,
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API health (before SPA catch-all) ────────────────────────────────────────
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "version": "2.5.0"}

@app.get("/api/health", include_in_schema=False)
def api_health():
    return {"status": "ok", "version": "2.5.0"}

@app.post("/api/backup-db", include_in_schema=False)
def backup_db():
    """Manually trigger a DB backup to Azure Blob."""
    from app.database import upload_db_to_blob
    ok = upload_db_to_blob()
    return {"status": "ok" if ok else "skipped", "blob": "ai-dqm/ai_dqm.db"}

# ── Routers ───────────────────────────────────────────────────────────────────

try:
    from app.routers import datasources
    app.include_router(datasources.router, prefix="/datasources", tags=["datasources"])
    print("Datasources router loaded")
except Exception as e:
    print(f"Datasources router failed: {e}")

try:
    from app.routers import datasets
    app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
    print("Datasets router loaded")
except Exception as e:
    print(f"Datasets router failed: {e}")

try:
    from app.routers import global_context
    app.include_router(global_context.router, prefix="/context", tags=["context"])
    print("Global context router loaded")
except Exception as e:
    print(f"Global context router failed: {e}")

try:
    from app.routers import lineage
    app.include_router(lineage.router)
    print("Lineage router loaded")
except Exception as e:
    print(f"Lineage router failed: {e}")

try:
    from app.routers import impact
    app.include_router(impact.router)
    print("Impact router loaded")
except Exception as e:
    print(f"Impact router failed: {e}")

try:
    from app.routers.dq_failure import router as dq_failure_router
    app.include_router(dq_failure_router)
    print("DQ Failure router loaded")
except Exception as e:
    print(f"DQ Failure router failed: {e}")

try:
    from app.routers.lineage_edges import router as lineage_edges_router
    app.include_router(lineage_edges_router)
    print("Lineage Edges router loaded")
except Exception as e:
    print(f"Lineage Edges router failed: {e}")

try:
    from app.routers.governance_routes import governance_router
    app.include_router(governance_router, prefix="/governance", tags=["governance"])
    print("Governance router loaded")
except Exception as e:
    print(f"Governance router failed: {e}")

try:
    from app.routers.notification_inbox_routes import notification_inbox_router
    app.include_router(notification_inbox_router, tags=["notifications"])
    print("Notification Inbox router loaded")
except Exception as e:
    print(f"Notification Inbox router failed: {e}")

try:
    from app.routers.policy_suggestions_routes import policy_suggestions_router
    app.include_router(policy_suggestions_router)
    print("Policy Suggestions router loaded")
except Exception as e:
    print(f"Policy Suggestions router failed: {e}")

try:
    from app.routers import dq_scores
    app.include_router(dq_scores.router, prefix="/dq-scores", tags=["dq-scores"])
    print("DQ Scores router loaded")
except Exception as e:
    print(f"DQ Scores router failed: {e}")

try:
    from app.routers import profiling_detail
    app.include_router(profiling_detail.router, prefix="/profiling-detail", tags=["profiling-detail"])
    print("Profiling Detail router loaded")
except Exception as e:
    print(f"Profiling Detail router failed: {e}")

try:
    from app.routers import dq_rules
    app.include_router(dq_rules.router)
    print("DQ Rules router loaded")
except Exception as e:
    print(f"DQ Rules router failed: {e}")

try:
    from app.routers import dq_engine
    app.include_router(dq_engine.router)
    print("DQ Engine router loaded")
except Exception as e:
    print(f"DQ Engine router failed: {e}")

try:
    from app.routers.quality_snapshots import router as quality_snapshots_router, QualitySnapshot
    QualitySnapshot.__table__.create(bind=engine, checkfirst=True)
    app.include_router(quality_snapshots_router)
    print("Quality Snapshots router loaded")
except Exception as e:
    print(f"Quality Snapshots router failed: {e}")

try:
    from app.routers import anomalies
    app.include_router(anomalies.router)
    print("Anomalies router loaded")
except Exception as e:
    print(f"Anomalies router failed: {e}")

try:
    from app.routers import scorecards
    app.include_router(scorecards.router, prefix="/scorecards", tags=["scorecards"])
    print("Scorecards router loaded")
except Exception as e:
    print(f"Scorecards router failed: {e}")

try:
    from app.routers import monitoring
    app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
    print("Monitoring router loaded")
except Exception as e:
    print(f"Monitoring router failed: {e}")

try:
    from app.routers import alerts
    app.include_router(alerts.router)
    print("Alerts router loaded")
except Exception as e:
    print(f"Alerts router failed: {e}")

try:
    from app.routers import knowledge_graph
    from app.models import KnowledgeGraphEdge
    KnowledgeGraphEdge.__table__.create(bind=engine, checkfirst=True)
    app.include_router(knowledge_graph.router)
    print("Knowledge Graph router loaded")
except Exception as e:
    print(f"Knowledge Graph router failed: {e}")

try:
    from app.routers import ai_agent
    app.include_router(ai_agent.router)
    print("AI Agent router loaded")
except Exception as e:
    print(f"AI Agent router failed: {e}")

try:
    from app.routers import overview_dashboard
    app.include_router(overview_dashboard.router, prefix="/api")
    print("Overview Dashboard router loaded")
except Exception as e:
    print(f"Overview Dashboard router failed: {e}")

# ── Profiling scheduler ───────────────────────────────────────────────────────
try:
    from app.routers.profiling_detail import start_scheduler
    start_scheduler()
    print("Profiling scheduler started")
except Exception as e:
    print(f"Profiling scheduler failed to start: {e}")

# ── Health Metrics Router (for AI DQM Health Dashboard) ──────────────────────
try:
    from app.routers.health_metrics_router import router as health_metrics_router
    app.include_router(health_metrics_router)
    print("Health Metrics router loaded at /api/health-metrics")
except Exception as e:
    print(f"Health metrics router failed: {e}")


# ── Frontend static files — MUST be registered AFTER all API routers ─────────
_THIS_FILE   = _Path(__file__).resolve()
_APP_DIR     = _THIS_FILE.parent
_SOURCE_ROOT = _APP_DIR.parent

_FRONTEND_DIST = _APP_DIR / "static"
if not (_FRONTEND_DIST / "index.html").exists():
    _FRONTEND_DIST = _SOURCE_ROOT / "Frontend v25" / "dist"

print(f"[startup] Frontend dist: {_FRONTEND_DIST}")
print(f"[startup] index.html exists: {(_FRONTEND_DIST / 'index.html').exists()}")

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
        return _FileResponse(str(_FRONTEND_DIST / "index.html"))

    print("[startup] SPA catch-all registered — frontend active")
else:
    print("[startup] WARNING: index.html not found — serving API only")
    print(f"[startup] Checked: {_FRONTEND_DIST / 'index.html'}")

print("[startup] AI DQM Backend started with Health Metrics support")