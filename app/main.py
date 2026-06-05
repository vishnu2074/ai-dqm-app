# app/main.py

import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as _FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path

app = FastAPI(title="AI DQM Backend", version="3.1.0")

_HEALTH_DASHBOARD_URL = os.getenv("HEALTH_DASHBOARD_URL", "")
_extra = [_HEALTH_DASHBOARD_URL] if _HEALTH_DASHBOARD_URL else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:3000",
        "http://127.0.0.1:5173", "http://127.0.0.1:3000",
        "http://localhost:5174", "http://localhost:8001",
        *_extra, "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STARTUP_ERRORS: list[str] = []

@app.get("/health", include_in_schema=False)
@app.get("/api/health", include_in_schema=False)
def api_health():
    return {
        "status": "ok",
        "version": "3.1.0",
        "startup_errors": _STARTUP_ERRORS,
        "python": sys.version,
    }

@app.post("/admin/reset-database")
async def reset_database():
    """Reset database - deletes all data and recreates tables"""
    import os
    from pathlib import Path
    
    db_path = os.getenv("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")
    
    # Close all connections first
    from app.database import engine
    engine.dispose()
    
    # Delete the file
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✓ Deleted database at {db_path}")
    
    # Recreate tables
    from app.database import Base
    Base.metadata.create_all(bind=engine)
    
    return {
        "status": "ok",
        "message": "Database reset successfully",
        "db_path": db_path
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
                # ── 1. dq_rules columns ──────────────────────────────────────
                exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='dq_rules'")
                ).fetchone()
                if exists:
                    cols = conn.execute(text("PRAGMA table_info(dq_rules)")).fetchall()
                    existing = {row[1] for row in cols}
                    for col, defn in [
                        ("input_mode", "VARCHAR NOT NULL DEFAULT 'manual'"),
                        ("nl_text", "TEXT"),
                        ("regex_pattern", "TEXT"),
                        ("meta", "JSON"),
                    ]:
                        if col not in existing:
                            conn.exec_driver_sql(f"ALTER TABLE dq_rules ADD COLUMN {col} {defn}")

                # ── 2. profiling_runs columns ────────────────────────────────
                pr_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='profiling_runs'")
                ).fetchone()
                if pr_exists:
                    pr_cols = conn.execute(text("PRAGMA table_info(profiling_runs)")).fetchall()
                    pr_existing = {row[1] for row in pr_cols}

                    for col, defn in [
                        ("started_at", "TEXT"),
                        ("completed_at", "TEXT"),
                        ("duration_ms", "INTEGER"),
                        ("ai_summary", "TEXT"),
                    ]:
                        if col not in pr_existing:
                            try:
                                conn.exec_driver_sql(f"ALTER TABLE profiling_runs ADD COLUMN {col} {defn}")
                                print(f"[bootstrap] Added column profiling_runs.{col}")
                            except Exception as col_err:
                                _STARTUP_ERRORS.append(f"add_col_pr_{col}: {col_err}")

                    # ── LEGITIMATE BACKFILL: Reconstruct timestamps ──────────
                    try:
                        null_check = conn.execute(
                            text("SELECT COUNT(*) FROM profiling_runs WHERE started_at IS NULL AND timestamp IS NOT NULL")
                        ).fetchone()

                        if null_check and null_check[0] > 0:
                            print(f"[bootstrap] Backfilling {null_check[0]} profiling_runs timestamps...")

                            conn.exec_driver_sql("""
                                UPDATE profiling_runs
                                SET started_at = strftime('%Y-%m-%d %H:%M:%S', timestamp)
                                WHERE started_at IS NULL AND timestamp IS NOT NULL
                            """)

                            conn.exec_driver_sql("""
                                UPDATE profiling_runs
                                SET completed_at = datetime(
                                    started_at,
                                    '+' || MAX(1, CAST(ROUND(COALESCE(duration_ms, 1000) / 1000.0) AS INTEGER)) || ' seconds'
                                )
                                WHERE completed_at IS NULL AND started_at IS NOT NULL AND timestamp IS NOT NULL
                            """)

                            conn.commit()
                            print("[bootstrap] ✓ Timestamp backfill complete")
                    except Exception as backfill_err:
                        _STARTUP_ERRORS.append(f"timestamp_backfill: {backfill_err}")

                # ── 3. column_profiles columns ───────────────────────────────
                cp_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='column_profiles'")
                ).fetchone()
                if cp_exists:
                    cp_cols = conn.execute(text("PRAGMA table_info(column_profiles)")).fetchall()
                    cp_existing = {row[1] for row in cp_cols}

                    for col, defn in [
                        ("sensitivity_label", "TEXT DEFAULT 'Public'"),
                        ("ai_description", "TEXT"),
                    ]:
                        if col not in cp_existing:
                            try:
                                conn.exec_driver_sql(f"ALTER TABLE column_profiles ADD COLUMN {col} {defn}")
                                print(f"[bootstrap] Added column column_profiles.{col}")
                            except Exception as cp_err:
                                _STARTUP_ERRORS.append(f"add_col_cp_{col}: {cp_err}")

                # ── 4. drift_records created_at column ───────────────────────
                dr_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='drift_records'")
                ).fetchone()
                if dr_exists:
                    dr_cols = conn.execute(text("PRAGMA table_info(drift_records)")).fetchall()
                    dr_existing = {row[1] for row in dr_cols}

                    if "created_at" not in dr_existing:
                        try:
                            conn.exec_driver_sql("ALTER TABLE drift_records ADD COLUMN created_at TEXT")
                            print("[bootstrap] Added column drift_records.created_at")

                            conn.exec_driver_sql("""
                                UPDATE drift_records
                                SET created_at = (
                                    SELECT strftime('%Y-%m-%d %H:%M:%S', timestamp)
                                    FROM profiling_runs
                                    WHERE id = drift_records.profiling_run_id
                                )
                                WHERE created_at IS NULL AND profiling_run_id IS NOT NULL
                            """)
                            conn.commit()
                            print("[bootstrap] ✓ drift_records.created_at backfill complete")
                        except Exception as dr_err:
                            _STARTUP_ERRORS.append(f"drift_created_at: {dr_err}")

                # ── 5. notification_inbox: add missing columns ───────────────
                ni_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='notification_inbox'")
                ).fetchone()
                if ni_exists:
                    ni_cols = conn.execute(text("PRAGMA table_info(notification_inbox)")).fetchall()
                    ni_existing = {row[1] for row in ni_cols}

                    for col, defn in [
                        ("timestamp", "TEXT"),
                        ("dataset", "TEXT"),
                        ("source", "TEXT"),
                        ("view_route", "TEXT"),
                        ("is_read", "INTEGER DEFAULT 0"),
                        ("is_archived", "INTEGER DEFAULT 0"),
                        ("link", "TEXT"),
                    ]:
                        if col not in ni_existing:
                            try:
                                conn.exec_driver_sql(f"ALTER TABLE notification_inbox ADD COLUMN {col} {defn}")
                                print(f"[bootstrap] Added column notification_inbox.{col}")
                            except Exception:
                                pass  # Column might already exist

                    # Copy created_at to timestamp for existing rows
                    if "timestamp" in ni_existing or "created_at" in ni_existing:
                        try:
                            conn.exec_driver_sql("""
                                UPDATE notification_inbox
                                SET timestamp = created_at
                                WHERE timestamp IS NULL AND created_at IS NOT NULL
                            """)
                            conn.commit()
                        except Exception:
                            pass

                # ── 6. Create governance_system_config table if missing ──────
                gsc_exists = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='governance_system_config'")
                ).fetchone()
                if not gsc_exists:
                    try:
                        conn.exec_driver_sql("""
                            CREATE TABLE governance_system_config (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                key TEXT UNIQUE NOT NULL,
                                value TEXT,
                                description TEXT,
                                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        print("[bootstrap] ✓ Created governance_system_config table")

                        # Insert default config values
                        default_configs = [
                            ("dq_scoring_schedule", "daily", "Schedule for automatic DQ scoring: hourly, daily, weekly, manual"),
                            ("email_notifications_enabled", "false", "Enable email notifications for alerts"),
                            ("slack_webhook_url", "", "Slack webhook URL for notifications"),
                            ("max_profiling_rows", "1000000", "Maximum rows to process in a single profiling run"),
                        ]
                        for key, value, desc in default_configs:
                            conn.exec_driver_sql("""
                                INSERT OR IGNORE INTO governance_system_config (key, value, description)
                                VALUES (:key, :value, :desc)
                            """, {"key": key, "value": value, "desc": desc})
                        conn.commit()
                    except Exception as gsc_err:
                        _STARTUP_ERRORS.append(f"governance_system_config: {gsc_err}")

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

    # ── Extract real DB path ─────────────────────────────────────────────────
    try:
        _db_url = str(engine.url)
        if _db_url.startswith("sqlite:////"):
            _db_path = "/" + _db_url[len("sqlite:////"):]
        elif _db_url.startswith("sqlite:///"):
            _db_path = _db_url[len("sqlite:///"):]
        else:
            _db_path = "/tmp/ai-dqm/ai_dqm.db"

        os.environ["DB_PATH"] = _db_path
        print(f"[startup] DB_PATH set to: {_db_path}")
    except Exception as e:
        os.environ.setdefault("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")
        _STARTUP_ERRORS.append(f"db_path_extract: {e}")

    print("[startup] DB bootstrap complete")

except Exception as e:
    msg = f"DB bootstrap FAILED: {e}"
    _STARTUP_ERRORS.append(msg)
    print(f"[startup] WARNING: {msg}")
    os.environ.setdefault("DB_PATH", "/tmp/ai-dqm/ai_dqm.db")


# ── LLM client factory ─────────────────────────────────────────────────────
_llm_client_instance = None

def get_llm_client():
    global _llm_client_instance
    if _llm_client_instance is not None:
        return _llm_client_instance

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    if not endpoint or not api_key:
        print("[llm] WARNING: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set — LLM disabled")
        return None

    try:
        from openai import OpenAI
        _llm_client_instance = OpenAI(
            base_url=f"{endpoint}/v1",
            api_key=api_key,
        )
        print(f"[llm] Client initialised → {endpoint}/v1")
        return _llm_client_instance
    except Exception as e:
        _STARTUP_ERRORS.append(f"llm_client_init: {e}")
        print(f"[llm] ERROR initialising client: {e}")
        return None

get_llm_client()


# ── Router loader helper ────────────────────────────────────────────────────
def _load(label: str, fn):
    try:
        fn()
        print(f"[router] ✓ {label}")
    except Exception as e:
        msg = f"{label}: {e}"
        _STARTUP_ERRORS.append(msg)
        print(f"[router] ✗ {msg}")


# ── Routers ─────────────────────────────────────────────────────────────────
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

def _start_scheduler():
    from app.routers.profiling_detail import start_scheduler
    start_scheduler()
_load("profiling_scheduler", _start_scheduler)

def _reg_health_metrics():
    from app.routers.health_metrics_router import router as hm_router
    app.include_router(hm_router)
_load("health_metrics_router", _reg_health_metrics)


# ── Frontend static files ───────────────────────────────────────────────────
_THIS_FILE = _Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
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
        if full_path.startswith("api/") or full_path == "api":
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return _FileResponse(str(_FRONTEND_DIST / "index.html"))

    print("[startup] SPA catch-all registered — frontend active")
else:
    print("[startup] WARNING: index.html not found — API-only mode")

print("[startup] AI DQM Backend ready ✓")