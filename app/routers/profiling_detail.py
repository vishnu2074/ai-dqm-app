"""
python-backend/app/routers/profiling_detail.py

GET  /{dataset_id}/profile  — page-load fetch, no new run
POST /{dataset_id}/run      — triggers run_dq_scoring (same as DQ Scores Run button)

FIXED: Scheduler now handles missing tables gracefully (governance_system_config, datasets)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.profiling_detail import run_detail_profiling, get_detail_profile
from app.routers.quality_snapshots import record_snapshot

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: int, db: Session = Depends(get_db)):
    """
    Page-load fetch. Returns the latest completed run's profile.
    Returns status='NO_DATA' if no run exists yet — no side effects.
    """
    try:
        return get_detail_profile(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile fetch failed: {str(e)}")


@router.post("/{dataset_id}/run")
def run_profiling(dataset_id: int, db: Session = Depends(get_db)):
    """
    'Run Profiling' button. Calls run_dq_scoring() — identical to DQ Scores run.
    Both pages now share the exact same ProfilingRun row and timestamp.
    Auto-records a quality snapshot after every successful run.
    """
    try:
        result = run_detail_profiling(db, dataset_id)

        # Auto-record today's quality snapshot so the trend chart stays current
        try:
            record_snapshot(dataset_id)
        except Exception as snap_err:
            print(f"[profiling] snapshot record failed (non-fatal): {snap_err}")

        # ── Mirror to Delta ──────────────────────────────────────────────────
        try:
            from app.delta_sync import (sync_profiling_run, sync_column_profile,
                                        sync_quality_check, sync_drift_record)
            from app.models import ProfilingRun, ColumnProfile, QualityCheck, DriftRecord
            run_obj = (db.query(ProfilingRun)
                .filter(ProfilingRun.dataset_id == dataset_id,
                        ProfilingRun.status == "COMPLETED")
                .order_by(ProfilingRun.id.desc()).first())
            if run_obj:
                sync_profiling_run(run_obj)
                col_profiles = db.query(ColumnProfile).filter(
                    ColumnProfile.profiling_run_id == run_obj.id).all()
                for cp in col_profiles:
                    sync_column_profile(cp, dataset_id)
                q_checks = db.query(QualityCheck).filter(
                    QualityCheck.profiling_run_id == run_obj.id).all()
                for qc in q_checks:
                    sync_quality_check(qc, dataset_id)
                drift_recs = db.query(DriftRecord).filter(
                    DriftRecord.profiling_run_id == run_obj.id).all()
                for dr in drift_recs:
                    sync_drift_record(dr, dataset_id)
                print(f"[delta_sync] profiling run {run_obj.id} mirrored: "
                      f"{len(col_profiles)} columns, {len(q_checks)} checks, "
                      f"{len(drift_recs)} drift records")
        except Exception as _e:
            print(f"[delta_sync] profiling mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────

        try:
            from app.routers.notification_inbox_routes import create_inbox_notification
            from app.models import Dataset

            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            ds_name = (ds.display_name or ds.physical_name or f"Dataset {dataset_id}") if ds else f"Dataset {dataset_id}"

            create_inbox_notification(
                title=f"Profiling Completed: {ds_name}",
                message=f"Data profiling run completed for '{ds_name}'. Check the profiling dashboard for results.",
                category="quality",
                severity="info",
                link="/profiling",
                dataset=ds_name,
            )
        except Exception:
            pass

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling run failed: {str(e)}")


# Keep backward-compat alias used by old frontend versions
@router.get("/{dataset_id}/statistical-profile")
def get_statistical_profile(dataset_id: int, db: Session = Depends(get_db)):
    return get_profile(dataset_id, db)


# ── Background scheduler ──────────────────────────────────────────────────────
# Runs in a daemon thread — checks every minute whether any dataset is due
# for an automatic profiling run based on its dq_scoring_schedule setting.

import threading
import time as _time


def _run_profiling_for_dataset(dataset_id: int):
    """Fire a profiling run for a single dataset (called from scheduler thread)."""
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            run_detail_profiling(db, dataset_id)
            print(f"[scheduler] Profiling completed for dataset {dataset_id}")
        finally:
            db.close()
    except Exception as e:
        print(f"[scheduler] Profiling FAILED for dataset {dataset_id}: {e}")


def _scheduler_loop():
    """
    Daemon loop — runs forever, checks for due profiling jobs every 60s.
    FIXED: Now handles missing tables gracefully (governance_system_config, datasets).
    """
    print("[scheduler] Background profiling scheduler started")
    while True:
        try:
            from app.database import engine
            from sqlalchemy import text

            # ── Get schedule config (handle missing table) ────────────────────
            schedule = "daily"  # default
            try:
                with engine.connect() as conn:
                    schedule_val = conn.execute(
                        text("SELECT value FROM governance_system_config WHERE key='dq_scoring_schedule'")
                    ).scalar()
                    if schedule_val:
                        schedule = schedule_val.strip().lower()
            except Exception as config_err:
                # Table doesn't exist yet or other error — use default
                if "no such table" in str(config_err).lower():
                    print(f"[scheduler] governance_system_config table not found, using default schedule: {schedule}")
                else:
                    print(f"[scheduler] Error reading schedule config: {config_err}")

            # ── Determine if we should run now ────────────────────────────────
            now = _time.localtime()
            should_run = False

            if schedule == "hourly":
                should_run = (now.tm_min == 0)
            elif schedule == "daily":
                should_run = (now.tm_hour == 2 and now.tm_min == 0)  # 2 AM daily
            elif schedule == "weekly":
                should_run = (now.tm_wday == 0 and now.tm_hour == 2 and now.tm_min == 0)  # Monday 2 AM
            # "manual" or anything else → never auto-run

            # ── Get datasets to profile (handle missing table) ────────────────
            if should_run:
                dataset_ids = []
                try:
                    with engine.connect() as conn:
                        dataset_ids = [
                            row[0] for row in
                            conn.execute(text("SELECT id FROM datasets")).fetchall()
                        ]
                except Exception as ds_err:
                    if "no such table" in str(ds_err).lower():
                        print(f"[scheduler] datasets table not found, skipping scheduled run")
                    else:
                        print(f"[scheduler] Error reading datasets: {ds_err}")

                # ── Fire profiling runs for each dataset ──────────────────────
                for did in dataset_ids:
                    threading.Thread(
                        target=_run_profiling_for_dataset,
                        args=(did,),
                        daemon=True,
                    ).start()

        except Exception as e:
            print(f"[scheduler] Loop error: {e}")

        _time.sleep(60)  # check every minute


def start_scheduler():
    """Call once from main.py on app startup to begin background scheduling."""
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()