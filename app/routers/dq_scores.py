from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services import dq_scores as dq_scores_service

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# DQ SCORING EXECUTION
# ============================================================================

@router.post("/{dataset_id}/run")
def run_dq_scoring(dataset_id: int, db: Session = Depends(get_db)):
    """
    Run DQ scoring on a dataset.
    Computes all metrics from actual data.
    Auto-creates the FIRST baseline for this dataset if none exists yet (handled in service).
    """
    try:
        run = dq_scores_service.run_dq_scoring(db, dataset_id)

        # ── Mirror to Delta ──────────────────────────────────────────────────
        try:
            from app.delta_sync import sync_profiling_run, sync_column_profile, sync_quality_check, sync_drift_record
            from app.models import ProfilingRun, ColumnProfile, QualityCheck, DriftRecord
            run_obj = (db.query(ProfilingRun)
                .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
                .order_by(ProfilingRun.id.desc()).first())
            if run_obj:
                sync_profiling_run(run_obj)
                for cp in db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run_obj.id).all():
                    sync_column_profile(cp, dataset_id)
                for qc in db.query(QualityCheck).filter(QualityCheck.profiling_run_id == run_obj.id).all():
                    sync_quality_check(qc, dataset_id)
                for dr in db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run_obj.id).all():
                    sync_drift_record(dr, dataset_id)
                print(f"[delta_sync] dq_scores run {run_obj.id} mirrored to Delta")
        except Exception as _e:
            print(f"[delta_sync] dq_scores mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────

        response = {
            "status": "success",
            "dqScoreRunId": run.id,
            "timestamp": run.timestamp.isoformat(),
            "message": f"DQ scoring completed in {run.duration_ms}ms",
        }

        # notification
        try:
            from app.routers.notification_inbox_routes import create_inbox_notification
            from app.models import Dataset

            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            ds_name = (ds.display_name or ds.physical_name or f"Dataset {dataset_id}") if ds else f"Dataset {dataset_id}"

            create_inbox_notification(
                title=f"DQ Scoring Completed: {ds_name}",
                message=f"Data quality scoring run completed for '{ds_name}'.",
                category="quality",
                severity="info",
                link="/dq-scores",
                dataset=ds_name,
            )
        except Exception:
            pass

        return response
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        from app.services.datasources import _sanitize_error
        msg = _sanitize_error(str(e))
        raise HTTPException(status_code=500, detail=f"DQ scoring failed: {msg}")


# ============================================================================
# SUMMARY & METRICS
# ============================================================================

@router.get("/{dataset_id}/summary")
def get_summary(dataset_id: int, db: Session = Depends(get_db)):
    """Get DQ score summary for a dataset."""
    try:
        return dq_scores_service.get_dq_scores_summary(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get("/{dataset_id}/columns")
def get_column_scores(dataset_id: int, db: Session = Depends(get_db)):
    """Get detailed column DQ scores for all columns in latest run."""
    try:
        summary = dq_scores_service.get_dq_scores_summary(db, dataset_id)
        if summary.get("status") != "COMPLETED":
            return {
                "status": "NO_DATA",
                "message": "No completed DQ scoring run",
                "columns": [],
            }
        return {
            "status": "success",
            "totalColumns": summary["totalColumns"],
            "columns": summary["columnProfiles"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get column data: {str(e)}")


# ============================================================================
# INCREMENTAL RUNS (History)
# ============================================================================

@router.get("/{dataset_id}/runs")
def get_incremental_runs(dataset_id: int, limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Get historical DQ scoring runs."""
    try:
        runs = dq_scores_service.get_incremental_runs(db, dataset_id, limit=limit)
        return {"status": "success", "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get runs: {str(e)}")


# ============================================================================
# BASELINES
# ============================================================================

@router.get("/{dataset_id}/baselines/status")
def get_baseline_status(dataset_id: int, db: Session = Depends(get_db)):
    """Returns baseline status for the selected dataset."""
    try:
        status = dq_scores_service.get_baseline_status(db, dataset_id)
        return {"status": "success", **status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get baseline status: {str(e)}")


@router.get("/{dataset_id}/baselines/candidates")
def get_baseline_candidates(dataset_id: int, limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Lists completed DQ scoring runs that can be selected as baseline."""
    try:
        candidates = dq_scores_service.get_baseline_candidates(db, dataset_id, limit=limit)
        return {"status": "success", "candidates": candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get baseline candidates: {str(e)}")


@router.get("/{dataset_id}/baselines/comparison")
def get_baseline_comparison(dataset_id: int, db: Session = Depends(get_db)):
    """Compare current metrics against active baseline."""
    try:
        comparison = dq_scores_service.get_baseline_comparison(db, dataset_id)
        return {"status": "success", **comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get baseline comparison: {str(e)}")


@router.post("/{dataset_id}/baselines/set/{run_id}")
def set_baseline(dataset_id: int, run_id: int, db: Session = Depends(get_db)):
    """Set a specific COMPLETED DQ scoring run as the active baseline."""
    try:
        success = dq_scores_service.set_baseline(db, dataset_id, run_id)
        if not success:
            raise HTTPException(status_code=404, detail="Run not found for this dataset")
        return {"status": "success", "message": f"Baseline set from run {run_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set baseline: {str(e)}")


# ============================================================================
# SCHEMA & DRIFT
# ============================================================================

@router.get("/{dataset_id}/schema/history")
def get_schema_history(dataset_id: int, limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Get schema change history."""
    try:
        changes = dq_scores_service.get_schema_history(db, dataset_id, limit=limit)
        return {"status": "success", "changes": changes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema history: {str(e)}")


@router.get("/{dataset_id}/drift")
def get_drift_data(dataset_id: int, limit: int = Query(1000, ge=1, le=10000) , db: Session = Depends(get_db)):
    """Get distribution drift data over time."""
    try:
        drift = dq_scores_service.get_drift_data(db, dataset_id, limit=limit)
        return {"status": "success", "driftRecords": drift}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift data: {str(e)}")


# ============================================================================
# QUALITY CHECKS
# ============================================================================

@router.get("/{dataset_id}/quality-checks")
def get_quality_checks(dataset_id: int, db: Session = Depends(get_db)):
    """Get all quality check violations for latest run."""
    try:
        checks = dq_scores_service.get_quality_checks(db, dataset_id)
        return {"status": "success", "checks": checks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality checks: {str(e)}")


# Backward-compatible alias
@router.get("/{dataset_id}/temporal-checks")
def get_temporal_checks(dataset_id: int, db: Session = Depends(get_db)):
    return get_quality_checks(dataset_id, db)


# ============================================================================
# ADVANCED DRIFT ANALYSIS (NEW)
# ============================================================================

@router.get("/{dataset_id}/drift-analysis")
def get_drift_analysis(
    dataset_id: int,
    current_run_id: int = Query(..., description="Current run ID"),
    previous_run_id: int = Query(..., description="Previous run ID"),
    db: Session = Depends(get_db)
):
    """
    Perform detailed drift analysis between two runs.
    Returns:
      - explanation: natural language summary
      - column_comparison: list of columns with drift scores for both runs
      - column_drilldown: empty (use separate endpoint)
      - distribution_summary: aggregated value counts for the top drifted column
      - using_fallback: boolean indicating whether snapshots were missing
    """
    try:
        result = dq_scores_service.get_drift_analysis(
            db, dataset_id, current_run_id, previous_run_id
        )
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


@router.get("/{dataset_id}/drift-analysis/column/{column_name}")
def get_column_drift_details(
    dataset_id: int,
    column_name: str,
    current_run_id: int = Query(..., description="Current run ID"),
    previous_run_id: int = Query(..., description="Previous run ID"),
    db: Session = Depends(get_db)
):
    """
    Get detailed per-value/bin drift breakdown for a specific column.
    Returns:
      - column: column name
      - values: list of categories or bins with previous and current percentages
      - method: PSI, KS Test, etc.
      - drift_score: overall drift score for this column
    """
    try:
        details = dq_scores_service.get_column_drift_details(
            db, dataset_id, column_name, current_run_id, previous_run_id
        )
        return {"status": "success", **details}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column drift details failed: {str(e)}")


# ============================================================================
# UTILITY / HEALTH
# ============================================================================

@router.get("/{dataset_id}/health")
def get_dq_scores_health(dataset_id: int, db: Session = Depends(get_db)):
    """Quick health check: is DQ scoring data available for this dataset?"""
    try:
        summary = dq_scores_service.get_dq_scores_summary(db, dataset_id)
        return {
            "dataset_id": dataset_id,
            "has_dq_scores": summary.get("status") == "COMPLETED",
            "status": summary.get("status"),
            "message": summary.get("message", "OK"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))