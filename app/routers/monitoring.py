# python-backend/app/routers/monitoring.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services import monitoring as monitoring_service

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── SLA threshold helpers ──────────────────────────────────────────────────────

_SLA_KEYS = [
    "sla_threshold_completeness",
    "sla_threshold_uniqueness",
    "sla_threshold_validity",
    "sla_threshold_consistency",
    "sla_threshold_accuracy",
    "sla_threshold_integrity",
    "sla_warning_offset",
]

_SLA_DEFAULTS = {
    "completeness": 80, "uniqueness": 50, "validity": 80,
    "consistency": 80, "accuracy": 80, "integrity": 80,
    "warning_offset": 5,
}


def _read_sla_thresholds(db: Session) -> dict:
    """Read SLA thresholds from governance_system_config.
    Returns None if no rows found so the service falls back to its own defaults.
    """
    try:
        from app.routers.governance_routes import GovernanceSystemConfig
        rows = {r.key: r.value for r in db.query(GovernanceSystemConfig)
                .filter(GovernanceSystemConfig.key.in_(_SLA_KEYS)).all()}
        if not rows:
            return None
        def _i(key, default):
            try: return int(rows[key])
            except (KeyError, ValueError, TypeError): return default
        return {
            "completeness":   _i("sla_threshold_completeness", 80),
            "uniqueness":     _i("sla_threshold_uniqueness",   50),
            "validity":       _i("sla_threshold_validity",     80),
            "consistency":    _i("sla_threshold_consistency",  80),
            "accuracy":       _i("sla_threshold_accuracy",     80),
            "integrity":      _i("sla_threshold_integrity",    80),
            "warning_offset": _i("sla_warning_offset",          5),
        }
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/summary")
def get_summary(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_monitoring_summary(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring summary: {str(e)}")


@router.get("/{dataset_id}/metrics-trends")
def get_metrics_trends(dataset_id: int, limit: int = 30, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_metrics_trends(db, dataset_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics trends: {str(e)}")


@router.get("/{dataset_id}/drift")
def get_drift_monitoring(dataset_id: int, limit: int = 30, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_drift_monitoring(db, dataset_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift data: {str(e)}")


@router.get("/{dataset_id}/risk-forecast")
def get_risk_forecast(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_risk_forecast(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk forecast: {str(e)}")


@router.get("/{dataset_id}/column-health")
def get_column_health(dataset_id: int, limit: int = 20, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_column_health(db, dataset_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get column health: {str(e)}")


@router.get("/{dataset_id}/sla-status")
def get_sla_status(dataset_id: int, limit: int = 30, db: Session = Depends(get_db)):
    """SLA & Threshold Breach Tracker — thresholds read from governance_system_config."""
    try:
        thresholds = _read_sla_thresholds(db)
        return monitoring_service.get_sla_status(db, dataset_id, limit=limit, thresholds=thresholds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SLA status: {str(e)}")


@router.get("/{dataset_id}/sla-thresholds")
def get_sla_thresholds(dataset_id: int, db: Session = Depends(get_db)):
    """Return current SLA thresholds so the frontend can pre-fill the inline editor."""
    try:
        t = _read_sla_thresholds(db)
        return t if t is not None else dict(_SLA_DEFAULTS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SLA thresholds: {str(e)}")


@router.put("/{dataset_id}/sla-thresholds")
def save_sla_thresholds(dataset_id: int, body: dict, db: Session = Depends(get_db)):
    """
    Save SLA thresholds globally to governance_system_config.
    Body: { completeness, uniqueness, validity, consistency, accuracy, integrity, warning_offset }
    dataset_id is accepted for route symmetry but thresholds are global across datasets.
    """
    try:
        from app.routers.governance_routes import GovernanceSystemConfig
        key_map = {
            "completeness":   "sla_threshold_completeness",
            "uniqueness":     "sla_threshold_uniqueness",
            "validity":       "sla_threshold_validity",
            "consistency":    "sla_threshold_consistency",
            "accuracy":       "sla_threshold_accuracy",
            "integrity":      "sla_threshold_integrity",
            "warning_offset": "sla_warning_offset",
        }
        for field, db_key in key_map.items():
            if field in body:
                val = max(0, min(100, int(body[field])))
                existing = db.query(GovernanceSystemConfig).filter(
                    GovernanceSystemConfig.key == db_key
                ).first()
                if existing:
                    existing.value = str(val)
                else:
                    db.add(GovernanceSystemConfig(key=db_key, value=str(val)))
        db.commit()
        return _read_sla_thresholds(db) or dict(_SLA_DEFAULTS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save SLA thresholds: {str(e)}")


@router.get("/{dataset_id}/compare-runs")
def compare_runs(dataset_id: int, run_a: int, run_b: int, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_compare_runs(db, dataset_id, run_a, run_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare runs: {str(e)}")


@router.get("/{dataset_id}/execution-runs")
def get_execution_runs(dataset_id: int, limit: int = 50, db: Session = Depends(get_db)):
    try:
        return monitoring_service.get_execution_runs(db, dataset_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution runs: {str(e)}")


@router.post("/{dataset_id}/run-manual-check")
def run_manual_check(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return monitoring_service.trigger_manual_check(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        from app.services.datasources import _sanitize_error
        msg = _sanitize_error(str(e))
        raise HTTPException(status_code=500, detail=f"Manual check failed: {msg}")