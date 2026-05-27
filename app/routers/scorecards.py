from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.services import scorecards as svc
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{dataset_id}/full")
def get_full_scorecard(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_full_scorecard(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /full for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/kpi")
def get_kpi(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_kpi_summary(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /kpi for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/trend")
def get_trend(dataset_id: int, days: int = 90, db: Session = Depends(get_db)):
    try:
        return {"trend": svc.get_quality_trend(db, dataset_id, days=days)}
    except Exception as e:
        logger.error(f"Error in /trend for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/risk-contributors")
def get_risk(dataset_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    try:
        return {"contributors": svc.get_risk_contributors(db, dataset_id, top_n=top_n)}
    except Exception as e:
        logger.error(f"Error in /risk-contributors for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/velocity")
def get_velocity(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_quality_velocity(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /velocity for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/rules-coverage")
def get_coverage(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_rules_coverage(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /rules-coverage for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/violation-heatmap")
def get_heatmap(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_violation_heatmap(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /violation-heatmap for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/freshness")
def get_freshness(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_freshness_score(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /freshness for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/schema-stability")
def get_schema_stability(dataset_id: int, days: int = 7, db: Session = Depends(get_db)):
    try:
        return svc.get_schema_stability(db, dataset_id, days=days)
    except Exception as e:
        logger.error(f"Error in /schema-stability for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/drift-kpi")
def get_drift_kpi(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_drift_kpi(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /drift-kpi for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/run-comparison")
def get_run_comparison(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_run_comparison(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /run-comparison for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/incident-timeline")
def get_incident_timeline(dataset_id: int, days: int = 30, db: Session = Depends(get_db)):
    try:
        return svc.get_incident_timeline(db, dataset_id, days=days)
    except Exception as e:
        logger.error(f"Error in /incident-timeline for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/column-risk-table")
def get_column_risk_table(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return svc.get_column_risk_table(db, dataset_id)
    except Exception as e:
        logger.error(f"Error in /column-risk-table for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/null-trend")
def get_null_trend(dataset_id: int, days: int = 90, db: Session = Depends(get_db)):
    try:
        return svc.get_null_trend(db, dataset_id, days=days)
    except Exception as e:
        logger.error(f"Error in /null-trend for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{dataset_id}/generate-report")
def generate_report(dataset_id: int, db: Session = Depends(get_db)):
    from app.models import Dataset
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        name = dataset.display_name or dataset.physical_name if dataset else f"Dataset {dataset_id}"
        return svc.generate_ai_report(db, dataset_id, name)
    except Exception as e:
        logger.error(f"Error in /generate-report for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── SLA target endpoints (global — not per-dataset) ───────────────────────────

@router.get("/{dataset_id}/sla-target")
def get_sla_target(dataset_id: int, db: Session = Depends(get_db)):
    """Return the persisted overall health SLA target (default 85). dataset_id unused."""
    try:
        return svc.get_scorecard_sla_target(db)
    except Exception as e:
        logger.error(f"Error in /sla-target: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{dataset_id}/sla-target")
def save_sla_target(dataset_id: int, body: dict, db: Session = Depends(get_db)):
    """Persist overall health SLA target. Body: { target: number }"""
    try:
        target = int(body.get("target", 85))
        return svc.save_scorecard_sla_target(db, target)
    except Exception as e:
        logger.error(f"Error in PUT /sla-target: {e}")
        raise HTTPException(status_code=500, detail=str(e))