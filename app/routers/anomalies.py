# python-backend/app/routers/anomalies.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services import anomalies as anomalies_service

def _notif(title: str, message: str, category: str = "System",
           severity: str = "info", link: str = None, dataset: str = None,
           notif_type: str = "ALERT", source: str = None):
    """Fire-and-forget notification. Never raises."""
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
        create_inbox_notification(
            title=title, message=message, category=category, severity=severity,
            link=link, dataset=dataset, notif_type=notif_type,
            source=source or category,
        )
    except Exception:
        pass



router = APIRouter(prefix="/anomalies", tags=["anomalies"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{dataset_id}/")
def get_anomalies(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return anomalies_service.get_anomalies(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@router.post("/{dataset_id}/analyse/{check_id}")
def analyse_anomaly(dataset_id: int, check_id: int,
                    force: bool = False,
                    db: Session = Depends(get_db)):
    try:
        return anomalies_service.analyse_anomaly(db, dataset_id, check_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class FixPayload(BaseModel):
    selected_remediation: Optional[str] = None


@router.post("/{dataset_id}/fix/{check_id}")
def fix_anomaly(dataset_id: int, check_id: int,
                payload: FixPayload = FixPayload(),
                db: Session = Depends(get_db)):
    """
    Fix anomaly by generating a DQ rule.
    If selected_remediation is provided, the rule is generated to implement
    that specific remediation approach.
    """
    try:
        result = anomalies_service.fix_anomaly(
            db, dataset_id, check_id,
            selected_remediation=payload.selected_remediation,
        )
        # ── Mirror new rule to Delta ─────────────────────────────────────────
        try:
            from app.delta_sync import sync_dq_rule
            from app.models import DQRule
            rule_obj = db.query(DQRule).filter(
                DQRule.dataset_id == dataset_id
            ).order_by(DQRule.id.desc()).first()
            if rule_obj:
                sync_dq_rule(rule_obj)
        except Exception as _e:
            print(f"[delta_sync] anomaly fix rule mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────
        _notif(
            title="Anomaly Fixed",
            message=f"Anomaly check #{check_id} on dataset {dataset_id} has been resolved.",
            category="anomaly", severity="info", link="/anomalies",
            notif_type="ANOMALY", source="Anomaly Engine",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fix failed: {str(e)}")


class StatusPayload(BaseModel):
    status: str  # open | investigating | resolved


@router.patch("/{dataset_id}/status/{check_id}")
def update_status(dataset_id: int, check_id: int, payload: StatusPayload,
                  db: Session = Depends(get_db)):
    try:
        result = anomalies_service.update_anomaly_status(db, dataset_id, check_id, payload.status)
        if payload.status.lower() in ("resolved", "investigating"):
            _notif(
                title=f"Anomaly Status: {payload.status.title()}",
                message=f"Anomaly #{check_id} on dataset {dataset_id} marked as {payload.status}.",
                category="anomaly", severity="info", link="/anomalies",
                notif_type="ANOMALY", source="Anomaly Engine",
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")