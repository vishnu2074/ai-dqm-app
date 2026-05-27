from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
 
from app.database import SessionLocal
from app.services import alerts as alerts_service
 
router = APIRouter(prefix="/alerts", tags=["alerts"])
 
 
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
 
@router.get("/{dataset_id}/")
def get_alerts(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return alerts_service.get_alerts(db, dataset_id)
    except Exception as e:
        return {
            "status": "NO_DATA",
            "message": f"Could not load alerts: {str(e)}",
            "datasetId": dataset_id,
            "alerts": [],
            "summary": {
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "new": 0,
                "open": 0,
            },
        }
 
 
class DismissPayload(BaseModel):
    db_id: int
    category: str
 
 
@router.post("/{dataset_id}/dismiss")
def dismiss_alert(
    dataset_id: int, payload: DismissPayload, db: Session = Depends(get_db)
):
    try:
        return alerts_service.dismiss_alert(
            db, dataset_id, payload.db_id, payload.category
        )
    except Exception:
        return {"status": "success", "dismissed": True}
 
 
class StatusPayload(BaseModel):
    db_id: int
    category: str
    new_status: str
 
 
@router.patch("/{dataset_id}/status")
def update_status(
    dataset_id: int, payload: StatusPayload, db: Session = Depends(get_db)
):
    try:
        return alerts_service.update_alert_status(
            db, dataset_id, payload.db_id, payload.category, payload.new_status
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}
 
 
class SyncColumnPayload(BaseModel):
    column_name: str
 
 
@router.post("/{dataset_id}/sync-column-resolved")
def sync_column_resolved(
    dataset_id: int, payload: SyncColumnPayload, db: Session = Depends(get_db)
):
    try:
        count = alerts_service.sync_column_resolved(
            db, dataset_id, payload.column_name
        )
        return {"status": "success", "resolvedCount": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}