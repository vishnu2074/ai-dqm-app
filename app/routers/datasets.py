from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.schemas import DatasetBase, DatasetResponse
from app.services import datasets as dataset_service

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



router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_datasets(db: Session = Depends(get_db)):
    datasets = dataset_service.get_datasets(db)
    return [DatasetResponse.from_orm(d) for d in datasets]


@router.post("")
def create_dataset(dto: DatasetBase, db: Session = Depends(get_db)):
    try:
        dataset = dataset_service.create_dataset(db, dto)
        ds_name = (dataset.display_name or dataset.physical_name or "Dataset").split("/")[-1]
        _notif(
            title=f"Dataset Registered: {ds_name}",
            message=f"Dataset '{ds_name}' has been registered and is ready for profiling.",
            category="dataset", severity="info", link="/datasets",
            notif_type="ALERT", source="Datasets",
        )
        # ── Mirror to Delta ──────────────────────────────────────────────────
        try:
            from app.delta_sync import sync_dataset
            sync_dataset(dataset)
        except Exception as _e:
            print(f"[delta_sync] dataset mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────
        return DatasetResponse.from_orm(dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Bulk registration ──────────────────────────────────────────────────────────

class BulkFile(BaseModel):
    physical_name: str
    display_name:  str = ""


class BulkRegisterPayload(BaseModel):
    datasource_id: int
    files:         List[BulkFile]


@router.post("/bulk")
def bulk_register_datasets(payload: BulkRegisterPayload, db: Session = Depends(get_db)):
    """
    Register multiple datasets from the same data source in a single request.
    Returns per-file results — skips already-registered files, collects errors,
    never aborts the whole batch for a single failure.
    """
    if not payload.files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(payload.files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files per bulk registration")

    files = [
        {
            "physical_name": f.physical_name,
            "display_name":  f.display_name or f.physical_name.split("/")[-1],
        }
        for f in payload.files
    ]

    try:
        result = dataset_service.bulk_create_datasets(db, payload.datasource_id, files)
        registered = result.get("registered", len(files))
        _notif(
            title=f"Bulk Registration Complete",
            message=f"{registered} dataset(s) registered from data source {payload.datasource_id}.",
            category="dataset", severity="info", link="/datasets",
            notif_type="ALERT", source="Datasets",
        )
        # ── Mirror created datasets to Delta ─────────────────────────────────
        try:
            from app.delta_sync import sync_dataset
            from app.models import Dataset as DatasetModel
            for created in result.get("created", []):
                ds_obj = db.query(DatasetModel).filter(DatasetModel.id == created["id"]).first()
                if ds_obj:
                    sync_dataset(ds_obj)
        except Exception as _e:
            print(f"[delta_sync] bulk dataset mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk registration failed: {str(e)}")


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    try:
        dataset_service.delete_dataset(db, dataset_id)
        return {"message": "Dataset deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))