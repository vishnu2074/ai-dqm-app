from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.schemas import DataSourceCreate, DataSourceTest, DataSourceResponse
from app.services import datasources as ds_service

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

@router.post("/test")
def test_connection(dto: DataSourceTest):
    result = ds_service.test_connection(dto)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@router.post("")
def create_datasource(dto: DataSourceCreate, db: Session = Depends(get_db)):
    try:
        ds = ds_service.create_datasource(db, dto)
        _notif(
            title=f"Data Source Connected: {ds.name}",
            message=f"Data source '{ds.name}' ({ds.type}) connected successfully.",
            category="datasource", severity="info", link="/data-sources",
            notif_type="ALERT", source="Data Sources",
        )
        # ── Mirror to Delta ──────────────────────────────────────────────────
        try:
            from app.delta_sync import sync_datasource
            sync_datasource(ds)
        except Exception as _e:
            print(f"[delta_sync] datasource mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────
        return DataSourceResponse.from_orm(ds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("")
def get_datasources(db: Session = Depends(get_db)):
    dss = ds_service.get_datasources(db)
    return [DataSourceResponse.from_orm(ds) for ds in dss]

@router.get("/{id}/files")
def list_files(id: int, db: Session = Depends(get_db)):
    try:
        files = ds_service.list_physical_datasets(db, id)
        return files
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{id}")
def delete_datasource(id: int, db: Session = Depends(get_db)):
    try:
        ds_service.delete_datasource(db, id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/{id}/toggle")
def toggle_datasource(id: int, db: Session = Depends(get_db)):
    try:
        ds = ds_service.toggle_datasource(db, id)
        return DataSourceResponse.from_orm(ds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))