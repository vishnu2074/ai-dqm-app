# python-backend/app/routers/dq_engine.py
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services import dq_engine as dq_engine_service

router = APIRouter(prefix="/dq-engine", tags=["dq-engine"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class PreviewPayload(BaseModel):
    version_id: Optional[int] = None
    rule_codes: Optional[List[str]] = None
    mode: str = "flag"  # flag | filter
    preview_rows: int = 25
    samples_per_rule: int = 10


@router.post("/{dataset_id}/preview")
def preview_apply(dataset_id: int, payload: PreviewPayload, db: Session = Depends(get_db)):
    try:
        return dq_engine_service.preview_apply_rules(
            db=db,
            dataset_id=dataset_id,
            version_id=payload.version_id,
            rule_codes=payload.rule_codes,
            mode=payload.mode,
            preview_rows=payload.preview_rows,
            samples_per_rule=payload.samples_per_rule,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview apply failed: {str(e)}")


class SavePayload(BaseModel):
    run_id: int
    description: Optional[str] = None
    created_by: str = "Admin"


@router.post("/{dataset_id}/save")
def save_run(dataset_id: int, payload: SavePayload, db: Session = Depends(get_db)):
    try:
        return dq_engine_service.save_preview_as_new_version(
            db=db,
            dataset_id=dataset_id,
            run_id=payload.run_id,
            description=payload.description,
            created_by=payload.created_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.get("/{dataset_id}/runs/{run_id}/download")
def download_preview_csv(dataset_id: int, run_id: int, db: Session = Depends(get_db)):
    try:
        path = dq_engine_service.get_preview_file_path(db, dataset_id, run_id)
        filename = f"dataset_{dataset_id}_dq_preview_run_{run_id}.csv"
        return FileResponse(path, media_type="text/csv", filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")