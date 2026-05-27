from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models import Dataset, DataSource
from app.schemas import DatasetBase


def get_datasets(db: Session):
    return db.query(Dataset).all()


def create_dataset(db: Session, dto: DatasetBase) -> Dataset:
    ds = db.query(DataSource).filter(DataSource.id == dto.datasource_id).first()
    if not ds:
        raise ValueError("Data source not found")

    existing = db.query(Dataset).filter(
        Dataset.datasource_id == dto.datasource_id,
        Dataset.physical_name == dto.physical_name
    ).first()
    if existing:
        raise ValueError("Dataset already registered for this data source")

    dataset = Dataset(
        datasource_id=dto.datasource_id,
        physical_name=dto.physical_name,
        display_name=dto.display_name
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def bulk_create_datasets(
    db: Session,
    datasource_id: int,
    files: List[Dict[str, str]],   # [{physical_name, display_name}, ...]
) -> Dict[str, Any]:
    """
    Register multiple datasets in one call.
    Skips already-registered files (returns them in 'skipped').
    Continues on error — collects all results.
    """
    ds = db.query(DataSource).filter(DataSource.id == datasource_id).first()
    if not ds:
        raise ValueError("Data source not found")

    created  = []
    skipped  = []
    errors   = []

    for f in files:
        physical_name = f.get("physical_name", "")
        display_name  = f.get("display_name", physical_name.split("/")[-1])

        try:
            existing = db.query(Dataset).filter(
                Dataset.datasource_id == datasource_id,
                Dataset.physical_name == physical_name,
            ).first()

            if existing:
                skipped.append({
                    "physical_name": physical_name,
                    "display_name":  display_name,
                    "reason":        "Already registered",
                })
                continue

            dataset = Dataset(
                datasource_id=datasource_id,
                physical_name=physical_name,
                display_name=display_name,
            )
            db.add(dataset)
            db.flush()   # get the id without committing yet
            created.append({
                "id":            dataset.id,
                "physical_name": physical_name,
                "display_name":  display_name,
            })

        except Exception as e:
            db.rollback()
            errors.append({
                "physical_name": physical_name,
                "error":         str(e),
            })
            # re-open a clean transaction
            continue

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise ValueError(f"Bulk commit failed: {e}")

    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "error_count":   len(errors),
        "created":  created,
        "skipped":  skipped,
        "errors":   errors,
    }


def get_datasets_by_datasource(db: Session, datasource_id: int):
    return db.query(Dataset).filter(Dataset.datasource_id == datasource_id).all()


def delete_dataset(db: Session, dataset_id: int) -> bool:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
    db.delete(dataset)
    db.commit()
    return True