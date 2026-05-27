from sqlalchemy.orm import Session
from app.models import GlobalContext, DataSource, Dataset
from app.schemas import GlobalContextBase

def get_global_context(db: Session):
    # For now, assume single context, id=1
    ctx = db.query(GlobalContext).first()
    if not ctx:
        ctx = GlobalContext()
        db.add(ctx)
        db.commit()
        db.refresh(ctx)
    return ctx

def set_global_context(db: Session, dto: GlobalContextBase):
    ctx = get_global_context(db)
    if dto.active_datasource_id:
        ds = db.query(DataSource).filter(DataSource.id == dto.active_datasource_id).first()
        if not ds:
            raise ValueError("Data source not found")
    if dto.active_dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == dto.active_dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found")
        if dto.active_datasource_id and dataset.datasource_id != dto.active_datasource_id:
            raise ValueError("Dataset does not belong to selected data source")
    ctx.active_datasource_id = dto.active_datasource_id
    ctx.active_dataset_id = dto.active_dataset_id
    db.commit()
    db.refresh(ctx)
    return ctx