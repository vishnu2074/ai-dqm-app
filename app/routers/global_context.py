from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.schemas import GlobalContextBase, GlobalContextResponse
from app.services import global_context as ctx_service

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_context(db: Session = Depends(get_db)):
    ctx = ctx_service.get_global_context(db)
    return GlobalContextResponse.from_orm(ctx)

@router.post("/")
def set_context(dto: GlobalContextBase, db: Session = Depends(get_db)):
    try:
        ctx = ctx_service.set_global_context(db, dto)
        return GlobalContextResponse.from_orm(ctx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))