"""
app/routers/lineage_edges.py
CRUD endpoints for user-defined dataset → dataset lineage connections.
FIXED: Made validation lenient - allows edges even if graph is empty
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from app.graph.lineage_engine import LineageEngine, invalidate_cache

try:
    from app.models import LineageEdge
except ImportError:
    from app.models.lineage_edge import LineageEdge

router = APIRouter(prefix="/lineage/edges", tags=["Lineage Edges"])

class EdgePayload(BaseModel):
    source: str
    target: str

@router.get("")
def get_edges(db: Session = Depends(get_db)):
    full_graph = LineageEngine.get_full_graph()
    db_edges = db.query(LineageEdge).all()
    return {
        "nodes": full_graph["nodes"],
        "edges": [
            {"source": e.source, "target": e.target}
            for e in db_edges
        ],
    }

@router.post("")
def add_edge(payload: EdgePayload, db: Session = Depends(get_db)):
    """Add a directed edge source → target. FIXED: Lenient validation."""
    if payload.source == payload.target:
        raise HTTPException(
            status_code=400,
            detail="Source and target cannot be the same node."
        )

    # FIXED: Get graph but don't fail if it's empty
    try:
        full_graph = LineageEngine.get_full_graph()
        node_ids = {n["id"] for n in full_graph.get("nodes", [])}
    except Exception as e:
        print(f"[lineage] Warning: Could not get lineage graph: {e}")
        node_ids = set()

    # FIXED: Only validate if we have nodes to validate against
    if node_ids:
        if payload.source not in node_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Source node '{payload.source}' not found in lineage graph."
            )
        if payload.target not in node_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Target node '{payload.target}' not found in lineage graph."
            )

    # Reject duplicate
    existing = db.query(LineageEdge).filter_by(
        source=payload.source, target=payload.target
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="This connection already exists.")

    edge = LineageEdge(source=payload.source, target=payload.target)
    db.add(edge)
    db.commit()

    # Mirror to Delta
    try:
        from app.delta_sync import sync_lineage_edge
        sync_lineage_edge(edge)
    except Exception as _e:
        print(f"[delta_sync] lineage edge mirror failed (non-fatal): {_e}")

    invalidate_cache()

    return {"status": "created", "source": payload.source, "target": payload.target}

@router.delete("")
def delete_edge(payload: EdgePayload, db: Session = Depends(get_db)):
    edge = db.query(LineageEdge).filter_by(
        source=payload.source, target=payload.target
    ).first()
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found.")

    db.delete(edge)
    db.commit()

    try:
        from app.delta_sync import delete_lineage_edge
        delete_lineage_edge(payload.source, payload.target)
    except Exception as _e:
        print(f"[delta_sync] lineage edge delete mirror failed (non-fatal): {_e}")

    invalidate_cache()

    return {"status": "deleted", "source": payload.source, "target": payload.target}