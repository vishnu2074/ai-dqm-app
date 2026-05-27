"""
app/routers/lineage_edges.py
────────────────────────────
CRUD endpoints for user-defined dataset → dataset lineage connections.

Edges are stored in the `lineage_edges` DB table (not JSON).
Every add/delete busts the lineage graph cache so the next
GET /lineage/graph immediately reflects the change.

Endpoints:
  GET    /lineage/edges          → list all edges + all nodes (for UI dropdowns)
  POST   /lineage/edges          → add a new edge
  DELETE /lineage/edges          → remove an edge
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

# Import model — adjust path if LineageEdge lives inside models.py directly
try:
    from app.models import LineageEdge
except ImportError:
    from app.models.lineage_edge import LineageEdge


router = APIRouter(prefix="/lineage/edges", tags=["Lineage Edges"])


class EdgePayload(BaseModel):
    source: str   # node id  (e.g. "customers_csv")
    target: str   # node id  (e.g. "customer_360_view")


# ── GET ────────────────────────────────────────────────────────────────────────

@router.get("")
def get_edges(db: Session = Depends(get_db)):
    """
    Returns all nodes (for dropdowns) + all user-defined edges.
    The frontend Edge Manager calls this to populate its UI.
    """
    full_graph = LineageEngine.get_full_graph()
    db_edges   = db.query(LineageEdge).all()

    return {
        "nodes": full_graph["nodes"],
        "edges": [
            {"source": e.source, "target": e.target}
            for e in db_edges
        ],
    }


# ── POST ───────────────────────────────────────────────────────────────────────

@router.post("")
def add_edge(payload: EdgePayload, db: Session = Depends(get_db)):
    """Add a directed edge source → target. Idempotent — rejects duplicates."""
    if payload.source == payload.target:
        raise HTTPException(
            status_code=400,
            detail="Source and target cannot be the same node."
        )

    # Validate both node IDs exist in the live graph
    full_graph = LineageEngine.get_full_graph()
    node_ids   = {n["id"] for n in full_graph["nodes"]}

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

    # ── Mirror to Delta ──────────────────────────────────────────────────────
    try:
        from app.delta_sync import sync_lineage_edge
        sync_lineage_edge(edge)
    except Exception as _e:
        print(f"[delta_sync] lineage edge mirror failed (non-fatal): {_e}")
    # ────────────────────────────────────────────────────────────────────────

    # Bust cache so the new edge appears immediately in /lineage/graph
    invalidate_cache()

    return {"status": "created", "source": payload.source, "target": payload.target}


# ── DELETE ─────────────────────────────────────────────────────────────────────

@router.delete("")
def delete_edge(payload: EdgePayload, db: Session = Depends(get_db)):
    """Remove a directed edge source → target."""
    edge = db.query(LineageEdge).filter_by(
        source=payload.source, target=payload.target
    ).first()

    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found.")

    db.delete(edge)
    db.commit()

    # ── Mirror to Delta ──────────────────────────────────────────────────────
    try:
        from app.delta_sync import delete_lineage_edge
        delete_lineage_edge(payload.source, payload.target)
    except Exception as _e:
        print(f"[delta_sync] lineage edge delete mirror failed (non-fatal): {_e}")
    # ────────────────────────────────────────────────────────────────────────

    # Bust cache so the removed edge disappears immediately
    invalidate_cache()

    return {"status": "deleted", "source": payload.source, "target": payload.target}