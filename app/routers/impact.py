from fastapi import APIRouter
from app.graph.impact_engine import ImpactEngine

router = APIRouter(prefix="/impact", tags=["Impact"])


@router.get("/{node_id}")
def get_impact(node_id: str):
    return ImpactEngine.calculate_impact(node_id)