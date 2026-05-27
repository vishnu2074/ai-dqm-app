from fastapi import APIRouter, Query
from app.graph.lineage_engine import LineageEngine

router = APIRouter(prefix="/lineage", tags=["Lineage"])


@router.get("/graph")
def get_graph(dataset: str | None = Query(None)):
    if dataset:
        return LineageEngine.get_dataset_graph(dataset)
    return LineageEngine.get_full_graph()


@router.get("/{node_id}")
def get_node_lineage(node_id: str):
    return LineageEngine.get_node_lineage(node_id)