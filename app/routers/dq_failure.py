from fastapi import APIRouter
from app.graph.dq_failure_engine import DQFailureEngine

router = APIRouter(prefix="/dq-failure", tags=["DQ Failure"])


@router.get("/simulate/{rule_id}")
def simulate_dq_failure(rule_id: str):
    """
    Simulate what happens downstream if a DQ rule fails.
    Returns all affected datasets, reports, and propagation paths.
    """
    return DQFailureEngine.simulate_failure(rule_id)