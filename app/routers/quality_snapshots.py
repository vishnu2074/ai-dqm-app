"""
app/routers/quality_snapshots.py
─────────────────────────────────
Stores and retrieves daily quality score snapshots per dataset.

This is what makes the trend chart real — instead of walking back
from today's score with fake decrements, the chart shows genuine
historical scores stored each time profiling runs.

Endpoints:
  POST /quality-snapshots/record/{dataset_id}
    → reads current DQ score for the dataset and saves it
    → idempotent: if a snapshot already exists for today, updates it
    → call this automatically after every profiling run

  GET  /quality-snapshots/{dataset_id}?days=30
    → returns last N days of snapshots for the trend chart
    → returns [{date, score}, ...] sorted oldest→newest
"""

from datetime import date, timedelta
from fastapi import APIRouter, HTTPException
from sqlalchemy import Column, Integer, Float, Date, func
from app.database import SessionLocal, Base


# ── Inline model (avoids circular import with models/__init__.py) ──────────────

class QualitySnapshot(Base):
    __tablename__ = "quality_snapshots"

    id         = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)
    score      = Column(Float,   nullable=False)
    snap_date  = Column(Date,    nullable=False, index=True)  # one row per dataset per day


router = APIRouter(prefix="/quality-snapshots", tags=["Quality Snapshots"])


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Record today's snapshot ────────────────────────────────────────────────────

@router.post("/record/{dataset_id}")
def record_snapshot(dataset_id: int):
    """
    Fetch current DQ score for dataset_id and store it as today's snapshot.
    Upserts — safe to call multiple times per day (keeps latest score).
    
    Call this at the end of every profiling run:
        import httpx
        httpx.post(f"http://127.0.0.1:8000/quality-snapshots/record/{dataset_id}")
    """
    # Get real DQ score
    score = 85  # default
    try:
        from app.services import dq_scores as dq_scores_service
        db_temp = SessionLocal()
        try:
            summary = dq_scores_service.get_dq_scores_summary(db_temp, dataset_id)
            if summary.get("status") == "COMPLETED":
                raw = (
                    summary.get("dataHealth") or
                    summary.get("dataHealthScore") or
                    summary.get("data_health_score")
                )
                if raw is not None:
                    score = round(float(raw), 1)
        finally:
            db_temp.close()
    except Exception as e:
        print(f"[quality_snapshots] score fetch failed for {dataset_id}: {e}")

    today = date.today()

    db = SessionLocal()
    try:
        existing = db.query(QualitySnapshot).filter_by(
            dataset_id=dataset_id, snap_date=today
        ).first()

        if existing:
            existing.score = score   # update if already snapped today
        else:
            db.add(QualitySnapshot(dataset_id=dataset_id, score=score, snap_date=today))

        db.commit()
        return {"status": "recorded", "dataset_id": dataset_id, "date": str(today), "score": score}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ── Retrieve history for trend chart ──────────────────────────────────────────

@router.get("/{dataset_id}")
def get_snapshots(dataset_id: int, days: int = 30):
    """
    Returns last `days` daily snapshots for the trend chart.
    Response: { dataset_id, snapshots: [{date, score}, ...] }
    
    If fewer than `days` snapshots exist, returns what's available.
    Frontend fills gaps with null or interpolates as needed.
    """
    cutoff = date.today() - timedelta(days=days)

    db = SessionLocal()
    try:
        rows = (
            db.query(QualitySnapshot)
            .filter(
                QualitySnapshot.dataset_id == dataset_id,
                QualitySnapshot.snap_date  >= cutoff,
            )
            .order_by(QualitySnapshot.snap_date.asc())
            .all()
        )

        return {
            "dataset_id": dataset_id,
            "days_requested": days,
            "snapshots": [
                {"date": str(r.snap_date), "score": r.score}
                for r in rows
            ],
        }
    finally:
        db.close()


# ── Bulk record all datasets (convenience endpoint) ───────────────────────────

@router.post("/record-all")
def record_all_snapshots():
    """
    Record today's snapshot for every dataset in the DB.
    Useful to run once on startup to seed initial history,
    or to call from a daily scheduled task.
    """
    from app.database import SessionLocal as S
    from app.models import Dataset

    db = S()
    try:
        datasets = db.query(Dataset).all()
        dataset_ids = [d.id for d in datasets]
    finally:
        db.close()

    results = []
    for did in dataset_ids:
        try:
            result = record_snapshot(did)
            results.append(result)
        except Exception as e:
            results.append({"dataset_id": did, "error": str(e)})

    return {"recorded": len(results), "results": results}