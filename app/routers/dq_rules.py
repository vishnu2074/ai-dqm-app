# python-backend/app/routers/dq_rules.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
from app.database import SessionLocal
from app.services import dq_rules as dq_rules_service
from app.services.ai_recommendations import simulate_ai_rule

def _notif(title: str, message: str, category: str = "System",
           severity: str = "info", link: str = None, dataset: str = None,
           notif_type: str = "ALERT", source: str = None):
    """Fire-and-forget notification. Never raises."""
    try:
        from app.routers.notification_inbox_routes import create_inbox_notification
        create_inbox_notification(
            title=title, message=message, category=category, severity=severity,
            link=link, dataset=dataset, notif_type=notif_type,
            source=source or category,
        )
    except Exception:
        pass



router = APIRouter(
    prefix="/dq-rules",
    tags=["dq-rules"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{dataset_id}/summary")
def get_rules_summary(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.get_rules_summary(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules summary: {str(e)}")


@router.get("/{dataset_id}/active")
def get_active_rules(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.get_active_rules(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active rules: {str(e)}")


@router.get("/{dataset_id}/history")
def get_rule_history(dataset_id: int, limit: int = 50, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.get_rule_history(db, dataset_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rule history: {str(e)}")


@router.get("/{dataset_id}/discovered")
def discovered_rules(dataset_id: int, db: Session = Depends(get_db)):
    try:
        result = dq_rules_service.get_discovered_rules(db, dataset_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/columns")
def get_dataset_columns(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.get_dataset_columns(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset columns: {str(e)}")


# ==========================
# ✅ AI Recommended Rules
# ==========================

@router.get("/{dataset_id}/ai-recommended")
def get_ai_recommended_rules(dataset_id: int, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.get_ai_recommended_rules(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SimulateAIRulePayload(BaseModel):
    rule: Dict[str, Any]


@router.post("/{dataset_id}/ai-recommended/simulate")
def simulate_ai_recommended_rule(dataset_id: int, payload: SimulateAIRulePayload, db: Session = Depends(get_db)):
    """
    Simulates passRate + violations for a candidate rule using dq_engine._apply_rule
    """
    try:
        return simulate_ai_rule(db, dataset_id, payload.rule)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simulate rule: {str(e)}")


class ApproveAIRulePayload(BaseModel):
    rule: Dict[str, Any]


@router.post("/{dataset_id}/ai-recommended/approve")
def approve_ai_recommended_rule(dataset_id: int, payload: ApproveAIRulePayload, db: Session = Depends(get_db)):
    """
    Approves AI rule -> saves into dq_rules table (as DSL rule)
    and removes the pending AI recommendation entry.
    """
    try:
        return dq_rules_service.approve_ai_recommended_rule(db, dataset_id, payload.rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve AI rule: {str(e)}")


# ==========================
# Manual Rule CRUD
# ==========================

class CreateDQRulePayload(BaseModel):
    input_mode: Literal["nl", "regex", "dsl"] = Field(..., description="nl | regex | dsl")
    text: str = Field(..., min_length=1, description="NL text OR regex pattern OR DSL condition")
    name: Optional[str] = None
    type: Optional[str] = None
    column: Optional[str] = None
    severity: Optional[str] = "Medium"
    status: Optional[str] = "Active"


@router.post("/{dataset_id}/rules")
def create_rule(dataset_id: int, payload: CreateDQRulePayload, db: Session = Depends(get_db)):
    try:
        result = dq_rules_service.create_rule(
            db,
            dataset_id,
            input_mode=payload.input_mode,
            text=payload.text,
            name=payload.name,
            rule_type=payload.type,
            column=payload.column,
            severity=payload.severity or "Medium",
            status=payload.status or "Active",
        )
        rule_name = payload.name or payload.text[:40]
        _notif(
            title=f"DQ Rule Created: {rule_name}",
            message=f"New DQ rule '{rule_name}' created for dataset {dataset_id}.",
            category="rule", severity="info", link="/dq-rules",
            notif_type="ALERT", source="DQ Rules",
        )
        # ── Mirror to Delta ──────────────────────────────────────────────────
        try:
            from app.delta_sync import sync_dq_rule
            from app.models import DQRule
            rule_obj = db.query(DQRule).filter(
                DQRule.dataset_id == dataset_id
            ).order_by(DQRule.id.desc()).first()
            if rule_obj:
                sync_dq_rule(rule_obj)
        except Exception as _e:
            print(f"[delta_sync] dq_rule mirror failed (non-fatal): {_e}")
        # ────────────────────────────────────────────────────────────────────
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")


class UpdateDQRulePayload(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    column: Optional[str] = None
    condition: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None


@router.put("/{dataset_id}/rules/{rule_code}")
def update_rule(dataset_id: int, rule_code: str, payload: UpdateDQRulePayload, db: Session = Depends(get_db)):
    try:
        return dq_rules_service.update_rule(db, dataset_id, rule_code, payload.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {str(e)}")


@router.delete("/{dataset_id}/rules/{rule_code}")
def delete_rule(dataset_id: int, rule_code: str, db: Session = Depends(get_db)):
    try:
        # Before deleting, find the rule's DB id so we can re-open linked anomalies
        from app.models import DQRule, QualityCheck
        rule_obj = db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id,
            DQRule.rule_code == rule_code,
        ).first()

        result = dq_rules_service.delete_rule(db, dataset_id, rule_code)

        # Re-open any anomalies that were resolved by this rule
        # ondelete="SET NULL" sets resolved_by_rule_id=NULL automatically
        # but status stays "resolved" — we must flip it back to "open"
        if rule_obj:
            affected = db.query(QualityCheck).filter(
                QualityCheck.resolved_by_rule_id == None,  # noqa: E711
                QualityCheck.status == "resolved",
            ).all()
            # Filter: checks that HAD this rule (now NULL after cascade)
            # We identify them by checking checks with no rule id that were resolved
            # More precisely: re-query checks linked to this dataset's runs
            from app.models import ProfilingRun
            run_ids = [r.id for r in db.query(ProfilingRun).filter(
                ProfilingRun.dataset_id == dataset_id
            ).all()]
            reopened = db.query(QualityCheck).filter(
                QualityCheck.profiling_run_id.in_(run_ids),
                QualityCheck.resolved_by_rule_id == None,  # noqa: E711
                QualityCheck.status == "resolved",
            ).all()
            for chk in reopened:
                chk.status = "open"
            if reopened:
                db.commit()
                print(f"[dq_rules] Re-opened {len(reopened)} anomalies after deleting rule {rule_code}")

        _notif(
            title=f"DQ Rule Deleted: {rule_code}",
            message=f"Rule '{rule_code}' was permanently deleted from dataset {dataset_id}.",
            category="rule", severity="warning", link="/dq-rules",
            notif_type="ALERT", source="DQ Rules",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {str(e)}")