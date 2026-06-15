# python-backend/app/services/ai_agent.py
"""
AI Copilot — Proper Graph-Based Multi-Agent Architecture

Graph nodes with conditional edges (LangGraph-style, no extra dependency):

    ┌─────────────────────────────────────────────────────────────┐
    │  MASTER LLM  — reads message, returns route + ack           │
    └──────┬──────────────────────────────────────────────────────┘
           │ routes to:
    ┌──────▼───────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐
    │ QUERY AGENT  │ │DIAGNOSE AGENT│ │ ACTION AGENT │ │REPORT AGENT │
    │ Q&A, scores, │ │ Root cause,  │ │ Plan+execute │ │ Full struct. │
    │ scorecards,  │ │ evidence,    │ │ rules, alerts│ │ report with  │
    │ monitoring,  │ │ fix steps    │ │ anomaly fix, │ │ all sections │
    │ lineage, KG  │ │              │ │ KG create,   │ │              │
    └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘
           │                │                │                │
           └────────────────┴────────────────┴────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  SUGGESTION AGENT  │
                          │  Follow-ups + next │
                          │  action sidebar    │
                          └────────────────────┘

Persistence:
  - In-memory session memory per dataset_id (cleared on new chat)
  - SQLite-backed conversation history (survives restarts)
  - Pinned messages stored in DB
  - Feedback stored in DB

Folder mode:
  - Resolves all dataset_ids under the folder from the DB
  - Builds aggregated context from ALL datasets in the folder
  - Responses are grounded only in real DB data — no hallucination
"""

import json
import os
import re
import uuid
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict
from pathlib import Path

from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import (
    Dataset, ProfilingRun, ColumnProfile,
    QualityCheck, DriftRecord, SchemaHistory,
    DQRule, DQRuleRun, DQRuleRunResult,
)
import time as _time
from app.services.llm_tracker import track_llm_call

# ─── LLM ──────────────────────────────────────────────────────────────────────

_MODEL = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )


def _llm(system: str, user: str, history: List[Dict] = None, max_tokens: int = 900) -> str:
    _t0 = _time.time()
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history[-8:])
    msgs.append({"role": "user", "content": user})
    try:
        resp = _client().chat.completions.create(
            model=_MODEL, messages=msgs, temperature=0.3, max_tokens=max_tokens,
        )
        out = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        track_llm_call(
            feature="agent", model=_MODEL,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            latency_ms=(_time.time() - _t0) * 1000,
            success=True, input_length=len(user), output_length=len(out),
        )
        return out
    except Exception as e:
        track_llm_call(
            feature="agent", model=_MODEL,
            latency_ms=(_time.time() - _t0) * 1000,
            success=False, error_type=type(e).__name__, input_length=len(user),
        )
        return f"LLM error: {e}"


def _llm_json(system: str, user: str, max_tokens: int = 600) -> Any:
    raw = _llm(system, user, max_tokens=max_tokens).strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None


# ─── Persistent Storage (SQLite) ──────────────────────────────────────────────

_DB_PATH = Path(os.getenv("AGENT_HISTORY_DB", "agent_history.db"))


def _get_hist_conn():
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_history_db():
    conn = _get_hist_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id  INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            intent      TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS folder_chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            folder      TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            intent      TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS pinned_messages (
            id          TEXT PRIMARY KEY,
            dataset_id  INTEGER NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS agent_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  TEXT,
            dataset_id  INTEGER,
            rating      INTEGER,
            comment     TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_chat_dataset   ON chat_history(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_folder_chat    ON folder_chat_history(folder);
        CREATE INDEX IF NOT EXISTS idx_pin_dataset    ON pinned_messages(dataset_id);
    """)
    conn.commit()
    conn.close()


try:
    _init_history_db()
except Exception as _e:
    print(f"[agent] History DB init failed (non-fatal): {_e}")


def _save_history(dataset_id: int, role: str, content: str, intent: str = ""):
    try:
        conn = _get_hist_conn()
        conn.execute(
            "INSERT INTO chat_history (dataset_id, role, content, intent) VALUES (?,?,?,?)",
            (dataset_id, role, content[:8000], intent),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _save_folder_history(folder: str, role: str, content: str, intent: str = ""):
    try:
        conn = _get_hist_conn()
        conn.execute(
            "INSERT INTO folder_chat_history (folder, role, content, intent) VALUES (?,?,?,?)",
            (folder, role, content[:8000], intent),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_history(dataset_id: int, limit: int = 40) -> List[Dict]:
    """Load persisted history for a dataset — survives server restarts."""
    try:
        conn = _get_hist_conn()
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
            (dataset_id, limit),
        ).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception:
        return []


def load_folder_history(folder: str, limit: int = 40) -> List[Dict]:
    """Load persisted history for a folder."""
    try:
        conn = _get_hist_conn()
        rows = conn.execute(
            "SELECT role, content FROM folder_chat_history WHERE folder=? ORDER BY id DESC LIMIT ?",
            (folder, limit),
        ).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception:
        return []


def clear_history(dataset_id: int):
    try:
        conn = _get_hist_conn()
        conn.execute("DELETE FROM chat_history WHERE dataset_id=?", (dataset_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def clear_folder_history(folder: str):
    try:
        conn = _get_hist_conn()
        conn.execute("DELETE FROM folder_chat_history WHERE folder=?", (folder,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def save_pin(dataset_id: int, msg_id: str, content: str):
    try:
        conn = _get_hist_conn()
        conn.execute(
            "INSERT OR REPLACE INTO pinned_messages (id, dataset_id, content) VALUES (?,?,?)",
            (msg_id, dataset_id, content[:8000]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def delete_pin(dataset_id: int, msg_id: str):
    try:
        conn = _get_hist_conn()
        conn.execute("DELETE FROM pinned_messages WHERE dataset_id=? AND id=?", (dataset_id, msg_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_pins_db(dataset_id: int) -> List[Dict]:
    try:
        conn = _get_hist_conn()
        rows = conn.execute(
            "SELECT id, content, created_at FROM pinned_messages WHERE dataset_id=? ORDER BY created_at DESC",
            (dataset_id,),
        ).fetchall()
        conn.close()
        return [{"id": r["id"], "content": r["content"], "timestamp": r["created_at"]} for r in rows]
    except Exception:
        return []


def save_feedback(message_id: str, dataset_id: int, rating: int, comment: str):
    try:
        conn = _get_hist_conn()
        conn.execute(
            "INSERT INTO agent_feedback (message_id, dataset_id, rating, comment) VALUES (?,?,?,?)",
            (message_id, dataset_id, rating, comment or ""),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─── In-memory stores ─────────────────────────────────────────────────────────

_pending_actions: Dict[str, Dict] = {}
MAX_HISTORY = 40


# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    message:             str
    dataset_id:          Optional[int]
    history:             List[Dict]
    route:               str
    tool_results:        Dict[str, Any]
    response:            str
    suggested_followups: List[str]
    suggested_actions:   List[Dict]
    pending_actions:     List[Dict]
    chart_data:          Optional[Dict]
    redirect:            Optional[Dict]


# ─── Data Tools ───────────────────────────────────────────────────────────────

class DataTools:
    def __init__(self, db: Session):
        self.db = db

    # ── Folder helpers ────────────────────────────────────────────────────────

    def resolve_folder_datasets(self, folder: str) -> List[Dict]:
        """
        Return all datasets whose physical_name starts with dqm/raw/<folder>/
        Also try display_name matching as fallback.
        Returns list of {id, name, physical_name}.
        """
        prefix = f"dqm/raw/{folder}/"
        datasets = (
            self.db.query(Dataset)
            .filter(Dataset.physical_name.like(f"{prefix}%"))
            .all()
        )
        # Fallback: try display_name or physical_name containing the folder name
        if not datasets:
            datasets = (
                self.db.query(Dataset)
                .filter(
                    (Dataset.physical_name.like(f"%{folder}%")) |
                    (Dataset.display_name.like(f"%{folder}%"))
                )
                .limit(30)
                .all()
            )
        result = []
        for ds in datasets:
            name = ds.display_name or (ds.physical_name or "").split("/")[-1] or f"ds_{ds.id}"
            result.append({"id": ds.id, "name": name, "physical_name": ds.physical_name or ""})
        return result

    def folder_dataset_details(self, folder: str) -> List[Dict]:
        """
        Returns full details for all datasets in a folder including health, issues, rows.
        Used for the sidebar dataset list and folder greeting.
        """
        base_datasets = self.resolve_folder_datasets(folder)
        result = []
        for ds_info in base_datasets:
            ds_id = ds_info["id"]
            run = (
                self.db.query(ProfilingRun)
                .filter(ProfilingRun.dataset_id == ds_id, ProfilingRun.status == "COMPLETED")
                .order_by(desc(ProfilingRun.id)).first()
            )
            if run:
                cols = self.db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
                health = round(sum(c.health_score for c in cols) / len(cols), 1) if cols else None
                issues = sum(1 for c in cols if c.status in ("WARNING", "CRITICAL"))
                result.append({
                    "id":             ds_id,
                    "name":           ds_info["name"],
                    "physical_name":  ds_info["physical_name"],
                    "health":         health,
                    "issues":         issues,
                    "rows":           run.rows_processed,
                    "total_columns":  len(cols),
                    "active_rules":   self.db.query(DQRule).filter(
                        DQRule.dataset_id == ds_id,
                        DQRule.status == "Active"
                    ).count(),
                    "has_data":       True,
                })
            else:
                result.append({
                    "id":            ds_id,
                    "name":          ds_info["name"],
                    "physical_name": ds_info["physical_name"],
                    "health":        None,
                    "issues":        None,
                    "rows":          None,
                    "total_columns": None,
                    "active_rules":  0,
                    "has_data":      False,
                })
        return result

    def folder_summary(self, folder: str) -> Dict:
        """Aggregate summary across all datasets in a folder."""
        datasets = self.folder_dataset_details(folder)
        if not datasets:
            return {"status": "NO_DATA", "folder": folder, "dataset_count": 0}

        with_data = [d for d in datasets if d["has_data"] and d["health"] is not None]
        if not with_data:
            return {
                "status": "NO_PROFILING",
                "folder": folder,
                "dataset_count": len(datasets),
                "datasets": datasets,
            }

        avg_health = round(sum(d["health"] for d in with_data) / len(with_data), 1)
        total_issues = sum(d["issues"] or 0 for d in with_data)
        worst = min(with_data, key=lambda d: d["health"])
        best  = max(with_data, key=lambda d: d["health"])
        return {
            "status":          "OK",
            "folder":          folder,
            "dataset_count":   len(datasets),
            "profiled_count":  len(with_data),
            "avg_health":      avg_health,
            "total_issues":    total_issues,
            "worst_dataset":   {"name": worst["name"], "health": worst["health"]},
            "best_dataset":    {"name": best["name"],  "health": best["health"]},
            "datasets":        datasets,
        }

    # ── Per-dataset helpers (unchanged) ───────────────────────────────────────

    def dataset_info(self, dataset_id: int) -> dict:
        ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            return {"id": dataset_id, "name": f"dataset_{dataset_id}"}
        return {
            "id":   ds.id,
            "name": ds.display_name or (ds.physical_name or "").split("/")[-1] or f"ds_{ds.id}",
            "physical_name": ds.physical_name,
        }

    def all_datasets(self) -> list:
        result = []
        for ds in self.db.query(Dataset).limit(30).all():
            run = (
                self.db.query(ProfilingRun)
                .filter(ProfilingRun.dataset_id == ds.id, ProfilingRun.status == "COMPLETED")
                .order_by(desc(ProfilingRun.id)).first()
            )
            cols = self.db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all() if run else []
            health = round(sum(c.health_score for c in cols) / len(cols), 1) if cols else None
            result.append({
                "id":     ds.id,
                "name":   ds.display_name or (ds.physical_name or "").split("/")[-1] or f"ds_{ds.id}",
                "health": health,
                "rows":   run.rows_processed if run else None,
                "issues": sum(1 for c in cols if c.status in ("WARNING", "CRITICAL")),
            })
        return result

    def dq_summary(self, dataset_id: int) -> dict:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return {"status": "NO_DATA"}
        cols = self.db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        avg_c = round(sum(c.completeness for c in cols) / len(cols), 1) if cols else 0
        avg_h = round(sum(c.health_score for c in cols) / len(cols), 1) if cols else 0
        return {
            "run_id": run.id, "total_rows": run.rows_processed,
            "total_columns": len(cols), "avg_completeness": avg_c,
            "avg_health": avg_h,
            "issues": sum(1 for c in cols if c.status in ("WARNING", "CRITICAL")),
            "last_run": run.timestamp.isoformat() if run.timestamp else None,
        }

    def column_profiles(self, dataset_id: int) -> list:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return []
        return [
            {
                "column": c.column_name, "type": c.data_type,
                "completeness": c.completeness, "uniqueness": c.uniqueness,
                "null_count": c.null_count, "distinct_count": c.distinct_count,
                "health_score": c.health_score, "status": c.status,
            }
            for c in self.db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        ]

    def quality_checks(self, dataset_id: int, open_only: bool = False) -> list:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return []
        q = self.db.query(QualityCheck).filter(QualityCheck.profiling_run_id == run.id)
        if open_only:
            q = q.filter(QualityCheck.status.notin_(["resolved"]))
        return [
            {
                "id": c.id, "check_type": c.check_type, "column": c.column_name,
                "severity": c.severity, "violation_count": c.violation_count,
                "description": c.description,
                "status": (getattr(c, "status", "open") or "open").lower(),
            }
            for c in q.all()
        ]

    def active_rules(self, dataset_id: int) -> list:
        from sqlalchemy import func as _func
        return [
            {
                "rule_code": r.rule_code, "rule_name": r.name,
                "column": r.column, "condition": r.condition,
                "severity": r.severity, "type": getattr(r, "type", None),
            }
            for r in self.db.query(DQRule).filter(
                DQRule.dataset_id == dataset_id,
                _func.lower(DQRule.status) == "active",
            ).all()
        ]

    def drift_summary(self, dataset_id: int) -> list:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return []
        return [
            {"column": r.column_name, "drift_score": round(r.drift_score or 0, 1), "drift_type": r.drift_type}
            for r in self.db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run.id).all()
            if (r.drift_score or 0) > 10
        ]

    def health_trend(self, dataset_id: int) -> list:
        runs = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(ProfilingRun.id.asc()).limit(15).all()
        )
        result = []
        for r in runs:
            cols = self.db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == r.id).all()
            health = round(sum(c.health_score for c in cols) / len(cols), 1) if cols else 0
            issues = sum(1 for c in cols if c.status in ("WARNING", "CRITICAL"))
            result.append({"run_id": r.id, "health": health, "issues": issues,
                           "timestamp": r.timestamp.isoformat() if r.timestamp else None, "label": f"#{r.id}"})
        return result

    def schema_changes(self, dataset_id: int) -> list:
        return [
            {"change_type": c.change_type, "column": c.column_name,
             "old_type": c.old_type, "new_type": c.new_type, "impact": c.impact}
            for c in self.db.query(SchemaHistory)
            .filter(SchemaHistory.dataset_id == dataset_id)
            .order_by(desc(SchemaHistory.id)).limit(10).all()
        ]

    def run_history(self, dataset_id: int, limit: int = 10) -> list:
        return [
            {"run_id": r.id, "status": r.status, "rows": r.rows_processed,
             "timestamp": r.timestamp.isoformat() if r.timestamp else None}
            for r in self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id)
            .order_by(desc(ProfilingRun.id)).limit(limit).all()
        ]

    def fetch_scorecards(self, dataset_id: int) -> dict:
        result: dict = {}
        try:
            from app.services import scorecards as svc
            try:
                full = svc.get_full_scorecard(self.db, dataset_id)
                if isinstance(full, dict):
                    return full
            except Exception:
                pass
            individual_calls = [
                ("kpi",             svc.get_kpi_summary,       {}),
                ("velocity",        svc.get_quality_velocity,   {}),
                ("rulesCoverage",   svc.get_rules_coverage,     {}),
                ("freshness",       svc.get_freshness_score,    {}),
                ("riskContributors",svc.get_risk_contributors,  {"top_n": 5}),
            ]
            for attr, key in [
                ("get_violation_heatmap",  "violationHeatmap"),
                ("get_incident_timeline",  "incidentTimeline"),
                ("get_column_risk_table",  "columnRiskTable"),
                ("get_schema_stability",   "schemaStability"),
                ("get_drift_kpi",          "driftKpi"),
                ("get_run_comparison",     "runComparison"),
            ]:
                fn_opt = getattr(svc, attr, None)
                if fn_opt:
                    individual_calls.append((key, fn_opt, {}))
            for key, fn, extra in individual_calls:
                try:
                    val = fn(self.db, dataset_id, **extra)
                    result[key] = val if isinstance(val, (dict, list)) else {}
                except Exception:
                    result[key] = {}
        except Exception as e:
            return {"error": str(e)}
        return result

    def fetch_monitoring(self, dataset_id: int) -> dict:
        result: dict = {}
        try:
            from app.services import monitoring as svc
            calls = [
                ("summary",        svc.get_monitoring_summary,  {}),
                ("column_health",  svc.get_column_health,       {"limit": 10}),
                ("sla_status",     svc.get_sla_status,          {"limit": 10}),
                ("risk_forecast",  svc.get_risk_forecast,       {}),
            ]
            for attr in ["get_metrics_trends", "get_drift_monitoring"]:
                fn = getattr(svc, attr, None)
                if fn:
                    key = attr.replace("get_", "").replace("_monitoring", "").replace("_trends", "_trend")
                    calls.append((key, fn, {"limit": 10}))
            for key, fn, extra in calls:
                try:
                    val = fn(self.db, dataset_id, **extra)
                    result[key] = val if isinstance(val, (dict, list)) else {}
                except Exception as _e:
                    result[key] = {"error": str(_e)[:80]}
        except Exception as e:
            return {"error": str(e)}
        return result

    def fetch_anomalies(self, dataset_id: int) -> dict:
        try:
            from app.services import anomalies as svc
            return svc.get_anomalies(self.db, dataset_id)
        except Exception as e:
            return {"error": str(e)}

    def fetch_alerts(self, dataset_id: int) -> list:
        try:
            from app.services.alerts import get_alerts
            return get_alerts(self.db, dataset_id)
        except Exception:
            return []

    def fetch_lineage(self, dataset_id: int) -> dict:
        try:
            from app.services import lineage as svc
            ds = self.dataset_info(dataset_id)
            name = ds.get("physical_name") or ds.get("name", "")
            return svc.get_graph(dataset=name)
        except Exception as e:
            return {"error": str(e), "nodes": [], "edges": []}

    def what_if_rule(self, dataset_id: int, column_name: str, condition_sql: str) -> dict:
        cols = self.column_profiles(dataset_id)
        col  = next((c for c in cols if c["column"].lower() == column_name.lower()), None)
        if not col:
            return {"column": column_name, "simulated": False, "message": "Column not found"}
        if "IS NOT NULL" in condition_sql.upper() or "NOT NULL" in condition_sql.upper():
            return {
                "column": column_name, "simulated": True,
                "would_fail": col["null_count"],
                "fail_pct": round((1 - col["completeness"] / 100) * 100, 1),
                "message": f"{col['null_count']:,} rows would fail NOT NULL on `{column_name}` ({100-col['completeness']:.1f}% of rows)"
            }
        return {
            "column": column_name, "simulated": True, "would_fail": None, "fail_pct": None,
            "message": f"Column `{column_name}`: {col['null_count']} nulls, {col['completeness']}% complete. Cannot simulate this condition without running SQL."
        }


# ─── Action Executor ──────────────────────────────────────────────────────────

class ActionExecutor:
    def __init__(self, db: Session):
        self.db = db

    def create_dq_rule(self, dataset_id: int, rule_name: str, column_name: str,
                       condition_sql: str, severity: str = "Medium", rule_type: str = "Validity") -> dict:
        n         = self.db.query(DQRule).filter(DQRule.dataset_id == dataset_id).count()
        rule_code = f"AGENT_{dataset_id}_{n+1:03d}"
        rule = DQRule(
            dataset_id=dataset_id, rule_code=rule_code,
            name=rule_name, column=column_name, condition=condition_sql,
            type=rule_type, severity=severity.capitalize(), status="Active", input_mode="dsl",
        )
        self.db.add(rule)
        self.db.commit()
        return {"rule_code": rule_code, "rule_name": rule_name, "column_name": column_name}

    def delete_dq_rule(self, dataset_id: int, rule_code: str) -> dict:
        rule = self.db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id, DQRule.rule_code == rule_code,
        ).first()
        if not rule:
            raise ValueError(f"Rule {rule_code} not found")
        rule.status = "Inactive"
        self.db.commit()
        return {"deleted": rule_code, "rule_name": rule.name}

    def update_dq_rule(self, dataset_id: int, rule_code: str, updates: dict) -> dict:
        rule = self.db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id, DQRule.rule_code == rule_code,
        ).first()
        if not rule:
            raise ValueError(f"Rule {rule_code} not found")
        if "severity"  in updates: rule.severity  = updates["severity"]
        if "condition" in updates: rule.condition  = updates["condition"]
        if "status"    in updates: rule.status     = updates["status"]
        self.db.commit()
        return {"updated": rule_code, "rule_name": rule.name}

    def dismiss_alerts(self, dataset_id: int, severity: str = None, check_type: str = None) -> dict:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return {"dismissed": 0}
        q = self.db.query(QualityCheck).filter(QualityCheck.profiling_run_id == run.id)
        if severity:
            q = q.filter(QualityCheck.severity == severity.upper())
        if check_type:
            q = q.filter(QualityCheck.check_type == check_type)
        count = 0
        for chk in q.all():
            chk.status = "resolved"
            count += 1
        self.db.commit()
        return {"dismissed": count}

    def fix_anomaly(self, dataset_id: int, check_id: int) -> dict:
        try:
            from app.services.anomalies import fix_anomaly
            return fix_anomaly(self.db, dataset_id, check_id)
        except Exception as e:
            return {"error": str(e)}

    def resolve_checks_by_severity(self, dataset_id: int, severity: str) -> dict:
        from sqlalchemy import func as _func
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return {"resolved": 0, "severity": severity}
        sev_upper = severity.strip().upper()
        checks = (
            self.db.query(QualityCheck)
            .filter(
                QualityCheck.profiling_run_id == run.id,
                _func.upper(QualityCheck.severity) == sev_upper,
                QualityCheck.status != "resolved",
            )
            .all()
        )
        for c in checks:
            c.status = "resolved"
        self.db.commit()
        return {"resolved": len(checks), "severity": sev_upper,
                "check_types": list({c.check_type for c in checks})}

    def fix_all_anomalies(self, dataset_id: int) -> dict:
        checks = self.db.query(QualityCheck).join(ProfilingRun).filter(
            ProfilingRun.dataset_id == dataset_id,
            ProfilingRun.status == "COMPLETED",
        ).order_by(desc(ProfilingRun.id)).limit(50).all()
        fixed = 0
        for chk in checks:
            try:
                from app.services.anomalies import fix_anomaly
                fix_anomaly(self.db, dataset_id, chk.id)
                fixed += 1
            except Exception:
                pass
        return {"fixed": fixed}

    def trigger_profiling(self, dataset_id: int) -> dict:
        from app.services.dq_scores import run_dq_scoring
        run = run_dq_scoring(self.db, dataset_id)
        return {"run_id": run.id, "status": run.status, "rows": run.rows_processed}

    def set_baseline(self, dataset_id: int) -> dict:
        from app.services.dq_scores import set_baseline as svc_bl, get_baseline_candidates
        cands = get_baseline_candidates(self.db, dataset_id, limit=1)
        if not cands:
            return {"error": "No completed runs to set as baseline."}
        run_id  = cands[0]["runId"]
        success = svc_bl(self.db, dataset_id, run_id)
        return {"baseline_set": success, "run_id": run_id}

    def resolve_checks(self, dataset_id: int, check_type: str = None, column_name: str = None) -> dict:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return {"resolved": 0}
        q = self.db.query(QualityCheck).filter(QualityCheck.profiling_run_id == run.id)
        if check_type:   q = q.filter(QualityCheck.check_type == check_type)
        if column_name:  q = q.filter(QualityCheck.column_name == column_name)
        count = 0
        for chk in q.all():
            chk.status = "resolved"
            count += 1
        self.db.commit()
        return {"resolved": count}

    def resolve_single_check(self, dataset_id: int, check_id: int = None,
                              check_type: str = None, column_name: str = None,
                              severity: str = None) -> dict:
        run = (
            self.db.query(ProfilingRun)
            .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
            .order_by(desc(ProfilingRun.id)).first()
        )
        if not run:
            return {"resolved": 0, "message": "No profiling run found."}
        if check_id and check_id > 0:
            chk = self.db.query(QualityCheck).filter(
                QualityCheck.id == check_id,
                QualityCheck.profiling_run_id == run.id,
            ).first()
            if chk:
                chk.status = "resolved"
                self.db.commit()
                return {"resolved": 1, "check_id": check_id,
                        "check_type": chk.check_type, "column": chk.column_name}
        q = self.db.query(QualityCheck).filter(
            QualityCheck.profiling_run_id == run.id,
            QualityCheck.status != "resolved",
        )
        if check_type:  q = q.filter(QualityCheck.check_type.ilike(f"%{check_type}%"))
        if column_name: q = q.filter(QualityCheck.column_name.ilike(f"%{column_name}%"))
        if severity:    q = q.filter(QualityCheck.severity.ilike(severity.upper()))
        matches = q.all()
        if not matches:
            return {"resolved": 0, "message": "No matching open check found."}
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        matches.sort(key=lambda c: sev_order.get((c.severity or "").upper(), 9))
        chk = matches[0]
        chk.status = "resolved"
        self.db.commit()
        return {"resolved": 1, "check_id": chk.id,
                "check_type": chk.check_type, "column": chk.column_name,
                "severity": chk.severity}

    def pause_all_rules(self, dataset_id: int) -> dict:
        from sqlalchemy import func as _func
        rules = self.db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id,
            _func.lower(DQRule.status) == "active",
        ).all()
        for r in rules:
            r.status = "Paused"
        self.db.commit()
        return {"paused": len(rules), "rule_codes": [r.rule_code for r in rules]}

    def resume_all_rules(self, dataset_id: int) -> dict:
        from sqlalchemy import func as _func
        rules = self.db.query(DQRule).filter(
            DQRule.dataset_id == dataset_id,
            _func.lower(DQRule.status) == "paused",
        ).all()
        for r in rules:
            r.status = "Active"
        self.db.commit()
        return {"resumed": len(rules)}

    def schedule_profiling(self, dataset_id: int, times: list, on_new_data: bool) -> dict:
        try:
            from app.routers.profiling_detail import _schedule_store
            _schedule_store[dataset_id] = {"times": sorted(set(times)), "on_new_data": on_new_data}
            return {"scheduled": True, "times": sorted(set(times)), "on_new_data": on_new_data}
        except Exception as e:
            return {"error": str(e)}


# ─── The 6 Agents ─────────────────────────────────────────────────────────────

class MasterAgent:
    SYSTEM = """You are the Master Routing Agent for an AI Data Quality Copilot.

Your ONLY job: read the user message and decide which specialist agent handles it.

Routes:
- "query"    → Wants information: stats, metrics, scorecards, monitoring, lineage, column details, comparisons, what-if, KG info, trends
- "diagnose" → Wants root cause analysis, investigation of why something happened
- "action"   → Wants to DO something: create/edit/delete rules, dismiss alerts, fix anomalies, trigger profiling, set baseline, schedule, create KG, resolve checks
- "report"   → Wants a full structured report or summary document
- "chitchat" → Greetings, capability questions, help

Return ONLY valid JSON:
{
  "route": "query|diagnose|action|report|chitchat",
  "user_ack": "One sentence shown to user while agent works",
  "reasoning": "Why this route"
}"""

    def route(self, message: str, context_summary: str) -> dict:
        result = _llm_json(self.SYSTEM, f"Context: {context_summary}\nMessage: {message}", max_tokens=200)
        if not result or "route" not in result:
            return {"route": "query", "user_ack": "Looking that up…", "reasoning": "fallback"}
        return result


class QueryAgent:
    SYSTEM = """You are the Query Agent for an AI DQ Assistant.
You answer questions about ALL tabs in the application.

CRITICAL ANTI-HALLUCINATION RULES:
- Use ONLY numbers and names explicitly present in the data context below.
- If a metric is missing from context, say "not available in current data" — never invent a number.
- If asked about a dataset not in the context, say it was not found in this folder.
- For folder-mode responses: always attribute each metric to its specific dataset by name.
- NEVER say a dataset has X% health unless that exact number appears in the context.

FORMAT:
- Bullet lists with • prefix. Use **bold** for key numbers.
- For folder queries: lead with the folder summary, then per-dataset breakdown.
- For anomalies: ALWAYS separate open vs resolved. State exact counts.
- Keep responses under 500 words unless a multi-dataset comparison is requested."""

    def answer(self, message: str, context: str, history: List[Dict]) -> str:
        return _llm(self.SYSTEM, f"Data:\n{context}\n\nQuestion: {message}", history, max_tokens=900)


class DiagnoseAgent:
    SYSTEM = """You are the Diagnosis Agent for an AI Data Quality Copilot.
You perform root cause analysis on data quality issues.

CRITICAL: Base your entire diagnosis on the data context provided.
Do NOT speculate about issues not evidenced in the context.
If context is sparse, say so and recommend running profiling first.

Always structure response:
## Observed Problem
## Most Likely Root Cause
## Supporting Evidence
## Recommended Fix

Be specific. Cite actual numbers. Keep each section 2-4 sentences."""

    def diagnose(self, message: str, context: str, history: List[Dict]) -> str:
        return _llm(self.SYSTEM, f"Data:\n{context}\n\nQuestion: {message}", history, max_tokens=800)


class ActionAgent:
    SYSTEM = """You are the Action Planning Agent for an AI Data Quality Copilot.
You plan concrete data quality actions based on user requests.

Available actions:
- create_rule: Create DQ validation rule (needs rule_name, column_name, condition_sql, severity, rule_type)
- delete_rule: Deactivate a specific rule (needs rule_code from context — NEVER invent codes)
- update_rule: Modify rule properties (needs rule_code from context, updates dict)
- pause_all_rules: Pause ALL active rules in one action
- resume_all_rules: Re-activate all paused rules
- dismiss_alerts: Dismiss quality check alerts (optional: severity, check_type)
- fix_anomaly: Fix a specific anomaly (needs check_id)
- resolve_single_check: Resolve ONE specific anomaly. Params: check_id (int), check_type, column_name, severity.
- fix_all_anomalies: Fix all anomalies for this dataset
- trigger_profiling: Re-run DQ scoring
- set_baseline: Set current run as quality baseline
- resolve_checks: Mark ALL quality checks as resolved
- resolve_checks_by_severity: Resolve ONLY checks of a specific severity
- schedule_profiling: Set auto-profiling schedule (needs times list, on_new_data bool)
- redirect_lineage: Navigate user to Lineage tab
- redirect_kg: Navigate to Knowledge Graph tab
- redirect_monitoring: Navigate to Monitoring tab
- redirect_anomalies: Navigate to Anomalies tab
- redirect_scorecards: Navigate to Scorecards tab
- redirect_dq_rules: Navigate to DQ Rules tab

CRITICAL:
- For rule_code in delete/update: ONLY use codes from context. NEVER invent codes.
- In folder mode: the dataset_id in params will be the specific dataset to act on.
- Return ONLY valid JSON:

{
  "response": "1-2 sentence friendly explanation",
  "actions": [
    {
      "type": "action_type",
      "label": "Short label",
      "description": "One sentence",
      "params": {
        "rule_name": "", "column_name": "", "condition_sql": "",
        "severity": "High|Medium|Low", "rule_type": "Validity|Completeness|Uniqueness|Consistency",
        "rule_code": "", "check_id": 0, "check_type": "",
        "times": [], "on_new_data": false, "updates": {}
      }
    }
  ]
}"""

    def plan(self, message: str, context: str, dataset_id: int) -> dict:
        result = _llm_json(
            self.SYSTEM,
            f"Dataset ID: {dataset_id}\nData context:\n{context}\n\nUser request: {message}",
            max_tokens=800,
        )
        if not result:
            return {
                "response": "I'll trigger a fresh profiling run to get updated metrics.",
                "actions": [{"type": "trigger_profiling", "label": "Run DQ Scoring",
                             "description": "Re-profile this dataset.", "params": {}}]
            }
        return result


class ReportAgent:
    SYSTEM = """You are the Report Generation Agent for an AI Data Quality Copilot.
Generate comprehensive data quality reports.

CRITICAL: Use ONLY data present in the context. Do not fabricate any metric.
If a section has no data, state "No data available for this section."

Structure:
## Executive Summary
## Key Metrics
## Issues Found
## Drift & Schema Changes
## Scorecard Overview
## Active Rules
## Monitoring Status
## Recommendations

Rules:
- Use • for bullets. **bold** for key values.
- Each section: 3-6 sentences or bullets.
- End with 5 ranked actionable recommendations."""

    def generate(self, message: str, context: str, history: List[Dict]) -> str:
        return _llm(self.SYSTEM, f"Data:\n{context}\n\nRequest: {message}", history, max_tokens=1400)


class SuggestionAgent:
    SYSTEM = """You are the Suggestion Agent. Generate context-aware follow-ups.

Return ONLY valid JSON:
{
  "followups": ["Q1?", "Q2?", "Q3?"],
  "actions": [
    {"label": "Label", "type": "type", "description": "One sentence"}
  ]
}

Action types: create_rule, delete_rule, trigger_profiling, set_baseline, resolve_checks,
resolve_single_check, resolve_checks_by_severity, dismiss_alerts, fix_all_anomalies,
generate_report, compare_datasets, what_if_rule, schedule_profiling, redirect_lineage,
redirect_kg, redirect_monitoring, redirect_scorecards, redirect_anomalies, redirect_dq_rules,
query_scorecards, query_monitoring

Generate 3 specific follow-up questions relevant to what was just discussed.
Generate 3-4 directly applicable next actions."""

    def suggest(self, message: str, response: str, context_summary: str) -> dict:
        result = _llm_json(
            self.SYSTEM,
            f"Context: {context_summary}\nUser asked: {message}\nAgent said: {response[:300]}",
            max_tokens=400,
        )
        if not result:
            return {
                "followups": ["What columns have the most issues?", "Show me the scorecard.", "How has health trended?"],
                "actions": [
                    {"label": "Run DQ Scoring",       "type": "trigger_profiling",  "description": "Refresh quality metrics."},
                    {"label": "Generate report",      "type": "generate_report",    "description": "Full structured report."},
                    {"label": "View Monitoring",      "type": "redirect_monitoring","description": "Open monitoring dashboard."},
                ]
            }
        return result


# ─── Tool Selector ────────────────────────────────────────────────────────────

def _select_tools(message: str, route: str) -> List[str]:
    msg = message.lower()
    tools = ["info", "summary", "rules"]

    if route in ("diagnose", "report"):
        return ["info", "summary", "columns", "checks", "drift", "schema", "history", "trend", "rules", "anomalies"]

    if route == "action":
        tools.extend(["checks", "columns", "rules", "alerts"])

    if any(w in msg for w in ["column", "null", "complet", "unique", "pattern", "profile"]):
        tools.extend(["columns", "checks"])
    if any(w in msg for w in ["quality", "check", "violat", "issue", "problem", "flag",
                               "outlier", "duplic", "anomal", "open", "resolve",
                               "severity", "critical", "high", "medium", "low"]):
        tools.extend(["checks", "anomalies"])
    if any(w in msg for w in ["drift", "distribut", "shift"]):
        tools.extend(["drift"])
    if any(w in msg for w in ["rule", "policy", "validat", "constraint"]):
        tools.append("rules")
    if any(w in msg for w in ["schema", "added column", "removed column", "type change"]):
        tools.append("schema")
    if any(w in msg for w in ["trend", "history", "over time", "previous", "baseline", "progress",
                               "improved", "dropped", "compared to", "last run", "worse", "better",
                               "when was", "last profil", "last scored", "timestamp",
                               "how many rows", "rows processed"]):
        tools.extend(["history", "trend"])
    if any(w in msg for w in ["alert", "dismiss", "incident"]):
        tools.extend(["checks", "alerts"])
    if any(w in msg for w in ["scorecard", "kpi", "velocity", "coverage", "heatmap", "freshness", "risk table", "incident timeline"]):
        tools.append("scorecards")
    if any(w in msg for w in ["monitor", "sla", "column health", "forecast", "execution"]):
        tools.append("monitoring")
    if any(w in msg for w in ["lineage", "upstream", "downstream", "pipeline", "source"]):
        tools.append("lineage")
    if any(w in msg for w in ["all dataset", "compare", "overall", "every dataset", "across"]):
        tools.append("all_datasets")
    if any(w in msg for w in ["knowledge graph", "kg", "relationship", "entity", "connected"]):
        tools.append("lineage")

    return list(set(tools))


# ─── Context Builder: single dataset ─────────────────────────────────────────

def _build_context(data_tools: DataTools, dataset_id: Optional[int], tools: List[str]) -> tuple:
    """Returns (context_string, raw_dict, chart_data, redirect)."""
    raw:   Dict[str, Any] = {}
    chart: Optional[dict] = None
    redir: Optional[dict] = None

    if dataset_id:
        if "info"        in tools: raw["info"]       = data_tools.dataset_info(dataset_id)
        if "summary"     in tools: raw["summary"]    = data_tools.dq_summary(dataset_id)
        if "columns"     in tools: raw["columns"]    = data_tools.column_profiles(dataset_id)
        if "checks"      in tools: raw["checks"]     = data_tools.quality_checks(dataset_id, open_only=True)
        if "drift"       in tools: raw["drift"]      = data_tools.drift_summary(dataset_id)
        if "rules"       in tools: raw["rules"]      = data_tools.active_rules(dataset_id)
        if "history"     in tools: raw["history"]    = data_tools.run_history(dataset_id)
        if "schema"      in tools: raw["schema"]     = data_tools.schema_changes(dataset_id)
        if "alerts"      in tools: raw["alerts"]     = data_tools.fetch_alerts(dataset_id)
        if "anomalies"   in tools: raw["anomalies"]  = data_tools.fetch_anomalies(dataset_id)
        if "scorecards"  in tools: raw["scorecards"] = data_tools.fetch_scorecards(dataset_id)
        if "monitoring"  in tools: raw["monitoring"] = data_tools.fetch_monitoring(dataset_id)
        if "lineage"     in tools: raw["lineage"]    = data_tools.fetch_lineage(dataset_id)
        if "trend"       in tools:
            raw["trend"] = data_tools.health_trend(dataset_id)
            if len(raw.get("trend", [])) >= 2:
                chart = {"type": "health_trend", "data": raw["trend"], "label": "Health Score Trend"}

    if "all_datasets" in tools:
        raw["all_datasets"] = data_tools.all_datasets()
        if raw["all_datasets"]:
            chart = {"type": "dataset_comparison",
                     "data": [d for d in raw["all_datasets"] if d.get("health")],
                     "label": "Dataset Health Comparison"}

    if "scorecards" in tools and isinstance(raw.get("scorecards"), dict):
        sc = raw["scorecards"]
        kpi = sc.get("kpi") or sc.get("kpiSummary") or {}
        if isinstance(kpi, dict):
            kpi_vals = []
            for label, keys in [
                ("Health",       ["overallHealthScore","overall_health"]),
                ("Completeness", ["completenessScore","completeness"]),
                ("Uniqueness",   ["uniquenessScore","uniqueness"]),
                ("Validity",     ["validityScore","validity"]),
            ]:
                for k in keys:
                    v = kpi.get(k)
                    if v is not None:
                        kpi_vals.append({"metric": label, "value": round(float(v), 1)})
                        break
            if len(kpi_vals) >= 2 and not chart:
                chart = {"type": "scorecard_kpi", "data": kpi_vals, "label": "Scorecard KPI Breakdown"}

    if chart is None and "columns" in tools:
        cols = raw.get("columns", [])
        if len(cols) >= 3:
            chart_cols = sorted(cols, key=lambda c: c.get("health_score", 100))[:10]
            chart = {
                "type": "column_health",
                "data": [{"name": c["column"], "health": c.get("health_score",0), "completeness": c.get("completeness",0)} for c in chart_cols],
                "label": "Column Health Scores",
            }

    lines = []
    info = raw.get("info", {})
    if info:
        lines.append(f"Dataset: {info.get('name')}")

    s = raw.get("summary", {})
    if s and s.get("status") != "NO_DATA":
        _ts_raw = s.get("last_run") or ""
        try:
            from datetime import timezone as _tz, timedelta as _td
            _IST = _td(hours=5, minutes=30)
            _dt_parsed = datetime.fromisoformat(_ts_raw.replace("Z", "+00:00")) if _ts_raw else None
            if _dt_parsed:
                if _dt_parsed.tzinfo is None:
                    _dt_parsed = _dt_parsed.replace(tzinfo=_tz.utc)
                _ist_time = _dt_parsed + _IST
                _ts_fmt = _ist_time.strftime("%d %b %Y %H:%M IST")
            else:
                _ts_fmt = "not available"
        except Exception:
            _ts_fmt = _ts_raw[:16] if _ts_raw else "not available"
        lines.append(
            f"DQ Summary: health={s.get('avg_health')}%, completeness={s.get('avg_completeness')}%, "
            f"issues={s.get('issues')}, rows={s.get('total_rows', 0):,}, run #{s.get('run_id')}, "
            f"last_profiled_at={_ts_fmt}"
        )

    checks = raw.get("checks", [])
    if checks:
        open_chks = [c for c in checks if c.get("status", "open").lower() in ("open", "investigating")]
        crit      = [c for c in open_chks if (c["severity"] or "").upper() in ("CRITICAL", "HIGH")]
        med       = [c for c in open_chks if (c["severity"] or "").upper() == "MEDIUM"]
        low       = [c for c in open_chks if (c["severity"] or "").upper() in ("LOW", "INFO")]
        top       = [c["check_type"] + " on " + str(c["column"]) + " [" + (c["severity"] or "?") + "]" for c in open_chks[:6]]
        lines.append(
            f"Open quality checks: {len(open_chks)} open "
            f"(critical/high={len(crit)}, medium={len(med)}, low={len(low)}). "
            f"Top open: {', '.join(top)}"
        )

    cols = raw.get("columns", [])
    if cols:
        bad = [c for c in cols if c["status"] in ("WARNING", "CRITICAL")]
        if bad:
            lines.append("Problem columns: " + ", ".join(
                f"{c['column']} ({c['completeness']}% complete, {c['null_count']} nulls)" for c in bad[:6]
            ))
        all_cols = "; ".join(
            f"{c['column']}: {c['type']}, {c['completeness']}% complete, {c['uniqueness']}% unique, health={c['health_score']}%"
            for c in cols[:15]
        )
        lines.append(f"All columns: {all_cols}")

    drift = raw.get("drift", [])
    if drift:
        lines.append("Drift: " + ", ".join(f"{d['column']} ({d['drift_score']}%)" for d in drift[:5]))

    schema = raw.get("schema", [])
    if schema:
        lines.append("Schema changes: " + ", ".join(f"{c['change_type']} on {c['column']}" for c in schema[:4]))

    hist = raw.get("history", [])
    if hist and len(hist) >= 1:
        from datetime import timedelta as _tdh, timezone as _tzh
        _IST_H = _tdh(hours=5, minutes=30)
        run_lines = []
        for _r in hist[:3]:
            _rts = _r.get("timestamp", "")
            try:
                _rdt = datetime.fromisoformat(_rts.replace("Z", "+00:00")) if _rts else None
                if _rdt:
                    if _rdt.tzinfo is None: _rdt = _rdt.replace(tzinfo=_tzh.utc)
                    _rts_fmt = (_rdt + _IST_H).strftime("%d %b %Y %H:%M IST")
                else:
                    _rts_fmt = "unknown time"
            except Exception:
                _rts_fmt = (_rts[:16] if _rts else "unknown")
            run_lines.append(f"#{_r.get('run_id','?')} at {_rts_fmt} ({_r.get('rows', 0):,} rows)")
        lines.append("Profiling run history: " + " → ".join(run_lines))

    trend = raw.get("trend", [])
    if len(trend) >= 2:
        lines.append(f"Health trend: {trend[0]['health']}% → {trend[-1]['health']}% over {len(trend)} runs")

    rules = raw.get("rules", [])
    lines.append(f"Active DQ rules: {len(rules)}" + (": " + ", ".join(r["rule_name"] for r in rules[:4]) if rules else ""))

    alerts = raw.get("alerts", [])
    if alerts:
        lines.append(f"Active alerts: {len(alerts)}")

    anomalies = raw.get("anomalies", {})
    if anomalies and anomalies.get("status") != "NO_DATA" and isinstance(anomalies, dict):
        anom_list  = anomalies.get("anomalies", [])
        if anom_list:
            open_anoms     = [a for a in anom_list if (a.get("status") or "open").lower() in ("open", "investigating")]
            resolved_anoms = [a for a in anom_list if (a.get("status") or "open").lower() == "resolved"]
            sev_breakdown  = {}
            for a in open_anoms:
                sev = (a.get("severity") or "unknown").upper()
                sev_breakdown[sev] = sev_breakdown.get(sev, 0) + 1
            sev_str = ", ".join(f"{k}={v}" for k, v in sorted(sev_breakdown.items()))
            top_open = [a.get("check_type", "") + " on " + str(a.get("column", "?")) for a in open_anoms[:5]]
            lines.append(
                f"Anomalies: {len(anom_list)} total — {len(open_anoms)} OPEN, {len(resolved_anoms)} resolved. "
                f"Open severity: {sev_str or 'none'}. "
                f"Top open: {', '.join(top_open)}"
            )

    # Abbreviated scorecard/monitoring/lineage sections (same as original)
    sc = raw.get("scorecards", {})
    if isinstance(sc, dict) and not sc.get("error"):
        kpi = sc.get("kpi") or sc.get("kpiSummary") or {}
        if isinstance(kpi, dict) and kpi:
            oh  = kpi.get("overallHealthScore") or kpi.get("overall_health")
            com = kpi.get("completenessScore")  or kpi.get("completeness")
            uni = kpi.get("uniquenessScore")    or kpi.get("uniqueness")
            val = kpi.get("validityScore")       or kpi.get("validity")
            parts = [f"overall_health={oh}%" if oh is not None else None,
                     f"completeness={com}%"  if com is not None else None,
                     f"uniqueness={uni}%"    if uni is not None else None,
                     f"validity={val}%"      if val is not None else None]
            lines.append("Scorecard KPI: " + ", ".join(p for p in parts if p))

    lin = raw.get("lineage", {})
    if lin and not lin.get("error"):
        nodes = lin.get("nodes", [])
        edges = lin.get("edges", [])
        if nodes:
            lines.append(f"Lineage: {len(nodes)} nodes, {len(edges)} relationships")

    datasets = raw.get("all_datasets", [])
    if datasets:
        d_strs = [f"{d['name']} (health: {d['health']}%, issues: {d.get('issues', 0)})" for d in datasets if d.get("health")]
        lines.append(f"All datasets ({len(datasets)}): {', '.join(d_strs)}")

    context = "\n".join(lines) or "No data available. Run DQ Scoring first."
    return context, raw, chart, redir


# ─── Context Builder: folder (aggregated across all datasets) ─────────────────

def _build_folder_context(data_tools: DataTools, folder: str, message: str) -> tuple:
    """
    Builds a comprehensive grounded context from ALL datasets in the folder.
    Returns (context_string, raw_dict, chart_data, redirect).
    All data comes directly from DB — no hallucination possible.
    """
    raw: Dict[str, Any] = {}
    chart: Optional[dict] = None
    redir: Optional[dict] = None

    # Resolve all datasets in this folder from DB
    folder_datasets = data_tools.folder_dataset_details(folder)
    raw["folder_datasets"] = folder_datasets

    if not folder_datasets:
        context = (
            f"Folder: {folder}\n"
            f"No datasets found in this folder. "
            f"Datasets must have physical_name starting with 'dqm/raw/{folder}/'."
        )
        return context, raw, chart, redir

    lines = [f"Folder: {folder}", f"Total datasets: {len(folder_datasets)}"]

    with_data    = [d for d in folder_datasets if d["has_data"]]
    without_data = [d for d in folder_datasets if not d["has_data"]]

    if with_data:
        healths    = [d["health"] for d in with_data if d["health"] is not None]
        avg_health = round(sum(healths) / len(healths), 1) if healths else None
        total_issues = sum(d["issues"] or 0 for d in with_data)
        total_rows   = sum(d["rows"] or 0 for d in with_data)
        lines.append(f"Profiled datasets: {len(with_data)}/{len(folder_datasets)}")
        if avg_health is not None:
            lines.append(f"Folder avg health: {avg_health}%")
        lines.append(f"Total issues across folder: {total_issues}")
        lines.append(f"Total rows across folder: {total_rows:,}")
    else:
        lines.append("No datasets have been profiled yet in this folder.")

    # Per-dataset breakdown — grounded on real DB data only
    lines.append("\n--- Per-Dataset Details ---")
    for ds in folder_datasets:
        ds_id = ds["id"]
        if not ds["has_data"]:
            lines.append(f"\nDataset: {ds['name']} — NOT PROFILED (no DQ data available)")
            continue

        lines.append(f"\nDataset: {ds['name']} (id={ds_id})")
        lines.append(f"  health={ds['health']}%, issues={ds['issues']}, rows={ds['rows']:,}, columns={ds['total_columns']}, active_rules={ds['active_rules']}")

        # Quality checks for this dataset
        checks = data_tools.quality_checks(ds_id, open_only=True)
        if checks:
            open_c = [c for c in checks if c.get("status", "open").lower() in ("open", "investigating")]
            crit   = sum(1 for c in open_c if (c["severity"] or "").upper() in ("CRITICAL", "HIGH"))
            med    = sum(1 for c in open_c if (c["severity"] or "").upper() == "MEDIUM")
            low    = sum(1 for c in open_c if (c["severity"] or "").upper() in ("LOW", "INFO"))
            top    = [c["check_type"] + " on " + str(c["column"]) for c in open_c[:3]]
            lines.append(f"  open_checks={len(open_c)} (crit/high={crit}, med={med}, low={low}): {', '.join(top)}")
        else:
            lines.append("  open_checks=0")

        # Drift
        drift = data_tools.drift_summary(ds_id)
        if drift:
            lines.append("  drift: " + ", ".join(f"{d['column']}({d['drift_score']}%)" for d in drift[:3]))

        # Worst columns
        cols = data_tools.column_profiles(ds_id)
        bad_cols = [c for c in cols if c["status"] in ("WARNING", "CRITICAL")]
        if bad_cols:
            lines.append("  problem_columns: " + ", ".join(
                f"{c['column']}({c['completeness']}% complete)" for c in bad_cols[:4]
            ))

    # Chart: folder health comparison
    chart_data_pts = [
        {"name": d["name"][:20], "health": d["health"]}
        for d in folder_datasets
        if d["has_data"] and d["health"] is not None
    ]
    if len(chart_data_pts) >= 2:
        chart = {
            "type":  "folder_comparison",
            "data":  sorted(chart_data_pts, key=lambda x: x["health"]),
            "label": f"Health Comparison — {folder}",
        }

    raw["folder_summary"] = {
        "avg_health":   round(sum(d["health"] for d in with_data if d["health"] is not None) / len(with_data), 1) if with_data else None,
        "total_issues": sum(d["issues"] or 0 for d in with_data),
        "dataset_count": len(folder_datasets),
        "profiled_count": len(with_data),
    }

    context = "\n".join(lines)
    return context, raw, chart, redir


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class AICopilotService:

    def __init__(self, db: Session):
        self.db       = db
        self.data     = DataTools(db)
        self.executor = ActionExecutor(db)
        self.master   = MasterAgent()
        self.query    = QueryAgent()
        self.diagnose = DiagnoseAgent()
        self.action   = ActionAgent()
        self.report   = ReportAgent()
        self.suggest  = SuggestionAgent()

    # ── Folder-specific service methods ───────────────────────────────────────

    def get_folder_datasets(self, folder: str) -> List[Dict]:
        """Return all datasets in a folder with health/issues/rows — used by the sidebar."""
        return self.data.folder_dataset_details(folder)

    def get_folder_summary(self, folder: str) -> Dict:
        """Aggregate folder-level summary."""
        return self.data.folder_summary(folder)

    def get_folder_suggestions(self, folder: str) -> Dict:
        """Return proactive suggestions and actions for a folder."""
        try:
            summary  = self.data.folder_summary(folder)
            datasets = summary.get("datasets", [])
            folder_name = folder

            if summary.get("status") == "NO_DATA" or not datasets:
                return {
                    "suggestions": [
                        f"What datasets are in the {folder_name} folder?",
                        "How do I start profiling these datasets?",
                    ],
                    "actions": [
                        {"label": "List folder datasets", "type": "query",
                         "description": f"Show all datasets in the {folder_name} folder."},
                    ],
                }

            worst = summary.get("worst_dataset")
            suggestions = [
                f"What are the top quality issues across all datasets in {folder_name}?",
                f"Which dataset in {folder_name} has the most critical issues?",
                f"Compare health scores of all datasets in {folder_name}.",
                f"Show me a report for all datasets in {folder_name}.",
                f"What columns have the most problems in {folder_name}?",
            ]

            actions = [
                {"label": f"View {folder_name} details",     "type": "query",              "description": f"Summary of all datasets in {folder_name}."},
                {"label": "List all datasets",               "type": "query",              "description": "Show all datasets with health scores."},
                {"label": "Generate folder report",          "type": "generate_report",    "description": "Full DQ report for all datasets."},
                {"label": "Fix all anomalies",               "type": "fix_all_anomalies",  "description": "Resolve open anomalies across datasets."},
                {"label": "Run profiling on all",            "type": "trigger_profiling",  "description": "Re-profile all datasets in this folder."},
            ]
            if worst:
                actions.insert(1, {
                    "label":       f"Diagnose {worst['name']}",
                    "type":        "diagnose",
                    "description": f"Root cause analysis for {worst['name']} ({worst['health']}% health).",
                })

            return {"suggestions": suggestions, "actions": actions, "summary": summary}
        except Exception as e:
            return {"suggestions": [], "actions": [], "error": str(e)}

    def get_folder_history(self, folder: str) -> List[Dict]:
        """Load chat history for a folder."""
        return load_folder_history(folder, MAX_HISTORY)

    def clear_folder_memory(self, folder: str) -> Dict:
        """Clear folder chat history."""
        clear_folder_history(folder)
        return {"status": "cleared"}

    # ── Main run (handles both dataset and folder mode) ───────────────────────

    async def run(
        self,
        message: str,
        dataset_id: Optional[int],
        history: List[Dict],
        folder: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:

        is_folder_mode = bool(folder and not dataset_id)

        # Load persisted history
        if is_folder_mode:
            db_history = load_folder_history(folder, MAX_HISTORY)
        elif dataset_id:
            db_history = load_history(dataset_id, MAX_HISTORY)
        else:
            db_history = []

        merged = db_history[-MAX_HISTORY:] if db_history else history[-MAX_HISTORY:]

        # ── MASTER: Route ─────────────────────────────────────────────────────
        yield {"type": "step", "content": "Master agent routing your request…"}
        await asyncio.sleep(0)

        # Build quick context summary for routing
        quick_ctx = ""
        if is_folder_mode:
            try:
                summary = self.data.folder_summary(folder)
                if summary.get("status") == "OK":
                    quick_ctx = (
                        f"Folder '{folder}': {summary.get('dataset_count')} datasets, "
                        f"avg_health={summary.get('avg_health')}%, "
                        f"total_issues={summary.get('total_issues')}"
                    )
                else:
                    quick_ctx = f"Folder '{folder}': {summary.get('dataset_count', 0)} datasets (not all profiled)"
            except Exception:
                quick_ctx = f"Folder '{folder}'"
        elif dataset_id:
            try:
                s    = self.data.dq_summary(dataset_id)
                info = self.data.dataset_info(dataset_id)
                if s.get("status") != "NO_DATA":
                    quick_ctx = (
                        f"Dataset '{info.get('name')}': "
                        f"health={s.get('avg_health')}%, "
                        f"issues={s.get('issues')}, "
                        f"completeness={s.get('avg_completeness')}%"
                    )
            except Exception:
                pass

        routing  = self.master.route(message, quick_ctx)
        route    = routing.get("route", "query")
        user_ack = routing.get("user_ack", "Working on it…")

        yield {"type": "step", "content": user_ack}
        await asyncio.sleep(0)

        # ── DATA FETCH ────────────────────────────────────────────────────────
        response       = ""
        pending_acts   = []
        redirect       = None
        chart_data     = None
        whatif         = None

        if is_folder_mode:
            # Folder mode: build aggregated context from all datasets in folder
            yield {"type": "step", "content": f"Fetching data for all datasets in '{folder}'…"}
            await asyncio.sleep(0)
            context, raw, chart_data, redir = _build_folder_context(self.data, folder, message)

            if route == "chitchat":
                folder_datasets = raw.get("folder_datasets", [])
                summary         = raw.get("folder_summary", {})
                response = self._build_folder_greeting(folder, folder_datasets, summary)

            elif route == "diagnose":
                yield {"type": "step", "content": "Diagnosis agent analysing root cause…"}
                await asyncio.sleep(0)
                response = self.diagnose.diagnose(message, context, merged)

            elif route == "action":
                yield {"type": "step", "content": "Action agent planning steps…"}
                await asyncio.sleep(0)
                # For folder mode, pick the most relevant dataset_id for the action
                # (worst health or first dataset with data)
                folder_datasets = raw.get("folder_datasets", [])
                with_data = [d for d in folder_datasets if d["has_data"] and d["health"] is not None]
                target_ds_id = None
                if with_data:
                    target_ds_id = min(with_data, key=lambda d: d["health"])["id"]
                elif folder_datasets:
                    target_ds_id = folder_datasets[0]["id"]

                if target_ds_id:
                    plan     = self.action.plan(message, context, target_ds_id)
                    response = plan.get("response", "Here's what I've planned:")
                    for act in plan.get("actions", []):
                        act_type = act.get("type", "")
                        if act_type.startswith("redirect_"):
                            tab_map = {
                                "redirect_lineage":    "lineage",
                                "redirect_kg":         "knowledge-graph",
                                "redirect_monitoring": "monitoring",
                                "redirect_anomalies":  "anomalies",
                                "redirect_scorecards": "scorecards",
                                "redirect_dq_rules":   "dq-rules",
                            }
                            redirect = {"tab": tab_map.get(act_type, ""), "dataset_id": target_ds_id}
                            continue
                        action_id = str(uuid.uuid4())[:8]
                        params    = act.get("params", {})
                        params["dataset_id"] = target_ds_id
                        pending = {
                            "action_id":   action_id,
                            "type":        act_type,
                            "label":       act.get("label", "Apply Action"),
                            "description": act.get("description", ""),
                            "params":      params,
                        }
                        _pending_actions[action_id] = pending
                        pending_acts.append(pending)
                else:
                    response = f"No profiled datasets found in folder '{folder}'. Please run DQ Scoring on datasets first."

            elif route == "report":
                yield {"type": "step", "content": "Report agent generating folder report…"}
                await asyncio.sleep(0)
                response = self.report.generate(message, context, merged)

            else:  # query
                yield {"type": "step", "content": "Query agent fetching answer…"}
                await asyncio.sleep(0)
                response = self.query.answer(message, context, merged)

        else:
            # Dataset mode (original logic)
            tools = _select_tools(message, route)
            tool_labels = {
                "summary": "DQ summary", "columns": "column profiles",
                "checks": "quality checks", "drift": "drift data",
                "rules": "active rules", "history": "run history",
                "trend": "health trend", "schema": "schema history",
                "alerts": "alerts", "anomalies": "anomalies",
                "scorecards": "scorecard metrics", "monitoring": "monitoring data",
                "lineage": "lineage graph", "all_datasets": "all datasets",
            }
            label_strs = [tool_labels[t] for t in tools if t in tool_labels and t != "info"]
            if label_strs:
                yield {"type": "step", "content": f"Fetching {', '.join(label_strs[:4])}…"}
                await asyncio.sleep(0)

            context, raw, chart_data, redir = _build_context(self.data, dataset_id, tools)

            # What-if simulation
            msg_lower = message.lower()
            if any(w in msg_lower for w in ["what if", "what would happen", "simulate", "how many would fail", "if i add"]):
                cols = raw.get("columns", [])
                for col in cols:
                    if col["column"].lower() in msg_lower:
                        condition = f"{col['column']} IS NOT NULL"
                        whatif = self.data.what_if_rule(dataset_id, col["column"], condition)
                        context += f"\nSimulation: {whatif.get('message', '')}"
                        break

            if route == "chitchat":
                yield {"type": "step", "content": "Preparing response…"}
                await asyncio.sleep(0)
                ds_name  = raw.get("info", {}).get("name", "your dataset") if dataset_id else "your datasets"
                response = (
                    f"Hi! I'm your **AI Data Quality Copilot** for **{ds_name}**.\n\n"
                    f"I can help you with:\n"
                    f"• **Answer questions** — quality metrics, column profiles, scorecards, monitoring, trends\n"
                    f"• **Diagnose issues** — root cause analysis with evidence from real data\n"
                    f"• **Create/edit/delete DQ rules** — just describe what you want in natural language\n"
                    f"• **Fix anomalies** — fix specific or all anomalies with a single request\n"
                    f"• **Dismiss alerts** — clear alerts by severity or type\n"
                    f"• **View scorecards** — KPI, velocity, coverage, heatmap, freshness, risk, incident timeline\n"
                    f"• **View monitoring** — SLA status, column health, drift, risk forecast\n"
                    f"• **Show lineage** — upstream/downstream relationships\n"
                    f"• **Generate reports** — full structured quality reports\n"
                    f"• **Compare datasets** — rank all datasets by health score\n"
                    f"• **Simulate rules** — 'what if I add a NOT NULL rule on email?'\n"
                    f"• **Schedule profiling** — set automated profiling times\n\nAsk me anything!"
                )

            elif route == "diagnose":
                yield {"type": "step", "content": "Diagnosis agent analysing root cause…"}
                await asyncio.sleep(0)
                response = self.diagnose.diagnose(message, context, merged)

            elif route == "action":
                yield {"type": "step", "content": "Action agent planning steps…"}
                await asyncio.sleep(0)
                plan     = self.action.plan(message, context, dataset_id or 0)
                response = plan.get("response", "Here's what I've planned:")
                for act in plan.get("actions", []):
                    act_type = act.get("type", "")
                    if act_type.startswith("redirect_"):
                        tab_map = {
                            "redirect_lineage":    "lineage",
                            "redirect_kg":         "knowledge-graph",
                            "redirect_monitoring": "monitoring",
                            "redirect_anomalies":  "anomalies",
                            "redirect_scorecards": "scorecards",
                            "redirect_dq_rules":   "dq-rules",
                        }
                        redirect = {"tab": tab_map.get(act_type, ""), "dataset_id": dataset_id}
                        continue
                    action_id = str(uuid.uuid4())[:8]
                    params    = act.get("params", {})
                    params["dataset_id"] = dataset_id
                    pending = {
                        "action_id":   action_id,
                        "type":        act_type,
                        "label":       act.get("label", "Apply Action"),
                        "description": act.get("description", ""),
                        "params":      params,
                    }
                    _pending_actions[action_id] = pending
                    pending_acts.append(pending)

            elif route == "report":
                yield {"type": "step", "content": "Report agent generating structured report…"}
                await asyncio.sleep(0)
                response = self.report.generate(message, context, merged)

            else:  # query
                yield {"type": "step", "content": "Query agent fetching answer…"}
                await asyncio.sleep(0)
                msg_lower = message.lower()
                cols      = raw.get("columns", [])
                deep_col  = None
                for c in cols:
                    if c["column"].lower() in msg_lower and any(
                        w in msg_lower for w in ["tell me about", "deep dive", "detail", "profile", "everything about", "explain"]
                    ):
                        deep_col = c
                        break
                if deep_col:
                    col_rules  = [r for r in raw.get("rules", [])  if r.get("column") == deep_col["column"]]
                    col_checks = [ch for ch in raw.get("checks", []) if ch.get("column") == deep_col["column"]]
                    col_ctx = (
                        f"Deep dive on column `{deep_col['column']}`:\n"
                        f"- Type: {deep_col['type']}\n"
                        f"- Completeness: {deep_col['completeness']}% ({deep_col['null_count']} nulls)\n"
                        f"- Uniqueness: {deep_col['uniqueness']}% ({deep_col['distinct_count']} distinct values)\n"
                        f"- Health score: {deep_col['health_score']}% ({deep_col['status']})\n"
                        f"- Active rules: {len(col_rules)}\n"
                        f"- Quality violations: {len(col_checks)} ({', '.join(c['check_type'] for c in col_checks[:3]) or 'none'})"
                    )
                    response = self.query.answer(message, col_ctx, merged)
                else:
                    response = self.query.answer(message, context, merged)

        # ── SUGGESTIONS ───────────────────────────────────────────────────────
        yield {"type": "step", "content": "Generating suggestions…"}
        await asyncio.sleep(0)
        sug = self.suggest.suggest(message, response, quick_ctx)

        # ── PERSIST HISTORY ───────────────────────────────────────────────────
        if is_folder_mode:
            _save_folder_history(folder, "user",      message,  route)
            _save_folder_history(folder, "assistant", response, route)
        elif dataset_id:
            _save_history(dataset_id, "user",      message,  route)
            _save_history(dataset_id, "assistant", response, route)

        yield {
            "type":                "done",
            "content":             response,
            "suggested_followups": sug.get("followups", []),
            "suggested_actions":   sug.get("actions", []),
            "pending_actions":     pending_acts,
            "pending_action":      pending_acts[0] if len(pending_acts) == 1 else None,
            "intent":              route,
            "chart_data":          chart_data,
            "whatif":              whatif,
            "redirect":            redirect,
        }

    def _build_folder_greeting(self, folder: str, folder_datasets: List[Dict], summary: Dict) -> str:
        """Build a grounded greeting for folder mode — uses only real DB data."""
        total = len(folder_datasets)
        if total == 0:
            return (
                f"Hi! I'm your **DQ Assistant** for folder **{folder}**.\n\n"
                f"No datasets found. Make sure datasets are registered with physical paths "
                f"starting with `dqm/raw/{folder}/`."
            )
        with_data    = [d for d in folder_datasets if d["has_data"]]
        without_data = [d for d in folder_datasets if not d["has_data"]]

        avg_h    = summary.get("avg_health")
        t_issues = summary.get("total_issues", 0)

        msg = f"Hi! I'm your **DQ Assistant** for folder **{folder}**.\n\n"
        if avg_h is not None:
            msg += f"This folder contains **{total}** dataset(s) with average health **{avg_h}%** and **{t_issues}** total open issues.\n\n"
        else:
            msg += f"This folder contains **{total}** dataset(s).\n\n"

        msg += "**Datasets in this folder:**\n"
        for ds in folder_datasets:
            if ds["has_data"] and ds["health"] is not None:
                icon = "🟢" if ds["health"] >= 80 else "🟡" if ds["health"] >= 60 else "🔴"
                msg += f"{icon} **{ds['name']}** — {ds['health']}% health, {ds['issues']} issues, {ds['rows']:,} rows\n"
            else:
                msg += f"⚪ **{ds['name']}** — not profiled yet\n"

        if without_data:
            msg += f"\n{len(without_data)} dataset(s) need DQ Scoring to generate quality metrics."

        msg += "\n\nAsk me anything about these datasets!"
        return msg

    # ── Execute Action ─────────────────────────────────────────────────────────

    def execute_action(self, action_id: str, dataset_id: Optional[int]) -> dict:
        action = _pending_actions.pop(action_id, None)
        if not action:
            return {"status": "error", "message": "Action not found or already executed."}

        params   = action.get("params", {})
        act_type = action.get("type")
        # params["dataset_id"] is set at planning time (may be a specific folder dataset)
        ds_id    = params.get("dataset_id") or dataset_id

        try:
            if act_type == "create_rule":
                r = self.executor.create_dq_rule(
                    ds_id, params.get("rule_name", "Agent Rule"),
                    params.get("column_name", ""), params.get("condition_sql", ""),
                    params.get("severity", "Medium"), params.get("rule_type", "Validity"),
                )
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Rule '{r['rule_name']}' created on `{r['column_name']}`."}

            elif act_type == "delete_rule":
                r = self.executor.delete_dq_rule(ds_id, params.get("rule_code", ""))
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Rule '{r.get('rule_name')}' deactivated."}

            elif act_type == "update_rule":
                r = self.executor.update_dq_rule(ds_id, params.get("rule_code", ""), params.get("updates", {}))
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Rule '{r.get('rule_name')}' updated."}

            elif act_type == "dismiss_alerts":
                r = self.executor.dismiss_alerts(ds_id, params.get("severity"), params.get("check_type"))
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Dismissed {r['dismissed']} alert(s)."}

            elif act_type == "fix_anomaly":
                r = self.executor.fix_anomaly(ds_id, params.get("check_id", 0))
                return {"status": "success", "action": act_type, "result": r,
                        "message": "Anomaly fix applied."}

            elif act_type == "fix_all_anomalies":
                r = self.executor.fix_all_anomalies(ds_id)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Fixed {r['fixed']} anomaly/anomalies."}

            elif act_type == "resolve_checks_by_severity":
                sev = params.get("severity", "LOW").upper()
                r   = self.executor.resolve_checks_by_severity(ds_id, sev)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Resolved {r['resolved']} {sev}-severity check(s). Types: {', '.join(r.get('check_types', [])[:4])}"}

            elif act_type == "trigger_profiling":
                r = self.executor.trigger_profiling(ds_id)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Profiling complete. Run #{r['run_id']} — {r['rows']:,} rows."}

            elif act_type == "set_baseline":
                r = self.executor.set_baseline(ds_id)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Baseline set from run #{r.get('run_id')}."}

            elif act_type in ("resolve_checks", "resolve_check"):
                r = self.executor.resolve_checks(ds_id, params.get("check_type"), params.get("column_name"))
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Resolved {r['resolved']} check(s)."}

            elif act_type == "resolve_single_check":
                r = self.executor.resolve_single_check(
                    ds_id, check_id=params.get("check_id", 0),
                    check_type=params.get("check_type"), column_name=params.get("column_name"),
                    severity=params.get("severity"),
                )
                if r.get("resolved", 0) > 0:
                    return {"status": "success", "action": act_type, "result": r,
                            "message": f"Resolved anomaly '{r.get('check_type')}' on column '{r.get('column')}'."}
                else:
                    return {"status": "success", "action": act_type, "result": r,
                            "message": r.get("message", "No matching open check found.")}

            elif act_type == "pause_all_rules":
                r = self.executor.pause_all_rules(ds_id)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Paused {r['paused']} DQ rule(s)."}

            elif act_type == "resume_all_rules":
                r = self.executor.resume_all_rules(ds_id)
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Resumed {r['resumed']} DQ rule(s)."}

            elif act_type == "schedule_profiling":
                r = self.executor.schedule_profiling(ds_id, params.get("times", []), params.get("on_new_data", False))
                return {"status": "success", "action": act_type, "result": r,
                        "message": f"Profiling scheduled at {', '.join(r.get('times', []))}. On new data: {r.get('on_new_data')}."}

            else:
                return {"status": "error", "message": f"Unknown action: {act_type}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_bulk(self, action_ids: List[str], dataset_id: Optional[int]) -> List[dict]:
        return [self.execute_action(aid, dataset_id) for aid in action_ids]

    # ── Proactive Suggestions ──────────────────────────────────────────────────

    def get_proactive_suggestions(self, dataset_id: int) -> dict:
        try:
            s    = self.data.dq_summary(dataset_id)
            chks = self.data.quality_checks(dataset_id)
            info = self.data.dataset_info(dataset_id)
            name = info.get("name", "this dataset")
            if s.get("status") == "NO_DATA":
                return {
                    "suggestions": ["Run DQ Scoring to profile this dataset.", "What checks are available?"],
                    "actions": [{"label": "Run DQ Scoring", "type": "trigger_profiling",
                                 "description": "Profile this dataset to generate metrics."}],
                }
            crit    = [c for c in chks if c["severity"] in ("CRITICAL", "HIGH")]
            top     = crit[0] if crit else (chks[0] if chks else None)
            actions = [
                {"label": "Explain quality issues",   "type": "query",              "description": f"Full breakdown of issues in {name}."},
                {"label": "Generate quality report",  "type": "generate_report",    "description": "Full structured report."},
                {"label": "View Scorecards",          "type": "redirect_scorecards","description": "Open scorecard dashboard."},
                {"label": "View Monitoring",          "type": "redirect_monitoring","description": "Open monitoring dashboard."},
                {"label": "Run fresh profiling",      "type": "trigger_profiling",  "description": "Re-run DQ scoring for fresh metrics."},
            ]
            if top:
                check_label = top["check_type"].replace("_", " ").title()
                actions.insert(1, {
                    "label":       f"Fix: {check_label} on {top['column']}",
                    "type":        "create_rule",
                    "description": f"{top['violation_count']} violations detected.",
                })
            return {
                "suggestions": [
                    f"What are the top quality issues in {name}?",
                    "Why did the health score change?",
                    f"Show me the scorecard for {name}.",
                    "Compare quality across all datasets.",
                    "Show me the lineage for this dataset.",
                ],
                "actions": actions, "summary": s, "top_issue": top,
            }
        except Exception as e:
            return {"suggestions": [], "actions": [], "error": str(e)}

    # ── Feedback & Pins ────────────────────────────────────────────────────────

    def record_feedback(self, message_id: str, rating: int, comment: Optional[str],
                        dataset_id: Optional[int], folder: Optional[str] = None) -> dict:
        save_feedback(message_id, dataset_id or 0, rating, comment or "")
        return {"status": "recorded"}

    def pin_message(self, dataset_id: int, msg_id: str, content: str) -> dict:
        save_pin(dataset_id, msg_id, content)
        return {"status": "pinned"}

    def get_pinned(self, dataset_id: int) -> list:
        return get_pins_db(dataset_id)

    def unpin_message(self, dataset_id: int, message_id: str) -> dict:
        delete_pin(dataset_id, message_id)
        return {"status": "unpinned"}

    def clear_memory(self, dataset_id: int) -> dict:
        clear_history(dataset_id)
        return {"status": "cleared"}

    def get_history(self, dataset_id: int) -> list:
        return load_history(dataset_id, MAX_HISTORY)