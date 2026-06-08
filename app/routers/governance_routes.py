# governance_routes.py — FULLY DYNAMIC, nothing hardcoded.
# Email functionality completely removed.

from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy import Column, String, Integer, Boolean, Text, JSON, func, text
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import Base, engine, SessionLocal
from app.models import GovernanceNotification
from datetime import datetime, date
import uuid, os, time
from app.routers.notification_inbox_routes import create_inbox_notification, _push_in_app

governance_router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Models ───────────────────────────────────────────────────────────────────

class GovernancePolicy(Base):
    __tablename__ = "governance_policies"
    __table_args__ = {'extend_existing': True}
    id             = Column(String, primary_key=True)
    name           = Column(String, nullable=False)
    description    = Column(Text, default="")
    status         = Column(String, default="Draft")
    policy_type    = Column(String, default="Quality")
    datasets_count = Column(Integer, default=0)
    rules          = Column(JSON, default=list)
    created_at     = Column(String, default=lambda: str(date.today()))
    updated_at     = Column(String, default=lambda: str(date.today()))

class GovernanceUser(Base):
    __tablename__ = "governance_users"
    __table_args__ = {'extend_existing': True}
    id              = Column(String, primary_key=True)
    name            = Column(String, nullable=False)
    email           = Column(String, unique=True, nullable=False, index=True)
    role            = Column(String, default="Viewer")
    status          = Column(String, default="Pending")
    login_method    = Column(String, default="unknown")
    last_active     = Column(String, default="Never")
    created_at      = Column(String, default=lambda: str(date.today()))
    datasets_access = Column(Integer, default=0)

class GovernanceSystemConfig(Base):
    __tablename__ = "governance_system_config"
    __table_args__ = {'extend_existing': True}
    key   = Column(String, primary_key=True)
    value = Column(Text)

class GovernanceAuditLog(Base):
    __tablename__ = "governance_audit_log"
    __table_args__ = {'extend_existing': True}
    id             = Column(String, primary_key=True)
    timestamp      = Column(String, nullable=False)
    user           = Column(String, default="System")
    action         = Column(String, nullable=False)
    resource_type  = Column(String, nullable=False)
    resource_name  = Column(String, default="")
    change_summary = Column(Text, default="")
    ip_address     = Column(String, default="unknown")
    severity       = Column(String, default="info")

class GovernanceDismissedSuggestion(Base):
    __tablename__ = "governance_dismissed_suggestions"
    __table_args__ = {'extend_existing': True}
    id = Column(String, primary_key=True)

# NOTE: create_all handled by main.py — using checkfirst=True here to avoid conflicts
try:
    Base.metadata.create_all(bind=engine, checkfirst=True)
except Exception:
    pass

# ── Safe migrations ────────────────────────────────────────────────────────────
for _migration_sql in [
    "ALTER TABLE governance_notifications ADD COLUMN slack_webhook TEXT",
    "ALTER TABLE governance_notifications ADD COLUMN recipient_email TEXT",
    "ALTER TABLE governance_notifications ADD COLUMN channel TEXT DEFAULT 'in_app'",
]:
    try:
        with engine.connect() as _c:
            _c.execute(text(_migration_sql))
            _c.commit()
    except Exception:
        pass

# ─── Seeds ────────────────────────────────────────────────────────────────────

_DEFAULT_NOTIFICATIONS = [
    {"id":"n_quality",    "title":"Quality Score Alerts",  "description":"Get notified when quality scores drop below threshold",  "enabled":True,  "channel":"in_app"},
    {"id":"n_anomaly",    "title":"Anomaly Detection",     "description":"Receive alerts for detected data anomalies",              "enabled":True,  "channel":"in_app"},
    {"id":"n_rule",       "title":"Rule Failures",         "description":"Get notified when data quality rules fail",               "enabled":True,  "channel":"in_app"},
    {"id":"n_daily",      "title":"Daily Summary",         "description":"Receive daily summary of data quality metrics",           "enabled":False, "channel":"in_app"},
    {"id":"n_weekly",     "title":"Weekly Reports",        "description":"Get weekly data quality reports",                         "enabled":True,  "channel":"in_app"},
    {"id":"n_schema",     "title":"Schema Changes",        "description":"Alert when schema changes are detected",                  "enabled":True,  "channel":"in_app"},
    {"id":"n_datasource", "title":"New Data Sources",      "description":"Notify when new data sources are connected",              "enabled":False, "channel":"in_app"},
    {"id":"n_compliance", "title":"Compliance Violations", "description":"Immediate alerts for compliance policy violations",       "enabled":True,  "channel":"in_app"},
    {"id":"n_policy",     "title":"Policy Changes",        "description":"Notify when governance policies are created or changed",  "enabled":True,  "channel":"in_app"},
    {"id":"n_user",       "title":"User Management",       "description":"Notify when users are added, edited or deactivated",      "enabled":True,  "channel":"in_app"},
]

_DEFAULT_SYSTEM_CONFIG = {
    "quality_score_threshold":     "70",
    "data_retention_period_days":  "2555",
    "dq_scoring_schedule":         "daily",
    "max_concurrent_rules":        "50",
    "enable_auto_remediation":     "false",
    "enable_ml_anomaly_detection": "true",
    "alert_cooldown_minutes":      "30",
    "max_dataset_size_gb":         "500",
    "slack_webhook_url":           "",
    "app_version":                 "2.5.0",
    "environment":                 "Production",
    "platform":                    "Azure Cloud",
    "data_source":                 "Azure Blob Storage",
    "min_completeness_pct":        "90",
    "critical_issue_threshold":    "5",
    "notify_on_score_drop":        "true",
    "notify_on_schema_change":     "true",
    "notify_on_new_anomaly":       "true",
    "notification_digest_hour":    "8",
    # ── SLA thresholds (Monitoring → SLA & Thresholds tab) ──────────────────
    "sla_threshold_completeness":  "80",
    "sla_threshold_uniqueness":    "50",
    "sla_threshold_validity":      "80",
    "sla_threshold_consistency":   "80",
    "sla_threshold_accuracy":      "80",
    "sla_threshold_integrity":     "80",
    "sla_warning_offset":          "5",
}

def _seed(db: Session):
    if db.query(GovernanceNotification).count() == 0:
        for n in _DEFAULT_NOTIFICATIONS:
            db.merge(GovernanceNotification(**n))
    existing_keys = {row.key for row in db.query(GovernanceSystemConfig).all()}
    for k, v in _DEFAULT_SYSTEM_CONFIG.items():
        if k not in existing_keys:
            db.add(GovernanceSystemConfig(key=k, value=str(v)))
    db.commit()

with SessionLocal() as _db:
    _seed(_db)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ip(r): return r.client.host if r.client else "unknown"
def _usr(r): return r.headers.get("X-User-Name","Admin User")

def _audit(db, action, rt, rn, cs, sev="info", user="System", ip="unknown"):
    e = GovernanceAuditLog(
        id=f"a_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user=user, action=action, resource_type=rt,
        resource_name=rn, change_summary=cs, ip_address=ip, severity=sev,
    )
    db.add(e); db.commit()
    return {c.name: getattr(e,c.name) for c in e.__table__.columns}

def _p2d(p): return {"id":p.id,"name":p.name,"description":p.description,"status":p.status,"policy_type":p.policy_type,"datasets_count":p.datasets_count,"rules":p.rules or [],"created_at":p.created_at,"updated_at":p.updated_at}
def _u2d(u): return {"id":u.id,"name":u.name,"email":u.email,"role":u.role,"status":u.status,"login_method":u.login_method,"last_active":u.last_active,"created_at":u.created_at,"datasets_access":u.datasets_access}

def _n2d(n):
    return {
        "id":              n.id,
        "title":           n.title,
        "description":     n.description,
        "enabled":         bool(n.enabled),
        "channel":         n.channel,
        "slack_webhook":   getattr(n, "slack_webhook", None) or "",
    }

def _cfg(db) -> dict:
    r = {x.key: x.value for x in db.query(GovernanceSystemConfig).all()}

    def _bool(key, default="true"):
        return r.get(key, default).lower() == "true"

    def _int(key, default):
        try:
            return int(r.get(key, default))
        except (ValueError, TypeError):
            return default

    return {
        "quality_score_threshold":     _int("quality_score_threshold", 70),
        "data_retention_period_days":  _int("data_retention_period_days", 2555),
        "max_concurrent_rules":        _int("max_concurrent_rules", 50),
        "alert_cooldown_minutes":      _int("alert_cooldown_minutes", 30),
        "max_dataset_size_gb":         _int("max_dataset_size_gb", 500),
        "min_completeness_pct":        _int("min_completeness_pct", 90),
        "critical_issue_threshold":    _int("critical_issue_threshold", 5),
        "notification_digest_hour":    _int("notification_digest_hour", 8),
        "dq_scoring_schedule":         r.get("dq_scoring_schedule", "daily"),
        "slack_webhook_url":           r.get("slack_webhook_url", ""),
        "app_version":                 r.get("app_version", "2.5.0"),
        "environment":                 r.get("environment", "Production"),
        "platform":                    r.get("platform", "Azure Cloud"),
        "data_source":                 r.get("data_source", "Azure Blob Storage"),
        "enable_auto_remediation":     _bool("enable_auto_remediation", "false"),
        "enable_ml_anomaly_detection": _bool("enable_ml_anomaly_detection", "true"),
        "notify_on_score_drop":        _bool("notify_on_score_drop", "true"),
        "notify_on_schema_change":     _bool("notify_on_schema_change", "true"),
        "notify_on_new_anomaly":       _bool("notify_on_new_anomaly", "true"),
        # ── SLA thresholds (Monitoring → SLA & Thresholds tab) ──────────────
        "sla_threshold_completeness":  _int("sla_threshold_completeness", 80),
        "sla_threshold_uniqueness":    _int("sla_threshold_uniqueness",   50),
        "sla_threshold_validity":      _int("sla_threshold_validity",     80),
        "sla_threshold_consistency":   _int("sla_threshold_consistency",  80),
        "sla_threshold_accuracy":      _int("sla_threshold_accuracy",     80),
        "sla_threshold_integrity":     _int("sla_threshold_integrity",    80),
        "sla_warning_offset":          _int("sla_warning_offset",          5),
    }

def _get_system_config(db) -> dict:
    """Read all config fields needed for sending notifications."""
    r = {x.key: x.value for x in db.query(GovernanceSystemConfig).all()}
    return {
        "slack_webhook_url":   r.get("slack_webhook_url", "").strip(),
    }

def _send_slack_notif(title: str, message: str, severity: str, link: Optional[str], system_config: dict) -> bool:
    import json as _json
    import urllib.request

    webhook = system_config.get("slack_webhook_url", "").strip()
    if not webhook:
        return False
    emoji   = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
    text    = f"{emoji} *{title}*\n{message}"
    if link:
        text += f"\n<http://localhost:5173{link}|View in AI DQM>"
    payload = _json.dumps({"text": text}).encode()
    try:
        req = urllib.request.Request(
            webhook, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("[notif] Slack failed: %s", e)
        return False

# ─── Notification dispatcher ───────────────────────────────────────────────────

_CATEGORY_TO_NOTIF_ID = {
    "policy":     "n_policy",
    "user":       "n_user",
    "quality":    "n_quality",
    "anomaly":    "n_anomaly",
    "rule":       "n_rule",
    "schema":     "n_schema",
    "compliance": "n_compliance",
    "datasource": "n_datasource",
}

def _notify_for_action(
    db: Session,
    category: str,
    title: str,
    message: str,
    severity: str = "info",
    link: str = None,
):
    import logging as _logging
    _log = _logging.getLogger(__name__)

    try:
        notif_id = _CATEGORY_TO_NOTIF_ID.get(category)
        if not notif_id:
            _push_in_app(title, message, notif_type="ALERT", category=category.capitalize(), severity=severity, link=link)
            return

        notif = db.query(GovernanceNotification).filter(
            GovernanceNotification.id == notif_id
        ).first()

        if not notif or not notif.enabled:
            _log.info(f"[notif] Suppressed (disabled or not found): {title}")
            return

        channel = notif.channel or "in_app"

        # Always push in-app
        _push_in_app(title, message, notif_type="ALERT", category=category.capitalize(), severity=severity, link=link)

        # Extra channel delivery
        if channel == "slack":
            webhook_override = getattr(notif, "slack_webhook", "").strip()
            cfg = _get_system_config(db)
            if webhook_override:
                cfg = {**cfg, "slack_webhook_url": webhook_override}
            _send_slack_notif(title, message, severity, link, cfg)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[notif] _notify_for_action failed: {e}")

# ─── Classification ────────────────────────────────────────────────────────────

def _classify(n, t):
    n,t = n.lower(), t.lower()
    if any(k in n for k in ["ssn","social_security","password","secret","token","credit_card","card_number","cvv","dob","birth","salary","income","tax_id","national_id"]): return "Confidential"
    if any(k in n for k in ["medical","health","diagnosis","prescription","legal","compliance","restricted","classified","insurance","nda"]): return "Restricted"
    if any(k in n for k in ["internal","employee","department","manager","cost","budget","revenue","profit","margin","vendor","supplier","contract"]) or t in ["integer","float","numeric","decimal"]: return "Internal"
    return "Public"

def _dedup_cols():
    try:
        with engine.connect() as conn:
            return conn.execute(text("""
                SELECT cp.column_name, cp.data_type,
                       COALESCE(d.display_name, d.physical_name, 'Unknown Dataset') AS dataset_name,
                       cp.null_count, cp.distinct_count, cp.health_score, pr.dataset_id
                FROM column_profiles cp
                JOIN profiling_runs pr ON cp.profiling_run_id = pr.id
                JOIN (SELECT dataset_id, MAX(id) AS max_run_id FROM profiling_runs GROUP BY dataset_id) latest
                     ON pr.dataset_id=latest.dataset_id AND pr.id=latest.max_run_id
                LEFT JOIN datasets d ON pr.dataset_id=d.id
                ORDER BY dataset_name, cp.column_name
            """)).fetchall()
    except Exception: return []

def _get_classifications():
    rows = _dedup_cols()
    c = {"Confidential":0,"Internal":0,"Public":0,"Restricted":0}
    for row in rows:
        cat = _classify(row[0] or "", row[1] or ""); c[cat] += 1
    t = len(rows)
    return [
        {"id":"c1","category":"Confidential","columns_count":c["Confidential"],"description":"Highly sensitive data — PII, credentials, financial","color":"red",   "pct":round(c["Confidential"]/t*100,1) if t else 0},
        {"id":"c2","category":"Internal",    "columns_count":c["Internal"],    "description":"Internal business data not for external sharing",    "color":"yellow","pct":round(c["Internal"]/t*100,1)     if t else 0},
        {"id":"c3","category":"Public",      "columns_count":c["Public"],      "description":"Data safe for external publication",                 "color":"green", "pct":round(c["Public"]/t*100,1)       if t else 0},
        {"id":"c4","category":"Restricted",  "columns_count":c["Restricted"],  "description":"Restricted access — legal and compliance only",      "color":"orange","pct":round(c["Restricted"]/t*100,1)   if t else 0},
    ]

def _sys_info(db):
    try:
        from sqlalchemy import inspect as si
        tables = si(engine).get_table_names()
        dc=rc=rc2=ac=0
        with engine.connect() as conn:
            if "datasets" in tables:        dc  = conn.execute(text("SELECT COUNT(*) FROM datasets")).scalar() or 0
            if "dq_rules" in tables:        rc  = conn.execute(text("SELECT COUNT(*) FROM dq_rules")).scalar() or 0
            if "profiling_runs" in tables:  rc2 = conn.execute(text("SELECT COUNT(*) FROM profiling_runs")).scalar() or 0
            if "temporal_checks" in tables: ac  = conn.execute(text("SELECT COUNT(*) FROM temporal_checks")).scalar() or 0
        db_path = "ai_dqm.db"
        sg = round(os.path.getsize(db_path)/(1024**3),4) if os.path.exists(db_path) else 0.0
        try:
            import psutil; uh = round((time.time()-psutil.Process(os.getpid()).create_time())/3600,1)
        except: uh = 0.0
        cfg = {x.key:x.value for x in db.query(GovernanceSystemConfig).all()}
        lb = "Never"
        if "profiling_runs" in tables and rc2>0:
            with engine.connect() as conn:
                ts = conn.execute(text("SELECT MAX(timestamp) FROM profiling_runs")).scalar()
                if ts: lb = str(ts)[:19]+" UTC"
        return {"version":cfg.get("app_version","2.5.0"),"database":"SQLite (ai_dqm.db)","storage_used_gb":sg,"storage_used_tb":round(sg/1024,6),"storage_total_tb":10.0,"active_connections":dc+rc,"uptime_hours":uh,"last_backup":lb,"environment":cfg.get("environment","Production"),"platform":cfg.get("platform","Azure Cloud"),"data_source":cfg.get("data_source","Azure Blob Storage"),"dataset_count":dc,"rule_count":rc,"profiling_run_count":rc2,"anomaly_count":ac}
    except Exception as e:
        return {"version":"2.5.0","database":"SQLite (ai_dqm.db)","storage_used_tb":0.0,"storage_total_tb":10.0,"active_connections":0,"uptime_hours":0,"last_backup":"Unknown","environment":"Production","platform":"Azure Cloud","data_source":"Azure Blob Storage","dataset_count":0,"rule_count":0,"profiling_run_count":0,"anomaly_count":0,"error":str(e)}

# ─── Policy suggestions ────────────────────────────────────────────────────────

SUGGESTION_TEMPLATES = [
    {"id":"sug_null",  "name":"Null Value Threshold Policy",   "description":"Flag datasets where null rate exceeds 15% in key columns",         "policy_type":"Quality",    "priority":"High",  "check_types":["null","completeness","null_rate","missing"],      "min_count":3,"reason_template":"Multiple datasets showing high null rates"},
    {"id":"sug_dup",   "name":"Duplicate Record Prevention",   "description":"Block ingestion when duplicate row rate exceeds 5%",                "policy_type":"Quality",    "priority":"High",  "check_types":["duplicate","uniqueness","duplicates"],             "min_count":5,"reason_template":"Duplicate anomalies firing consistently"},
    {"id":"sug_fresh", "name":"Stale Data Freshness Policy",   "description":"Alert when datasets are not refreshed within SLA window",           "policy_type":"Governance", "priority":"Medium","check_types":["freshness","timeliness","stale","recency"],        "min_count":2,"reason_template":"Pipelines missing freshness SLAs"},
    {"id":"sug_schema","name":"Schema Drift Auto-Quarantine",  "description":"Automatically quarantine datasets on unexpected schema changes",     "policy_type":"Governance", "priority":"Medium","check_types":["schema","schema_change","drift","structural"],      "min_count":2,"reason_template":"Schema changes causing downstream failures"},
    {"id":"sug_ref",   "name":"Cross-Source Consistency Check","description":"Validate reference data matches across joined sources on each run", "policy_type":"Quality",    "priority":"Low",   "check_types":["referential","integrity","consistency","foreign_key"],"min_count":5,"reason_template":"Referential integrity anomalies detected"},
]

def _anomaly_counts(db):
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT check_type, COUNT(*) FROM temporal_checks WHERE status!='resolved' GROUP BY check_type")).fetchall()
        return {(r[0] or "").lower().replace(" ","_"):r[1] for r in rows}
    except: return {}

def _has_policy(db, name):
    return db.query(GovernancePolicy).filter(func.lower(GovernancePolicy.name).contains(name.lower().split()[0])).first() is not None

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@governance_router.get("/policies")
def get_policies(db: Session=Depends(get_db)):
    return [_p2d(p) for p in db.query(GovernancePolicy).order_by(GovernancePolicy.created_at.desc()).all()]

@governance_router.post("/policies", status_code=201)
def create_policy(body: dict, request: Request, db: Session=Depends(get_db)):
    p = GovernancePolicy(
        id=f"p_{uuid.uuid4().hex[:8]}",
        name=body.get("name",""),
        description=body.get("description",""),
        status=body.get("status","Draft"),
        policy_type=body.get("policy_type","Quality"),
        datasets_count=body.get("datasets_count",0),
        rules=body.get("rules",[]),
        created_at=str(date.today()),
        updated_at=str(date.today()),
    )
    db.add(p)
    _audit(db, "Policy Created", "Policy", p.name, f"New {p.status} policy", "info", _usr(request), _ip(request))
    _notify_for_action(db, "policy", f"Policy Created: {p.name}", f"A new {p.policy_type} policy '{p.name}' was created with status '{p.status}'.", severity="info", link="/settings?tab=policies")
    return _p2d(p)

@governance_router.put("/policies/{pid}")
def update_policy(pid: str, body: dict, request: Request, db: Session=Depends(get_db)):
    p = db.query(GovernancePolicy).filter(GovernancePolicy.id==pid).first()
    if not p: raise HTTPException(404, "Not found")
    for f in ["name","description","status","policy_type","datasets_count","rules"]:
        if f in body: setattr(p, f, body[f])
    p.updated_at = str(date.today())
    _audit(db, "Policy Edited", "Policy", p.name, "Updated", "info", _usr(request), _ip(request))
    _notify_for_action(db, "policy", f"Policy Updated: {p.name}", f"The policy '{p.name}' was updated. Current status: {p.status}.", severity="info", link="/settings?tab=policies")
    return _p2d(p)

@governance_router.patch("/policies/{pid}")
def patch_policy(pid: str, body: dict, request: Request, db: Session=Depends(get_db)):
    p = db.query(GovernancePolicy).filter(GovernancePolicy.id==pid).first()
    if not p: raise HTTPException(404, "Not found")
    old = p.status
    for k, v in body.items():
        if hasattr(p, k): setattr(p, k, v)
    p.updated_at = str(date.today())
    if old != p.status:
        sev = "info" if p.status == "Active" else "warning"
        _audit(db, "Policy Activated" if p.status=="Active" else "Policy Deactivated", "Policy", p.name, f"{old}→{p.status}", sev, _usr(request), _ip(request))
        _notify_for_action(db, "policy", f"Policy {'Activated' if p.status=='Active' else 'Deactivated'}: {p.name}", f"Policy '{p.name}' status changed from {old} to {p.status}.", severity=sev, link="/settings?tab=policies")
    return _p2d(p)

@governance_router.delete("/policies/{pid}", status_code=204)
def delete_policy(pid: str, request: Request, db: Session=Depends(get_db)):
    p = db.query(GovernancePolicy).filter(GovernancePolicy.id==pid).first()
    if p:
        _notify_for_action(db, "policy", f"Policy Deleted: {p.name}", f"The policy '{p.name}' ({p.policy_type}) was permanently deleted.", severity="critical", link="/settings?tab=policies")
        _audit(db, "Policy Deleted", "Policy", p.name, "Deleted", "critical", _usr(request), _ip(request))
        db.delete(p); db.commit()

@governance_router.get("/classifications")
def get_classifications(): return _get_classifications()

@governance_router.get("/classifications/{category}/columns")
def get_classification_columns(category: str):
    cat = category.strip().capitalize()
    return [{"column_name":r[0]or"","data_type":r[1]or"","dataset_name":r[2]or"Unknown Dataset","null_count":r[3],"distinct_count":r[4],"health_score":r[5]} for r in _dedup_cols() if _classify(r[0]or"",r[1]or"")==cat]

@governance_router.get("/users")
def get_users(db: Session=Depends(get_db)): return [_u2d(u) for u in db.query(GovernanceUser).order_by(GovernanceUser.created_at).all()]

@governance_router.post("/users", status_code=201)
def create_user(body: dict, request: Request, db: Session=Depends(get_db)):
    if db.query(GovernanceUser).filter(GovernanceUser.email==body.get("email","")).first():
        raise HTTPException(409, "Email exists")
    u = GovernanceUser(
        id=f"u_{uuid.uuid4().hex[:8]}",
        name=body.get("name",""),
        email=body.get("email",""),
        role=body.get("role","Viewer"),
        status=body.get("status","Pending"),
        datasets_access=body.get("datasets_access",0),
        last_active="Never",
        created_at=str(date.today()),
    )
    db.add(u)
    _audit(db, "User Invited", "User", u.name, f"New {u.status} user", "info", _usr(request), _ip(request))
    _notify_for_action(db, "user", f"New User Invited: {u.name}", f"{u.name} ({u.email}) was invited with role '{u.role}'.", severity="info", link="/settings?tab=users")
    return _u2d(u)

@governance_router.put("/users/{uid}")
def update_user(uid: str, body: dict, request: Request, db: Session=Depends(get_db)):
    u = db.query(GovernanceUser).filter(GovernanceUser.id==uid).first()
    if not u: raise HTTPException(404, "Not found")
    old = u.role
    for f in ["name","email","role","status","datasets_access"]:
        if f in body: setattr(u, f, body[f])
    sev = "warning" if old != u.role else "info"
    _audit(db, "User Updated", "User", u.name, f"Role {old}→{u.role}" if old!=u.role else "Updated", sev, _usr(request), _ip(request))
    if old != u.role:
        _notify_for_action(db, "user", f"User Role Changed: {u.name}", f"{u.name}'s role was changed from '{old}' to '{u.role}'.", severity="warning", link="/settings?tab=users")
    return _u2d(u)

@governance_router.patch("/users/{uid}")
def patch_user(uid: str, body: dict, request: Request, db: Session=Depends(get_db)):
    u = db.query(GovernanceUser).filter(GovernanceUser.id==uid).first()
    if not u: raise HTTPException(404, "Not found")
    old = u.status
    for k, v in body.items():
        if hasattr(u, k): setattr(u, k, v)
    if old != u.status:
        _audit(db, f"User {'Activated' if u.status=='Active' else 'Deactivated'}", "User", u.name, f"{old}→{u.status}", "warning", _usr(request), _ip(request))
        _notify_for_action(db, "user", f"User {'Activated' if u.status=='Active' else 'Deactivated'}: {u.name}", f"{u.name} ({u.email}) was {'activated' if u.status=='Active' else 'deactivated'}.", severity="warning", link="/settings?tab=users")
    return _u2d(u)

@governance_router.post("/users/register", status_code=200)
def register_user(body: dict, request: Request, db: Session=Depends(get_db)):
    email = (body.get("email","")).lower().strip()
    if not email: raise HTTPException(400, "email required")
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    u = db.query(GovernanceUser).filter(GovernanceUser.email==email).first()
    if u:
        u.last_active = now
        if body.get("name"): u.name = body["name"]
        db.commit(); return _u2d(u)
    nu = GovernanceUser(
        id=f"u_{uuid.uuid4().hex[:8]}",
        email=email,
        name=body.get("name", email.split("@")[0].title()),
        role=body.get("role","Viewer"),
        status="Active",
        login_method=body.get("login_method","unknown"),
        last_active=now,
        created_at=str(date.today()),
        datasets_access=0,
    )
    db.add(nu)
    _audit(db, "User Signed In (First Time)", "User", nu.name, f"{email} auto-registered", "info", "System", _ip(request))
    return _u2d(nu)

@governance_router.get("/users/registered")
def get_registered(db: Session=Depends(get_db)):
    return [_u2d(u) for u in db.query(GovernanceUser).filter(GovernanceUser.last_active!="Never").all()]

# ─── Notifications CRUD ───────────────────────────────────────────────────────

@governance_router.get("/notifications")
def get_notifications(db: Session=Depends(get_db)):
    return [_n2d(n) for n in db.query(GovernanceNotification).all()]

@governance_router.put("/notifications")
def save_notifications(body: List[dict], request: Request, db: Session=Depends(get_db)):
    updated  = 0

    for item in body:
        title = item.get("title")
        if not title:
            continue

        n = db.query(GovernanceNotification).filter(
            GovernanceNotification.title == title
        ).first()
        if not n:
            continue

        n.enabled = bool(item.get("enabled", True))
        n.channel = item.get("channel", "in_app")
        
        # Handle slack_webhook if present in the table
        slack_webhook = item.get("slack_webhook", "").strip()
        if hasattr(n, 'slack_webhook'):
            n.slack_webhook = slack_webhook
        else:
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("UPDATE governance_notifications SET slack_webhook = :v WHERE id = :id"),
                        {"v": slack_webhook, "id": n.id},
                    )
                    conn.commit()
            except Exception:
                pass

        db.flush()

        _push_in_app(
            title=f"Preference Saved: {title}",
            message=f"Your notification preference for '{title}' has been saved. Channel: {n.channel}.",
            notif_type="ALERT", category="System", severity="info",
            link="/settings?tab=notifications",
        )

        updated += 1

    db.commit()
    _audit(db, "Notification Preferences Saved", "Notification", "Notification Preferences", f"{updated}/{len(body)} updated", "info", _usr(request), _ip(request))
    return {"status": "ok", "updated": updated}

@governance_router.patch("/notifications/{nid}")
def patch_notification(nid: str, body: dict, db: Session=Depends(get_db)):
    n = db.query(GovernanceNotification).filter(GovernanceNotification.id==nid).first()
    if not n: raise HTTPException(404, "Not found")
    for k, v in body.items():
        if hasattr(n, k): setattr(n, k, v)
    db.commit()
    if "slack_webhook" in body:
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("UPDATE governance_notifications SET slack_webhook = :v WHERE id = :id"),
                    {"v": body["slack_webhook"] or None, "id": nid},
                )
                conn.commit()
        except Exception: pass
    return _n2d(n)

# ─── System config ────────────────────────────────────────────────────────────

@governance_router.get("/system/config")
def get_system_config(db: Session=Depends(get_db)):
    return _cfg(db)

@governance_router.put("/system/config")
def save_system_config(body: dict, request: Request, db: Session=Depends(get_db)):
    changed = []
    for k, v in body.items():
        if isinstance(v, bool):
            sv = "true" if v else "false"
        else:
            sv = str(v) if v is not None else ""

        row = db.query(GovernanceSystemConfig).filter(GovernanceSystemConfig.key == k).first()
        if row:
            if row.value != sv:
                row.value = sv
                changed.append(k)
        else:
            db.add(GovernanceSystemConfig(key=k, value=sv))
            changed.append(k)

    db.commit()
    _audit(db, "System Config Saved", "System Config", "System Configuration", f"Updated: {', '.join(changed)}" if changed else "No changes", "warning", _usr(request), _ip(request))
    _push_in_app(
        title="System Configuration Updated",
        message=f"System configuration was updated. Changed fields: {', '.join(changed) if changed else 'none'}.",
        notif_type="ALERT", category="System", severity="warning",
        link="/settings?tab=system",
    )
    return _cfg(db)

@governance_router.get("/system/info")
def get_system_info(db: Session=Depends(get_db)): return _sys_info(db)

# ─── Policy suggestions ────────────────────────────────────────────────────────

@governance_router.get("/policy-suggestions")
def get_policy_suggestions(db: Session=Depends(get_db)):
    counts    = _anomaly_counts(db)
    dismissed = {r.id for r in db.query(GovernanceDismissedSuggestion).all()}
    out       = []
    for t in SUGGESTION_TEMPLATES:
        if t["id"] in dismissed or _has_policy(db, t["name"]): continue
        count = sum(v for k, v in counts.items() if any(ck in k for ck in t["check_types"]))
        if count < t["min_count"]: continue
        out.append({"id":t["id"],"name":t["name"],"description":t["description"],"policy_type":t["policy_type"],"priority":t["priority"],"reason":t["reason_template"],"triggered_by":f"{count} anomalies of this type detected"})
    return out

@governance_router.post("/policy-suggestions/{sid}/dismiss", status_code=204)
def dismiss_suggestion(sid: str, db: Session=Depends(get_db)):
    if not db.query(GovernanceDismissedSuggestion).filter(GovernanceDismissedSuggestion.id==sid).first():
        db.add(GovernanceDismissedSuggestion(id=sid)); db.commit()

@governance_router.post("/policy-suggestions/{sid}/adopt", status_code=204)
def adopt_suggestion(sid: str, request: Request, db: Session=Depends(get_db)):
    if not db.query(GovernanceDismissedSuggestion).filter(GovernanceDismissedSuggestion.id==sid).first():
        db.add(GovernanceDismissedSuggestion(id=sid))
    t = next((x for x in SUGGESTION_TEMPLATES if x["id"]==sid), None)
    if t: _audit(db, "Policy Suggestion Adopted", "Policy", t["name"], "Adopted", "info", _usr(request), _ip(request))

@governance_router.delete("/policy-suggestions/dismissed", status_code=204)
def reset_dismissed(db: Session=Depends(get_db)):
    db.query(GovernanceDismissedSuggestion).delete(); db.commit()

# ─── Audit log ────────────────────────────────────────────────────────────────

@governance_router.get("/audit-log")
def get_audit_log(severity: str=None, limit: int=100, db: Session=Depends(get_db)):
    q = db.query(GovernanceAuditLog).order_by(GovernanceAuditLog.timestamp.desc())
    if severity and severity != "all":
        q = q.filter(GovernanceAuditLog.severity==severity)
    return [{c.name: getattr(e,c.name) for c in e.__table__.columns} for e in q.limit(limit).all()]

@governance_router.post("/audit-log", status_code=201)
def create_audit_entry(body: dict, request: Request, db: Session=Depends(get_db)):
    return _audit(db, body.get("action","Unknown"), body.get("resource_type","System"), body.get("resource_name",""), body.get("change_summary",""), body.get("severity","info"), body.get("user", _usr(request)), body.get("ip_address", _ip(request)))