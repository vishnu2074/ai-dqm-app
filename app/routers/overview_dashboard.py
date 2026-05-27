"""
app/routers/overview_dashboard.py
Single aggregator endpoint for the Overview Dashboard.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal

router = APIRouter(tags=["overview"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[overview] {getattr(fn, '__name__', str(fn))} failed: {e}")
        return None

def _norm_kpi(raw):
    if not raw or raw.get("status") == "NO_DATA":
        return {"status":"NO_DATA","overallHealth":None,"overallTrend":"stable","totalColumns":None,"lastRunAt":None,"kpis":[]}
    return {**raw, "kpis": raw.get("kpis", []), "overallHealth": raw.get("overallHealth"), "overallTrend": raw.get("overallTrend","stable"), "totalColumns": raw.get("totalColumns"), "lastRunAt": raw.get("lastRunAt")}

def _norm_list(raw, key=None):
    if not raw: return []
    if isinstance(raw, list): return raw
    if key: return raw.get(key, [])
    for k in ("anomalies","runs","trend","points","contributors","risks","checks","results"):
        if k in raw: return raw[k]
    return []

def _norm_velocity(raw):
    if not raw: return {"velocity":None,"direction":"stable","momentum":"neutral"}
    return {**raw, "velocity": raw.get("velocity"), "direction": raw.get("direction","stable"), "momentum": raw.get("momentum","neutral")}

def _norm_coverage(raw):
    if not raw or raw.get("status")=="NO_DATA":
        return {"coverageScore":None,"totalActiveRules":0,"coveredColumns":0,"totalColumns":0,"uncoveredColumns":[]}
    return {**raw}

def _norm_freshness(raw):
    if not raw or raw.get("status")=="NO_DATA":
        return {"freshnessScore":None,"status":"NO_DATA","staleColumns":[],"dateColumns":0}
    return {**raw}

def _norm_forecast(raw):
    if not raw: return {"riskScore":None,"riskLevel":None,"criticalIssues":0,"warningIssues":0}
    return {**raw, "riskScore": raw.get("riskScore"), "riskLevel": raw.get("riskLevel"), "criticalIssues": raw.get("criticalIssues",0), "warningIssues": raw.get("warningIssues",0)}

def _norm_profiling(raw):
    if not raw: return {"totalRows":None,"totalColumns":None,"columns":[]}
    return {**raw, "totalRows": raw.get("totalRows"), "totalColumns": raw.get("totalColumns"), "columns": raw.get("columns",[])}

def _norm_mon(raw):
    if not raw or raw.get("status")=="NO_DATA":
        return {"driftScore":None,"successRate":None,"totalExecutionsToday":None,"avgQualityScore":None}
    return {**raw}


@router.get("/overview/{dataset_id}")
def get_overview(dataset_id: int, days: int = 20, db: Session = Depends(get_db)):
    from app.services.scorecards import (
        get_kpi_summary, get_quality_trend, get_risk_contributors,
        get_quality_velocity, get_rules_coverage, get_freshness_score,
    )
    from app.services.monitoring import get_risk_forecast, get_monitoring_summary
    from app.services.profiling_detail import get_detail_profile
    from app.services.dq_scores import get_incremental_runs, get_quality_checks
    from app.services.anomalies import get_anomalies

    kpi_raw   = _safe(get_kpi_summary,       db, dataset_id)
    trend_raw = _safe(get_quality_trend,      db, dataset_id, days)
    runs_raw  = _safe(get_incremental_runs,   db, dataset_id, 100)
    risks_raw = _safe(get_risk_contributors,  db, dataset_id, 24)
    an_raw    = _safe(get_anomalies,          db, dataset_id)
    vel_raw   = _safe(get_quality_velocity,   db, dataset_id)
    cov_raw   = _safe(get_rules_coverage,     db, dataset_id)
    fsh_raw   = _safe(get_freshness_score,    db, dataset_id)
    fct_raw   = _safe(get_risk_forecast,      db, dataset_id)
    pro_raw   = _safe(get_detail_profile,     db, dataset_id)
    mon_raw   = _safe(get_monitoring_summary, db, dataset_id)
    qc_raw    = _safe(get_quality_checks,     db, dataset_id)

    return {
        "dataset_id":    dataset_id,
        "kpi":           _norm_kpi(kpi_raw),
        "trend":         _norm_list(trend_raw),
        "runs":          _norm_list(runs_raw, "runs"),
        "risks":         _norm_list(risks_raw),
        "anomalies":     _norm_list(an_raw, "anomalies"),
        "velocity":      _norm_velocity(vel_raw),
        "coverage":      _norm_coverage(cov_raw),
        "freshness":     _norm_freshness(fsh_raw),
        "forecast":      _norm_forecast(fct_raw),
        "profiling":     _norm_profiling(pro_raw),
        "monSum":        _norm_mon(mon_raw),
        "qualityChecks": _norm_list(qc_raw),
    }


@router.get("/overview-all-health")
def get_all_datasets_health(db: Session = Depends(get_db)):
    from app.models import Dataset, DataSource
    from app.services.scorecards import get_kpi_summary
    from app.services.anomalies import get_anomalies

    datasets    = db.query(Dataset).all()
    sources_map = {s.id: s.name for s in db.query(DataSource).all()}
    results = []
    for ds in datasets:
        kpi = _safe(get_kpi_summary, db, ds.id)
        an  = _safe(get_anomalies,   db, ds.id)
        an_list = _norm_list(an, "anomalies")
        open_count = len([a for a in an_list if (a.get("status") or "").lower() not in ("resolved","ignored")])
        results.append({
            "id":      ds.id,
            "name":    ds.display_name or ds.physical_name or f"Dataset {ds.id}",
            "health":  kpi.get("overallHealth") if kpi else None,
            "lastRun": kpi.get("lastRunAt")     if kpi else None,
            "open":    open_count,
            "source":  sources_map.get(ds.datasource_id, "—"),
        })
    return {"datasets": results}
