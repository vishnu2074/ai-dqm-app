"""
AI DQM — Monitoring Service
Includes: summary, metrics-trends, drift, risk-forecast,
          column-health, sla-status, compare-runs, execution-runs, manual-check
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.models import (
    ColumnProfile,
    DriftRecord,
    ProfilingRun,
    QualityCheck,
)
from app.services.dq_scores import run_dq_scoring


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))

def _r(v: float, d: int = 2) -> float:
    return round(float(v), d)

# ─────────────────────────────────────────────────────────────
# EWMA helpers
# ─────────────────────────────────────────────────────────────

def _ewma(values: List[float], span: int = 5) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    result = [float(values[0])]
    for v in values[1:]:
        result.append(alpha * float(v) + (1 - alpha) * result[-1])
    return result

def _ewma_slope(smoothed: List[float], lookback: int = 4) -> float:
    if len(smoothed) < 2:
        return 0.0
    window = smoothed[-lookback:] if len(smoothed) >= lookback else smoothed
    xs = np.arange(len(window), dtype=float)
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(xs, window, 1)[0])

def _ewma_acceleration(smoothed: List[float], lookback: int = 4) -> float:
    if len(smoothed) < 3:
        return 0.0
    window = smoothed[-lookback:] if len(smoothed) >= lookback else smoothed
    if len(window) < 3:
        return 0.0
    slopes = [window[i+1] - window[i] for i in range(len(window) - 1)]
    return float(np.mean(np.diff(slopes))) if len(slopes) >= 2 else 0.0

def _uncertainty_band(base_std: float, step: int, method: str) -> float:
    growth = {"LINEAR": 0.40, "COMPOSITE": 0.32, "ML": 0.22}.get(method, 0.35)
    return base_std * (1.0 + step * growth)

# ─────────────────────────────────────────────────────────────
# Per-run DB queries
# ─────────────────────────────────────────────────────────────

def _risk_for_run(db: Session, run_id: int) -> float:
    critical = db.query(QualityCheck).filter(
        QualityCheck.profiling_run_id == run_id,
        QualityCheck.severity == "CRITICAL",
        QualityCheck.status != "resolved",
    ).count()
    warning  = db.query(QualityCheck).filter(
        QualityCheck.profiling_run_id == run_id,
        QualityCheck.severity.in_(["MEDIUM", "HIGH"]),
        QualityCheck.status != "resolved",
    ).count()
    return float(min(100, critical * 15 + warning * 5))

def _avg_metric(db: Session, run_id: int, attr: str) -> Optional[float]:
    val = db.query(func.avg(getattr(ColumnProfile, attr))).filter(ColumnProfile.profiling_run_id == run_id).scalar()
    return float(val) if val is not None else None

def _avg_drift(db: Session, run_id: int) -> float:
    val = db.query(func.avg(DriftRecord.drift_score)).filter(DriftRecord.profiling_run_id == run_id).scalar()
    return float(val) if val is not None else 0.0

# ─────────────────────────────────────────────────────────────
# History builder
# ─────────────────────────────────────────────────────────────

_DIMENSIONS = ["completeness", "validity", "uniqueness", "consistency", "accuracy", "integrity"]
_METRIC_WEIGHTS: Dict[str, float] = {
    "completeness": 0.28, "validity": 0.24, "drift_score": 0.22,
    "uniqueness": 0.14, "consistency": 0.12,
}

def _build_history(db: Session, runs: List[ProfilingRun]) -> List[Dict[str, Any]]:
    rows = []
    for run in reversed(runs):
        ts = run.timestamp
        row: Dict[str, Any] = {
            "run_id":      run.id,
            "date":        f"Run #{run.id}",
            "full_date":   ts.strftime("%d/%m/%Y %H:%M") if ts else None,
            "risk":        _risk_for_run(db, run.id),
            "drift_score": _avg_drift(db, run.id),
        }
        for dim in _DIMENSIONS:
            row[dim] = _avg_metric(db, run.id, dim) or 100.0
        rows.append(row)
    return rows

def _composite_risk(metric_vals: Dict[str, float]) -> float:
    risk = 0.0
    for metric, weight in _METRIC_WEIGHTS.items():
        val = metric_vals.get(metric, 100.0)
        risk += val * weight if metric == "drift_score" else (100.0 - val) * weight
    return _clamp(risk)

# ─────────────────────────────────────────────────────────────
# Risk Forecast — Chart builders
# ─────────────────────────────────────────────────────────────

def _chart1_composite_trajectory(history, method, steps):
    n    = len(history)
    span = min(max(3, n // 2), 10)
    metric_smoothed = {m: _ewma([h[m] for h in history], span=span) for m in _METRIC_WEIGHTS}
    smoothed_risk   = [_composite_risk({m: metric_smoothed[m][i] for m in _METRIC_WEIGHTS}) for i in range(n)]

    anomaly_contrib = [0.0] * n
    if method == "ML" and n >= 10:
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import MinMaxScaler
            feat_cols = _DIMENSIONS + ["drift_score", "risk"]
            X_raw = np.array([[h[c] for c in feat_cols] for h in history], dtype=float)
            X     = MinMaxScaler().fit_transform(X_raw)
            contamination = min(0.15, max(0.05, 5 / n))
            clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, max_samples=min(n, 64))
            clf.fit(X)
            raw_scores = clf.score_samples(X)
            s_min, s_max = raw_scores.min(), raw_scores.max()
            if s_max > s_min:
                anomaly_contrib = list(((s_max - raw_scores) / (s_max - s_min)) * 100.0)
        except ImportError:
            pass

    blended = ([0.6 * smoothed_risk[i] + 0.4 * anomaly_contrib[i] for i in range(n)]
               if method == "ML" else smoothed_risk)
    blended_smooth = _ewma(blended, span=5) if method == "ML" else smoothed_risk

    raw_risk  = [h["risk"] for h in history]
    residuals = [abs(raw_risk[i] - blended_smooth[i]) for i in range(n)]
    base_std  = max(float(np.std(residuals)) if len(residuals) > 1 else 2.0, 1.5)

    historical = [{"date": h["date"], "fullDate": h["full_date"], "actual": _r(h["risk"]), "smoothed": _r(blended_smooth[i])} for i, h in enumerate(history)]

    metric_slopes = {m: _ewma_slope(metric_smoothed[m], lookback=min(5, n)) for m in _METRIC_WEIGHTS}
    last_anomaly  = anomaly_contrib[-1] if method == "ML" else 0.0

    forecast = []
    for step in range(1, steps + 1):
        proj     = {m: _clamp(metric_smoothed[m][-1] + metric_slopes[m] * step) for m in _METRIC_WEIGHTS}
        pred_c   = _composite_risk(proj)
        pred     = _clamp(0.6 * pred_c + 0.4 * last_anomaly) if method == "ML" else pred_c
        band     = _uncertainty_band(base_std, step, method)
        forecast.append({"date": f"Pred +{step}", "predicted": _r(pred), "upper": _r(_clamp(pred + band)), "lower": _r(_clamp(pred - band))})

    current    = raw_risk[-1]
    final_pred = forecast[-1]["predicted"]
    slope_risk = _ewma_slope(blended_smooth, lookback=min(4, n))
    worst_dim  = max(_METRIC_WEIGHTS, key=lambda m: metric_slopes[m] if m == "drift_score" else -metric_slopes[m])

    alert = None
    if final_pred > current + 15:
        alert = f"Risk is forecast to rise sharply to **{final_pred}** in {steps} runs (currently {_r(current)}). Strongest adverse driver: **{worst_dim.replace('_', ' ').title()}**."
    elif slope_risk > 2.0:
        alert = f"Composite risk trending **upward** ({slope_risk:+.1f} per run). Projected to reach **{final_pred}** in {steps} runs."
    elif current >= 70:
        alert = f"Current risk is **HIGH ({_r(current)})**. Forecast confirms elevated risk for the next {steps} runs."

    return {
        "historical": historical, "forecast": forecast, "alertMessage": alert,
        "currentRisk": _r(current), "finalForecastRisk": final_pred,
        "trend": "up" if slope_risk > 0.5 else "down" if slope_risk < -0.5 else "stable",
    }


def _chart2_dimension_forecast(history, method, steps):
    n    = len(history)
    span = min(max(3, n // 2), 10)
    smoothed = {dim: _ewma([h[dim] for h in history], span=span) for dim in _DIMENSIONS}
    slopes   = {dim: _ewma_slope(smoothed[dim], lookback=min(5, n)) for dim in _DIMENSIONS}
    stds     = {dim: max(float(np.std([abs(history[i][dim] - smoothed[dim][i]) for i in range(n)])) if n > 1 else 1.5, 1.0) for dim in _DIMENSIONS}

    historical = [{"date": h["date"], "fullDate": h["full_date"], **{dim: _r(smoothed[dim][i]) for dim in _DIMENSIONS}} for i, h in enumerate(history)]
    forecast   = []
    for step in range(1, steps + 1):
        point: Dict[str, Any] = {"date": f"Pred +{step}"}
        for dim in _DIMENSIONS:
            pred = _clamp(smoothed[dim][-1] + slopes[dim] * step)
            band = _uncertainty_band(stds[dim], step, method)
            point[dim]             = _r(pred)
            point[f"{dim}_upper"]  = _r(_clamp(pred + band))
            point[f"{dim}_lower"]  = _r(_clamp(pred - band))
        forecast.append(point)

    final = forecast[-1]
    dim_changes = {dim: final[dim] - smoothed[dim][-1] for dim in _DIMENSIONS}
    ranked      = sorted(_DIMENSIONS, key=lambda d: dim_changes[d])

    return {
        "historical": historical, "forecast": forecast,
        "slopes": {dim: _r(slopes[dim], 3) for dim in _DIMENSIONS},
        "worstDimension": ranked[0] if ranked else None,
        "dimensionChanges": {dim: _r(dim_changes[dim]) for dim in _DIMENSIONS},
    }


def _chart3_drift_velocity(history, method, steps):
    n          = len(history)
    drift_vals = [h["drift_score"] for h in history]
    span       = min(max(3, n // 2), 8)
    smoothed   = _ewma(drift_vals, span=span)
    slope      = _ewma_slope(smoothed, lookback=min(4, n))
    accel      = _ewma_acceleration(smoothed, lookback=min(5, n))
    velocity   = [0.0] + [smoothed[i] - smoothed[i-1] for i in range(1, n)]
    base_std   = max(float(np.std(drift_vals)) if len(drift_vals) > 1 else 0.01, 0.005)

    historical = [{"date": h["date"], "fullDate": h["full_date"], "drift": _r(drift_vals[i], 4), "smoothed": _r(smoothed[i], 4), "velocity": _r(velocity[i], 4)} for i, h in enumerate(history)]
    forecast   = []
    last_vel   = velocity[-1]
    for step in range(1, steps + 1):
        pred  = _clamp(smoothed[-1] + slope * step, 0.0, 1.0)
        band  = _uncertainty_band(base_std, step, method)
        forecast.append({"date": f"Pred +{step}", "drift": _r(pred, 4), "upper": _r(_clamp(pred + band, 0, 1), 4), "lower": _r(_clamp(pred - band, 0, 1), 4), "velocity": _r(last_vel + accel * step, 4)})

    return {
        "historical": historical, "forecast": forecast,
        "currentSlope": _r(slope, 4), "acceleration": _r(accel, 4),
        "driftTrend": "accelerating" if accel > 0.001 else "decelerating" if accel < -0.001 else "stable",
    }


def _run_adaptive_forecast(db: Session, runs: List[ProfilingRun]) -> Dict[str, Any]:
    n = len(runs)
    if n < 3:
        history = _build_history(db, runs)
        return {
            "forecastMethod": "INSUFFICIENT", "forecastSteps": 0,
            "availableCharts": [], "chart1": None, "chart2": None, "chart3": None,
            "alertMessage": "Not enough data to generate a forecast. Run DQ scoring at least **3 times** to enable trend prediction.",
        }

    history = _build_history(db, runs)
    if n < 10:   method, steps = "LINEAR",    5
    elif n < 30: method, steps = "COMPOSITE", 8
    else:        method, steps = "ML",        10

    available = ["chart1"]
    if n >= 5: available.append("chart2")
    if n >= 4: available.append("chart3")

    chart1 = _chart1_composite_trajectory(history, method, steps)
    chart2 = _chart2_dimension_forecast(history, method, steps) if "chart2" in available else None
    chart3 = _chart3_drift_velocity(history, method, steps)     if "chart3" in available else None

    return {
        "forecastMethod": method, "forecastSteps": steps, "runCount": n,
        "availableCharts": available,
        "chart1": chart1, "chart2": chart2, "chart3": chart3,
        "alertMessage": chart1.get("alertMessage"),
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC: get_risk_forecast
# ─────────────────────────────────────────────────────────────

def get_risk_forecast(db: Session, dataset_id: int) -> Dict[str, Any]:
    runs = (db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED").order_by(desc(ProfilingRun.id)).limit(60).all())
    if not runs:
        return {"riskScore": 0, "riskLevel": "UNKNOWN", "criticalIssues": 0, "warningIssues": 0,
                "forecastMethod": "INSUFFICIENT", "forecastSteps": 0, "availableCharts": [],
                "chart1": None, "chart2": None, "chart3": None, "alertMessage": None}

    latest   = runs[0]
    critical = db.query(QualityCheck).filter(
        QualityCheck.profiling_run_id == latest.id,
        QualityCheck.severity == "CRITICAL",
        QualityCheck.status != "resolved",
    ).count()
    warning  = db.query(QualityCheck).filter(
        QualityCheck.profiling_run_id == latest.id,
        QualityCheck.severity.in_(["MEDIUM", "HIGH"]),
        QualityCheck.status != "resolved",
    ).count()
    risk_score = min(100, critical * 15 + warning * 5)
    risk_level = "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW"
    forecast   = _run_adaptive_forecast(db, runs)
    return {"riskScore": risk_score, "riskLevel": risk_level, "criticalIssues": critical, "warningIssues": warning, **forecast}


# ─────────────────────────────────────────────────────────────
# Column Health Heatmap
# ─────────────────────────────────────────────────────────────

def get_column_health(db: Session, dataset_id: int, limit: int = 20) -> Dict[str, Any]:
    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(desc(ProfilingRun.id))
        .limit(limit)
        .all()
    )
    if not runs:
        return {"runs": [], "rows": []}

    runs = list(reversed(runs))
    run_labels = [f"#{r.id}" for r in runs]

    grid: Dict[str, Dict[int, float]] = defaultdict(dict)
    for run in runs:
        profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        for p in profiles:
            if p.health_score is not None:
                grid[p.column_name][run.id] = round(float(p.health_score), 1)

    if not grid:
        return {"runs": run_labels, "rows": []}

    rows = []
    for col in sorted(grid.keys()):
        cells = []
        scores_for_trend = []
        for run in runs:
            score = grid[col].get(run.id, None)
            cells.append({"runId": run.id, "runLabel": f"#{run.id}", "score": score})
            if score is not None:
                scores_for_trend.append(score)

        valid = [s for s in scores_for_trend if s is not None]
        avg   = round(sum(valid) / len(valid), 1) if valid else 0.0

        trend = "stable"
        if len(valid) >= 6:
            first3 = sum(valid[:3]) / 3
            last3  = sum(valid[-3:]) / 3
            if   last3 > first3 + 2: trend = "up"
            elif last3 < first3 - 2: trend = "down"

        rows.append({"column": col, "cells": cells, "avgScore": avg, "trend": trend})

    rows.sort(key=lambda r: r["avgScore"])
    return {"runs": run_labels, "rows": rows}


# ─────────────────────────────────────────────────────────────
# SLA & Threshold Breach Tracker
# ─────────────────────────────────────────────────────────────

# Fallback hardcoded defaults — used only when governance_system_config has no values.
# The router reads admin-configured values from the DB and passes them as `thresholds`.
_SLA_THRESHOLDS: Dict[str, float] = {
    "completeness": 80.0,
    "validity":     80.0,
    "uniqueness":   50.0,
    "consistency":  80.0,
    "accuracy":     80.0,
    "integrity":    80.0,
}
_SLA_WARNING_BUFFER = 5.0


def get_sla_status(db: Session, dataset_id: int, limit: int = 30,
                   thresholds: dict = None) -> Dict[str, Any]:
    """
    For each quality dimension, returns:
    - current value vs threshold
    - breach/warning/ok status
    - consecutive breach run count
    - sparkline history (last `limit` runs)

    thresholds dict (passed from the monitoring router, read from governance_system_config):
        completeness, uniqueness, validity, consistency, accuracy, integrity → int 0-100
        warning_offset → int (score must be >= threshold + offset to be fully OK)

    If thresholds is None, falls back to _SLA_THRESHOLDS defaults.
    """
    # ── Resolve thresholds — DB values preferred, hardcoded as fallback ───────
    t = thresholds or {}
    dim_thresholds: Dict[str, float] = {
        "completeness": float(t.get("completeness", _SLA_THRESHOLDS["completeness"])),
        "uniqueness":   float(t.get("uniqueness",   _SLA_THRESHOLDS["uniqueness"])),
        "validity":     float(t.get("validity",     _SLA_THRESHOLDS["validity"])),
        "consistency":  float(t.get("consistency",  _SLA_THRESHOLDS["consistency"])),
        "accuracy":     float(t.get("accuracy",     _SLA_THRESHOLDS["accuracy"])),
        "integrity":    float(t.get("integrity",    _SLA_THRESHOLDS["integrity"])),
    }
    warning_buffer = float(t.get("warning_offset", _SLA_WARNING_BUFFER))

    runs = (
        db.query(ProfilingRun)
        .filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED")
        .order_by(desc(ProfilingRun.id))
        .limit(limit)
        .all()
    )
    if not runs:
        return {"dimensions": [], "totalBreaches": 0, "longestBreach": 0}

    runs = list(reversed(runs))  # oldest first for sparkline

    # Build per-run, per-dimension avg values
    dim_history: Dict[str, List[Dict]] = {dim: [] for dim in _DIMENSIONS}

    for run in runs:
        profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        label = f"Run #{run.id}"
        for dim in _DIMENSIONS:
            vals = [getattr(p, dim) for p in profiles if getattr(p, dim) is not None]
            val  = round(sum(vals) / len(vals), 2) if vals else None
            threshold = dim_thresholds.get(dim, 80.0)
            dim_history[dim].append({
                "runId":    run.id,
                "runLabel": label,
                "value":    val if val is not None else 100.0,
                "breached": (val is not None and val < threshold),
            })

    dimensions_out = []
    total_breaches = 0
    longest_breach = 0

    for dim in _DIMENSIONS:
        hist      = dim_history[dim]
        threshold = dim_thresholds.get(dim, 80.0)
        current   = hist[-1]["value"] if hist else 100.0

        # Consecutive breach streak counting back from latest run
        streak = 0
        for h in reversed(hist):
            if h["breached"]:
                streak += 1
            else:
                break

        # Longest breach streak across all runs
        max_streak, cur_s = 0, 0
        for h in hist:
            if h["breached"]:
                cur_s += 1
                max_streak = max(max_streak, cur_s)
            else:
                cur_s = 0

        if streak > 0:
            total_breaches += 1
        longest_breach = max(longest_breach, max_streak)

        # Status: breach < threshold, warning < threshold+offset, ok otherwise
        if current < threshold:
            status = "breach"
        elif current < threshold + warning_buffer:
            status = "warning"
        else:
            status = "ok"

        last_breach_id = None
        for h in reversed(hist):
            if h["breached"]:
                last_breach_id = h["runId"]
                break

        dimensions_out.append({
            "dimension":       dim,
            "threshold":       threshold,
            "currentValue":    round(current, 2),
            "status":          status,
            "breachDuration":  streak,
            "lastBreachRunId": last_breach_id,
            "history":         hist,
        })

    return {
        "dimensions":    dimensions_out,
        "totalBreaches": total_breaches,
        "longestBreach": longest_breach,
    }


# ─────────────────────────────────────────────────────────────
# Comparative Run Analysis
# ─────────────────────────────────────────────────────────────

def get_compare_runs(db: Session, dataset_id: int, run_a_id: int, run_b_id: int) -> Dict[str, Any]:
    def _get_run(rid: int) -> Optional[ProfilingRun]:
        return db.query(ProfilingRun).filter(ProfilingRun.id == rid, ProfilingRun.dataset_id == dataset_id).first()

    run_a = _get_run(run_a_id)
    run_b = _get_run(run_b_id)

    if not run_a or not run_b:
        raise ValueError("One or both run IDs not found for this dataset")

    def _metrics_for(run: ProfilingRun) -> Dict[str, Optional[float]]:
        profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        m: Dict[str, Optional[float]] = {}
        for dim in _DIMENSIONS:
            vals = [getattr(p, dim) for p in profiles if getattr(p, dim) is not None]
            m[dim] = round(sum(vals) / len(vals), 2) if vals else None
        m["riskScore"]  = _risk_for_run(db, run.id)
        m["driftScore"] = round(_avg_drift(db, run.id), 4)
        m["rowCount"]   = float(run.rows_processed or 0)
        m["colCount"]   = float(len(profiles))
        return m

    ma = _metrics_for(run_a)
    mb = _metrics_for(run_b)

    higher_better = set(_DIMENSIONS)
    lower_better  = {"riskScore", "driftScore"}

    diffs = []
    improved = degraded = unchanged = 0

    for metric in list(_DIMENSIONS) + ["riskScore", "driftScore", "rowCount", "colCount"]:
        va = ma.get(metric)
        vb = mb.get(metric)
        delta = round(float(vb) - float(va), 4) if (va is not None and vb is not None) else None

        if delta is None or abs(delta) < 0.01:
            direction = "neutral"; unchanged += 1
        elif metric in higher_better:
            direction = "better" if delta > 0 else "worse"
            if direction == "better": improved += 1
            else:                     degraded += 1
        elif metric in lower_better:
            direction = "better" if delta < 0 else "worse"
            if direction == "better": improved += 1
            else:                     degraded += 1
        else:
            direction = "neutral"; unchanged += 1

        diffs.append({"metric": metric, "runA": va, "runB": vb, "delta": delta, "direction": direction})

    ts_a = run_a.timestamp.strftime("%d/%m/%Y, %H:%M:%S") if run_a.timestamp else "—"
    ts_b = run_b.timestamp.strftime("%d/%m/%Y, %H:%M:%S") if run_b.timestamp else "—"

    return {
        "runA": {"id": run_a.id, "label": f"#{run_a.id}", "timestamp": ts_a, "metrics": ma},
        "runB": {"id": run_b.id, "label": f"#{run_b.id}", "timestamp": ts_b, "metrics": mb},
        "diffs":   diffs,
        "summary": {"improved": improved, "degraded": degraded, "unchanged": unchanged},
    }


# ─────────────────────────────────────────────────────────────
# Existing service functions (unchanged)
# ─────────────────────────────────────────────────────────────

def get_monitoring_summary(db: Session, dataset_id: int) -> Dict[str, Any]:
    latest = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED").order_by(desc(ProfilingRun.id)).first()
    if not latest:
        return {"status": "NO_DATA", "avgQualityScore": None, "avgQualityScoreDelta": 0,
                "driftScore": 0, "driftScoreDelta": 0, "totalExecutionsToday": 0,
                "successRate": None, "successRateDelta": 0, "lastRunAt": None}

    total_runs      = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id).count()
    successful_runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED").count()
    success_rate    = round((successful_runs / total_runs) * 100, 1) if total_runs else None
    avg_quality     = db.query(func.avg(ColumnProfile.health_score)).filter(ColumnProfile.profiling_run_id == latest.id).scalar()
    avg_drift_raw   = db.query(func.avg(DriftRecord.drift_score)).filter(DriftRecord.profiling_run_id == latest.id).scalar()
    drift_score     = round((avg_drift_raw or 0) / 100, 4)

    quality_delta = drift_delta = 0
    prev_run = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED", ProfilingRun.id < latest.id).order_by(desc(ProfilingRun.id)).first()
    if prev_run:
        prev_quality = db.query(func.avg(ColumnProfile.health_score)).filter(ColumnProfile.profiling_run_id == prev_run.id).scalar()
        if avg_quality is not None and prev_quality is not None:
            quality_delta = round(float(avg_quality) - float(prev_quality), 1)
        prev_drift_raw = db.query(func.avg(DriftRecord.drift_score)).filter(DriftRecord.profiling_run_id == prev_run.id).scalar()
        drift_delta = round(drift_score - round((prev_drift_raw or 0) / 100, 4), 4)

    return {
        "status": "OK",
        "avgQualityScore":      round(float(avg_quality), 1) if avg_quality is not None else None,
        "avgQualityScoreDelta": quality_delta,
        "driftScore":           drift_score,
        "driftScoreDelta":      drift_delta,
        "totalExecutionsToday": total_runs,
        "successRate":          success_rate,
        "successRateDelta":     0,
        "lastRunAt":            _iso(latest.timestamp),
    }


def get_metrics_trends(db: Session, dataset_id: int, limit: int = 30) -> Dict[str, Any]:
    runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED").order_by(desc(ProfilingRun.id)).limit(limit).all()
    results = []
    for run in reversed(runs):
        profiles = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == run.id).all()
        if not profiles:
            continue
        def avg(attr, _p=profiles):
            vals = [getattr(p, attr) for p in _p if getattr(p, attr) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        ts = run.timestamp
        results.append({"runId": run.id, "date": f"Run #{run.id}", "fullDate": ts.strftime("%d/%m/%Y %H:%M") if ts else None,
                         "timestamp": _iso(ts), "completeness": avg("completeness"), "validity": avg("validity"),
                         "uniqueness": avg("uniqueness"), "consistency": avg("consistency"),
                         "accuracy": avg("accuracy"), "integrity": avg("integrity"), "timeliness": avg("timeliness")})
    return {"trends": results}


def get_drift_monitoring(db: Session, dataset_id: int, limit: int = 30) -> Dict[str, Any]:
    runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id, ProfilingRun.status == "COMPLETED").order_by(desc(ProfilingRun.id)).limit(limit).all()
    time_series: List[Dict] = []
    column_max: Dict[str, float] = {}
    column_ts:  Dict[str, datetime] = {}
    for run in reversed(runs):
        records  = db.query(DriftRecord).filter(DriftRecord.profiling_run_id == run.id).all()
        ts       = run.timestamp
        date_lbl = f"Run #{run.id}"
        full_d   = ts.strftime("%d/%m/%Y %H:%M") if ts else None
        if not records:
            time_series.append({"date": date_lbl, "fullDate": full_d, "driftScore": 0.0, "topContributor": None})
            continue
        avg_d   = sum(r.drift_score for r in records) / (100 * len(records))
        top_rec = max(records, key=lambda r: r.drift_score)
        top_col = top_rec.column_name if top_rec.drift_score > 0 else None
        time_series.append({"date": date_lbl, "fullDate": full_d, "driftScore": round(avg_d, 4), "topContributor": top_col})
        for r in records:
            if r.drift_score > column_max.get(r.column_name, 0):
                column_max[r.column_name] = r.drift_score
                column_ts[r.column_name]  = ts
    alerts = []
    for col, score in sorted(column_max.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct  = round(score, 1)
        sev  = "Critical" if pct >= 70 else "High" if pct >= 40 else "Medium" if pct >= 15 else "Low"
        c_ts = column_ts.get(col)
        alerts.append({"column": col, "driftScore": round(score/100, 4), "type": "SIGNIFICANT" if pct >= 30 else "MINOR",
                        "severity": sev, "timestamp": c_ts.strftime("%d/%m/%Y %H:%M") if c_ts else "—", "drift": pct})
    return {"driftTimeSeries": time_series, "driftAlerts": alerts}


def get_execution_runs(db: Session, dataset_id: int, limit: int = 50) -> Dict[str, Any]:
    runs = db.query(ProfilingRun).filter(ProfilingRun.dataset_id == dataset_id).order_by(desc(ProfilingRun.id)).limit(limit).all()
    result = []
    for r in runs:
        ts = r.timestamp
        total_checks = critical_checks = cols_count = 0
        if r.status == "COMPLETED":
            total_checks    = db.query(QualityCheck).filter(QualityCheck.profiling_run_id == r.id).count()
            critical_checks = db.query(QualityCheck).filter(QualityCheck.profiling_run_id == r.id, QualityCheck.severity == "CRITICAL").count()
            cols_count      = db.query(ColumnProfile).filter(ColumnProfile.profiling_run_id == r.id).count()
        status_display = ("Completed with Warnings" if critical_checks > 0 else "Completed") if r.status == "COMPLETED" else ("Failed" if r.status == "FAILED" else r.status.capitalize())
        dur = "—"
        if r.duration_ms:
            dur = f"{r.duration_ms}ms" if r.duration_ms < 1000 else f"{round(r.duration_ms/1000,1)}s"
        result.append({"id": f"#{r.id}", "runId": r.id, "timestamp": ts.strftime("%d/%m/%Y, %H:%M:%S") if ts else "—",
                        "type": "Full Scan" if r.is_full_scan else "Incremental",
                        "rulesExecuted": cols_count, "rowsProcessed": r.rows_processed or 0,
                        "deltaRows": r.delta_rows or 0, "passed": max(0, cols_count - critical_checks),
                        "failed": critical_checks, "duration": dur, "status": status_display,
                        "errorMessage": r.error_message, "qualityAlerts": total_checks, "totalAlerts": total_checks})
    return {"runs": result}


def trigger_manual_check(db: Session, dataset_id: int) -> Dict[str, Any]:
    run = run_dq_scoring(db, dataset_id)
    return {"status": "STARTED", "runId": run.id}