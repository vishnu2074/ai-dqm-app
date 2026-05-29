"""
AI DQM Health Metrics Router v4
Correct schema, direct sqlite3 only (no SQLAlchemy parameter issues),
full error logging per query.
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/health-metrics", tags=["health-metrics"])

# ── DB path resolution ────────────────────────────────────────────────────────

def _get_db_file() -> str:
    """Get the DB file path from SQLAlchemy engine URL."""
    try:
        from app.database import engine
        url = str(engine.url)
        # sqlite:////tmp/ai-dqm/ai_dqm.db  →  /tmp/ai-dqm/ai_dqm.db
        if url.startswith("sqlite:////"):
            return url[len("sqlite:///"):]          # keep leading /
        if url.startswith("sqlite:///"):
            p = url[len("sqlite:///"):]             # relative path
            if os.path.exists(p):
                return os.path.abspath(p)
            # try from cwd
            cwd = os.path.join(os.getcwd(), p)
            if os.path.exists(cwd):
                return cwd
    except Exception as e:
        print(f"[hm] engine url error: {e}")
    return os.getenv("AIDQM_DB_PATH", "")


def _con() -> sqlite3.Connection:
    db = _get_db_file()
    if not db or not os.path.exists(db):
        raise RuntimeError(f"DB not found: {db!r}")
    con = sqlite3.connect(db, timeout=15, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _q(sql: str, params: tuple = ()) -> List[Dict]:
    """Execute SQL and return list of dicts. Logs errors."""
    try:
        con = _con()
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
        con.close()
        return rows
    except Exception as e:
        print(f"[hm] QUERY ERROR: {e!r} | SQL: {sql[:120]}")
        return []


def _scalar(sql: str, params: tuple = (), default=0):
    rows = _q(sql, params)
    if not rows:
        return default
    v = list(rows[0].values())
    return v[0] if v else default


def _safe(fn, fallback):
    try:
        return fn()
    except Exception as e:
        print(f"[hm] TAB ERROR in {fn.__name__}: {e!r}")
        return fallback


def _pct(v, g=85, w=65):
    if v is None: return "neutral"
    try: v = float(v)
    except: return "neutral"
    return "healthy" if v >= g else "warning" if v >= w else "critical"


def _status(v, g=80, w=60):
    if v is None: return "neutral"
    try: v = float(v)
    except: return "neutral"
    return "healthy" if v >= g else "warning" if v >= w else "critical"


# ── Debug endpoint ────────────────────────────────────────────────────────────

@router.get("/debug")
def debug():
    db = _get_db_file()
    rows_sample = {}
    cols_sample = {}
    if db and os.path.exists(db):
        con = sqlite3.connect(db, timeout=5)
        con.row_factory = sqlite3.Row
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        for t in tables:
            try:
                rows_sample[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                # get column names
                cols_sample[t] = [d[0] for d in con.execute(f"SELECT * FROM {t} LIMIT 1").description or []]
            except Exception as e:
                rows_sample[t] = f"error: {e}"
        con.close()
    else:
        tables = []

    # Sample actual rows from key tables
    samples = {}
    for tbl in ["profiling_runs", "quality_snapshots", "drift_records", "temporal_checks", "column_profiles"]:
        if tbl in (rows_sample if isinstance(rows_sample.get(tbl), int) else {}):
            try:
                con2 = sqlite3.connect(db, timeout=5)
                con2.row_factory = sqlite3.Row
                s = [dict(r) for r in con2.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchall()]
                samples[tbl] = s
                con2.close()
            except Exception as e:
                samples[tbl] = f"error: {e}"

    return {
        "db_file": db,
        "db_exists": os.path.exists(db) if db else False,
        "row_counts": rows_sample,
        "column_names": cols_sample,
        "sample_rows": samples,
    }


# ── datasets-list ─────────────────────────────────────────────────────────────

@router.get("/datasets-list")
def datasets_list():
    rows = _q("SELECT * FROM datasets LIMIT 200")
    if not rows:
        return []
    result = []
    for r in rows:
        # try every possible name column
        name = r.get("name") or r.get("display_name") or r.get("physical_name") or r.get("dataset_name") or f"Dataset {r.get('id','?')}"
        result.append({"id": r.get("id"), "display_name": name})
    return result


# ── Tab builders ──────────────────────────────────────────────────────────────

def _tab_global_llm():
    runs = _q("SELECT * FROM profiling_runs")
    completed = [r for r in runs if str(r.get("status","")).lower() == "completed"]
    n_c = len(completed)

    no_summary = [r for r in completed
                  if not r.get("summary") and not r.get("ai_summary")
                  and not r.get("description") and not r.get("notes")]
    hall = round(len(no_summary)/n_c*100, 1) if n_c else 0.0

    # latency from profiling run duration
    latencies = []
    for r in completed:
        for start_col, end_col in [("created_at","updated_at"),("started_at","completed_at"),("start_time","end_time")]:
            sv, ev = r.get(start_col), r.get(end_col)
            if sv and ev:
                try:
                    s = datetime.fromisoformat(str(sv).replace("Z","+00:00").replace(" ","T"))
                    e = datetime.fromisoformat(str(ev).replace("Z","+00:00").replace(" ","T"))
                    ms = (e-s).total_seconds()*1000
                    if 0 < ms < 3_600_000:
                        latencies.append(ms)
                    break
                except: pass
    avg_lat = round(sum(latencies)/len(latencies)) if latencies else 0

    runs_with_profiles = int(_scalar("SELECT COUNT(DISTINCT run_id) FROM column_profiles WHERE run_id IS NOT NULL"))
    relevance = round(runs_with_profiles/n_c*100,1) if n_c else (100.0 if runs_with_profiles else 0.0)

    return {
        "tab": "Global AI / LLM",
        "metrics": [
            {"id":"hallucination_rate","label":"Hallucination Rate","value":hall,"unit":"%",
             "status":_status(100-hall,95,80),
             "formula":"completed_runs_without_ai_summary / completed_runs × 100",
             "details":{"no_summary":len(no_summary),"completed":n_c}},
            {"id":"avg_llm_latency_ms","label":"Avg LLM Latency","value":avg_lat,"unit":"ms",
             "status":"healthy" if avg_lat<30000 else "warning" if avg_lat<120000 else "critical",
             "formula":"mean(run duration in ms) across completed profiling runs",
             "details":{"samples":len(latencies)}},
            {"id":"response_relevance","label":"Response Relevance","value":relevance,"unit":"%",
             "status":_pct(relevance),
             "formula":"runs_with_column_profiles / completed_runs × 100",
             "details":{"runs_with_profiles":runs_with_profiles,"completed":n_c}},
            {"id":"response_consistency","label":"Response Consistency","value":round(max(0.0,100.0-hall*1.5),1),"unit":"%",
             "status":_pct(round(max(0.0,100.0-hall*1.5),1)),
             "formula":"100 - (hallucination_rate × 1.5)","details":{}},
        ],
        "explainability":{
            "overview":"Global AI/LLM metrics derived from profiling run outcomes and column profile generation rates.",
            "improvement":"Ensure profiling runs complete and generate column profiles. Check LLM key in Render environment variables.",
            "low_success_rate":"High hallucination rate means completed runs aren't producing AI output. Check AZURE_OPENAI_API_KEY on Render.",
        },
    }


def _tab_profiling():
    runs = _q("SELECT * FROM profiling_runs")
    total = len(runs)
    completed = [r for r in runs if str(r.get("status","")).lower() == "completed"]
    failed    = [r for r in runs if str(r.get("status","")).lower() in ("failed","error")]
    n_c = len(completed)

    success_rate = round(n_c/total*100,1) if total else 0.0

    grounded = [r for r in completed if r.get("summary") or r.get("ai_summary") or r.get("description")]
    grounding = round(len(grounded)/n_c*100,1) if n_c else 0.0

    runs_with_drift = int(_scalar("SELECT COUNT(DISTINCT run_id) FROM drift_records WHERE run_id IS NOT NULL"))
    drift_cov = round(runs_with_drift/n_c*100,1) if n_c else 0.0

    durations = []
    for r in completed:
        for sc,ec in [("created_at","updated_at"),("started_at","completed_at"),("start_time","end_time")]:
            sv,ev = r.get(sc), r.get(ec)
            if sv and ev:
                try:
                    s=datetime.fromisoformat(str(sv).replace("Z","+00:00").replace(" ","T"))
                    e=datetime.fromisoformat(str(ev).replace("Z","+00:00").replace(" ","T"))
                    sec=(e-s).total_seconds()
                    if 0<sec<3600: durations.append(sec)
                    break
                except: pass
    avg_rt = round(sum(durations)/len(durations),1) if durations else 0.0

    return {
        "tab":"Profiling AI",
        "metrics":[
            {"id":"profiling_success_rate","label":"Profiling Success Rate","value":success_rate,"unit":"%",
             "status":_pct(success_rate),
             "formula":"completed_runs / total_runs × 100",
             "details":{"completed":n_c,"total":total,"failed":len(failed)}},
            {"id":"metadata_grounding_score","label":"Metadata Grounding Score","value":grounding,"unit":"%",
             "status":_pct(grounding),
             "formula":"runs_with_ai_summary / completed_runs × 100",
             "details":{"grounded":len(grounded),"completed":n_c}},
            {"id":"drift_detection_accuracy","label":"Drift Detection Coverage","value":drift_cov,"unit":"%",
             "status":_pct(drift_cov),
             "formula":"runs_producing_drift_records / completed_runs × 100",
             "details":{"runs_with_drift":runs_with_drift,"completed":n_c}},
            {"id":"avg_profiling_runtime_s","label":"Avg Profiling Runtime","value":avg_rt,"unit":"s",
             "status":"healthy" if avg_rt<120 else "warning" if avg_rt<600 else "critical",
             "formula":"mean(run duration in seconds) across completed runs",
             "details":{"samples":len(durations)}},
        ],
        "explainability":{
            "overview":"Profiling AI metrics measure dataset profiling reliability, AI summary generation, and drift detection.",
            "improvement":"Check data source credentials if success rate is low. LLM summaries require AZURE_OPENAI_API_KEY on Render.",
            "low_success_rate":"Failed runs mean the data source is unreachable or credentials expired.",
        },
    }


def _tab_dq_scores():
    snaps = _q("SELECT * FROM quality_snapshots ORDER BY created_at DESC LIMIT 100")
    # find score column
    score_col = None
    if snaps:
        for c in ["overall_score","health_score","score","quality_score","dq_score"]:
            if c in snaps[0]:
                score_col = c
                break

    health_scores = []
    if score_col:
        for s in snaps:
            v = s.get(score_col)
            if v is not None:
                try: health_scores.append(float(v))
                except: pass

    avg_h = round(sum(health_scores)/len(health_scores),1) if health_scores else 0.0
    accuracy = round(sum(1 for s in health_scores if 0<=s<=100)/len(health_scores)*100,1) if health_scores else 100.0

    results = _q("SELECT * FROM dq_rule_run_results LIMIT 500")
    passed  = [r for r in results if str(r.get("status","")).lower() in ("passed","pass","ok","success")]
    compliance = round(len(passed)/len(results)*100,1) if results else 0.0

    recent = sorted([s for s in snaps if s.get(score_col) if score_col],
                    key=lambda x: str(x.get("created_at","")))[-5:]
    if len(recent)>=2 and score_col:
        vals=[float(r.get(score_col,0)) for r in recent]
        velocity=round(vals[-1]-vals[0],1)
    else:
        velocity=0.0

    return {
        "tab":"DQ Scores",
        "metrics":[
            {"id":"health_score_accuracy","label":"Health Score Accuracy","value":accuracy,"unit":"%",
             "status":_pct(accuracy),
             "formula":"snapshots_in_valid_range / total_snapshots × 100",
             "details":{"valid":len(health_scores),"total":len(snaps),"score_col":score_col}},
            {"id":"rule_compliance_accuracy","label":"Rule Pass Rate","value":compliance,"unit":"%",
             "status":_pct(compliance) if results else "neutral",
             "formula":"passed_rule_results / total_rule_results × 100",
             "details":{"passed":len(passed),"total":len(results)}},
            {"id":"avg_health_score","label":"Avg Dataset Health Score","value":avg_h,"unit":"%",
             "status":_pct(avg_h),
             "formula":f"mean({score_col or 'score'}) from quality_snapshots",
             "details":{"snapshots":len(health_scores)}},
            {"id":"health_degradation_velocity","label":"Health Degradation Velocity","value":velocity,"unit":"pts",
             "status":"healthy" if velocity>=-5 else "warning" if velocity>=-15 else "critical",
             "formula":"snapshot[-1].score - snapshot[0].score over last 5",
             "details":{"window":len(recent)}},
        ],
        "explainability":{
            "overview":"DQ Scores use quality_snapshots (2 rows) and dq_rule_run_results for health and compliance tracking.",
            "improvement":"Run DQ rule evaluation jobs to populate dq_rule_run_results. Trigger more profiling runs to build snapshot history.",
            "low_success_rate":"Only 2 quality snapshots exist. Generate more by running profiling + scoring jobs.",
        },
    }


def _tab_dq_rules():
    rules = _q("SELECT * FROM dq_rules")
    total = len(rules)
    active = [r for r in rules if r.get("is_active") or r.get("enabled") or str(r.get("status","")).lower()=="active"]
    active_ids = {r.get("id") for r in active}

    runs    = _q("SELECT * FROM dq_rule_runs LIMIT 500")
    results = _q("SELECT * FROM dq_rule_run_results LIMIT 500")
    executed_ids = {r.get("rule_id") for r in runs if r.get("rule_id")}
    exec_rate = round(len(executed_ids & active_ids)/len(active_ids)*100,1) if active_ids else 0.0

    ai_rules  = [r for r in rules if str(r.get("source","")).lower() in
                 ("llm","ai","recommended","generated","auto")]
    accepted  = [r for r in ai_rules if r.get("is_active") or r.get("enabled")]
    accept_rate = round(len(accepted)/len(ai_rules)*100,1) if ai_rules else 0.0

    hall_rules = [r for r in ai_rules if r.get("id") not in executed_ids]
    hall_rate  = round(len(hall_rules)/len(ai_rules)*100,1) if ai_rules else 0.0

    return {
        "tab":"DQ Rules",
        "metrics":[
            {"id":"rule_execution_success_rate","label":"Rule Execution Rate","value":exec_rate,"unit":"%",
             "status":_pct(exec_rate) if active_ids else "neutral",
             "formula":"active_rules_with_run_history / total_active_rules × 100",
             "details":{"executed":len(executed_ids & active_ids),"active":len(active_ids),"total":total}},
            {"id":"rule_recommendation_acceptance_rate","label":"AI Rule Acceptance Rate","value":accept_rate,"unit":"%",
             "status":_pct(accept_rate) if ai_rules else "neutral",
             "formula":"active_ai_rules / total_ai_suggested_rules × 100",
             "details":{"accepted":len(accepted),"suggested":len(ai_rules)}},
            {"id":"hallucinated_rule_rate","label":"Hallucinated Rule Rate","value":hall_rate,"unit":"%",
             "status":_status(100-hall_rate,90,75),
             "formula":"ai_rules_never_executed / total_ai_rules × 100",
             "details":{"never_run":len(hall_rules),"ai_rules":len(ai_rules)}},
        ],
        "explainability":{
            "overview":"DQ Rules tracks 8 rules — execution rate, AI-suggested rules, and hallucination proxy.",
            "improvement":"Execute rules via DQ Engine tab. dq_rule_runs is empty — trigger a rule evaluation run.",
            "low_success_rate":"No rule runs recorded yet. Go to DQ Engine in the main app and run all active rules.",
        },
    }


def _tab_monitoring():
    runs  = _q("SELECT * FROM profiling_runs")
    drift = _q("SELECT * FROM drift_records LIMIT 500")
    total = len(runs)
    completed = [r for r in runs if str(r.get("status","")).lower()=="completed"]
    uptime = round(len(completed)/total*100,1) if total else 0.0

    sig = [d for d in drift if str(d.get("severity","")).upper() in ("HIGH","CRITICAL","MEDIUM","SEVERE")]
    precision = round(len(sig)/len(drift)*100,1) if drift else 0.0

    snaps = _q("SELECT * FROM quality_snapshots ORDER BY created_at DESC LIMIT 20")
    vals=[]
    for s in snaps:
        for c in ["overall_score","health_score","score","quality_score"]:
            v=s.get(c)
            if v is not None:
                try: vals.append(float(v)); break
                except: pass
    if len(vals)>=2:
        mean_v=sum(vals)/len(vals)
        std_v=(sum((x-mean_v)**2 for x in vals)/len(vals))**0.5
        volatility=round(std_v,1)
    else:
        volatility=0.0

    return {
        "tab":"Monitoring & Trends",
        "metrics":[
            {"id":"monitoring_uptime","label":"Profiling Run Success Rate","value":uptime,"unit":"%",
             "status":_pct(uptime),
             "formula":"completed_runs / total_runs × 100",
             "details":{"completed":len(completed),"total":total}},
            {"id":"drift_detection_precision","label":"Drift Alert Precision","value":precision,"unit":"%",
             "status":_pct(precision) if drift else "neutral",
             "formula":"MEDIUM/HIGH/CRITICAL drift_records / total × 100",
             "details":{"significant":len(sig),"total":len(drift)}},
            {"id":"forecast_error_rate","label":"Health Score Volatility","value":volatility,"unit":"pts std",
             "status":"healthy" if volatility<5 else "warning" if volatility<15 else "critical",
             "formula":"stddev(quality_snapshots scores)",
             "details":{"samples":len(vals)}},
        ],
        "explainability":{
            "overview":"Monitoring: profiling success rate, drift signal quality, health score stability.",
            "improvement":"208 drift_records exist. Check severity distribution — if all are LOW, the drift engine needs threshold tuning.",
            "high_drift":"208 drift records found. Review in main app Monitoring tab.",
        },
    }


def _tab_anomalies():
    checks = _q("SELECT * FROM temporal_checks LIMIT 500")
    total  = len(checks)

    # Inspect actual status values
    status_vals = set(str(c.get("status","")).lower() for c in checks)
    flagged = [c for c in checks if str(c.get("status","")).lower() in
               ("failed","fail","error","anomaly","alert","warning","flagged")]
    passed  = [c for c in checks if str(c.get("status","")).lower() in
               ("passed","pass","ok","success","normal","healthy")]

    flag_rate = round(len(flagged)/total*100,1) if total else 0.0
    pass_rate = round(len(passed)/total*100,1) if total else 0.0

    rca_att = [c for c in flagged if c.get("explanation") or c.get("reason") or c.get("details") or c.get("message")]
    rca_bad = [c for c in rca_att if len(str(c.get("explanation") or c.get("reason") or c.get("message") or ""))<20]
    rca_quality = round((len(rca_att)-len(rca_bad))/len(rca_att)*100,1) if rca_att else (100.0 if total else 0.0)

    results = _q("SELECT * FROM dq_rule_run_results LIMIT 500")
    recall  = round(len(flagged)/len(results)*100,1) if results else (100.0 if not flagged else 0.0)
    recall  = min(100.0, recall)

    return {
        "tab":"Anomalies AI",
        "metrics":[
            {"id":"anomaly_precision","label":"Temporal Check Anomaly Rate","value":flag_rate,"unit":"%",
             "status":_status(100-flag_rate,70,50) if total else "neutral",
             "formula":"failed_temporal_checks / total_temporal_checks × 100",
             "details":{"flagged":len(flagged),"total":total,"passed":len(passed),"status_values_found":list(status_vals)[:8]}},
            {"id":"anomaly_recall","label":"Detection Recall","value":recall,"unit":"%",
             "status":_pct(recall) if results else "neutral",
             "formula":"flagged_temporal_checks / total_rule_results × 100",
             "details":{"flagged":len(flagged),"rule_results":len(results)}},
            {"id":"rca_hallucination_rate","label":"RCA Quality Rate","value":rca_quality,"unit":"%",
             "status":_pct(rca_quality) if rca_att else "neutral",
             "formula":"checks_with_meaningful_explanation / flagged × 100",
             "details":{"good_rca":len(rca_att)-len(rca_bad),"attempts":len(rca_att)}},
            {"id":"auto_fix_success_rate","label":"Check Resolution Rate","value":pass_rate,"unit":"%",
             "status":_pct(pass_rate) if total else "neutral",
             "formula":"passed_temporal_checks / total × 100",
             "details":{"passed":len(passed),"total":total}},
        ],
        "explainability":{
            "overview":"Anomaly metrics use temporal_checks (353 rows) — AI-generated statistical validation checks across dataset columns.",
            "improvement":"Review temporal check status values. If all checks have unusual status strings, thresholds may need adjustment.",
            "many_critical":"353 temporal checks recorded. Review in main app — check what status values are actually being set.",
        },
    }


def _tab_lineage():
    edges = _q("SELECT * FROM lineage_edges LIMIT 500")
    total = len(edges)
    broken   = [e for e in edges if str(e.get("status","")).lower()=="broken" or e.get("is_stale")]
    low_conf = [e for e in edges if (e.get("confidence") or 1.0) < 0.5]
    ds_count = int(_scalar("SELECT COUNT(*) FROM datasets"))
    mapped   = set(e.get("source_dataset_id") for e in edges) | set(e.get("target_dataset_id") for e in edges)
    mapped.discard(None)
    coverage = round(len(mapped)/ds_count*100,1) if ds_count else 0.0

    return {
        "tab":"Data Lineage & Impact",
        "metrics":[
            {"id":"lineage_coverage","label":"Lineage Coverage","value":coverage,"unit":"%",
             "status":_pct(coverage) if ds_count else "neutral",
             "formula":"datasets_with_lineage_edges / total_datasets × 100",
             "details":{"mapped":len(mapped),"total":ds_count,"edges":total}},
            {"id":"broken_edge_count","label":"Broken Lineage Edges","value":len(broken),"unit":"",
             "status":"healthy" if not broken else "warning" if len(broken)<5 else "critical",
             "formula":"COUNT(lineage_edges WHERE status='broken')",
             "details":{"broken":len(broken),"total":total}},
            {"id":"missed_dependency_rate","label":"Low-Confidence Dependency Rate",
             "value":round(len(low_conf)/total*100,1) if total else 0.0,"unit":"%",
             "status":_pct(100-len(low_conf)/total*100) if total else "neutral",
             "formula":"edges_with_confidence < 0.5 / total × 100",
             "details":{"low_confidence":len(low_conf),"total":total}},
        ],
        "explainability":{
            "overview":"Lineage tracks upstream/downstream dataset relationships via lineage_edges (0 rows currently).",
            "improvement":"Lineage edges are auto-generated during profiling. Run full profiling on all 10 datasets.",
            "low_coverage":"0 lineage edges — expected if lineage engine hasn't been triggered yet.",
        },
    }


def _tab_kg():
    edges     = _q("SELECT * FROM knowledge_graph_edges LIMIT 1000")
    total     = len(edges)
    high_conf = [e for e in edges if float(e.get("confidence") or e.get("weight") or 0)>=0.7]
    null_conf = [e for e in edges if e.get("confidence") is None and e.get("weight") is None]
    col_edges = [e for e in edges if str(e.get("edge_type","")).lower() in
                 ("similar_to","references","maps_to","column_similarity","related_to")]
    total_cols = int(_scalar("SELECT COUNT(DISTINCT column_name) FROM column_profiles WHERE column_name IS NOT NULL"))
    precision  = round(len(high_conf)/total*100,1) if total else 0.0
    hall_rate  = round(len(null_conf)/total*100,1) if total else 0.0
    mapping    = round(len(col_edges)/total_cols*100,1) if total_cols and col_edges else 0.0

    return {
        "tab":"Knowledge Graph AI",
        "metrics":[
            {"id":"kg_relationship_precision","label":"Relationship Precision","value":precision,"unit":"%",
             "status":_pct(precision) if total else "neutral",
             "formula":"KG_edges_with_confidence≥0.7 / total_KG_edges × 100",
             "details":{"high_confidence":len(high_conf),"total":total}},
            {"id":"kg_column_mapping_accuracy","label":"Column Relationship Coverage","value":mapping,"unit":"%",
             "status":_pct(mapping) if total else "neutral",
             "formula":"column_relationship_edges / distinct_columns × 100",
             "details":{"col_edges":len(col_edges),"distinct_cols":total_cols}},
            {"id":"kg_hallucinated_relationship_rate","label":"Unscored Relationship Rate","value":hall_rate,"unit":"%",
             "status":_status(100-hall_rate,90,75) if total else "neutral",
             "formula":"edges_with_no_confidence / total × 100",
             "details":{"unscored":len(null_conf),"total":total}},
        ],
        "explainability":{
            "overview":"KG metrics use knowledge_graph_edges (0 rows). Graph construction hasn't been triggered yet.",
            "improvement":"Use the Knowledge Graph tab in the main AI DQM app to build the graph from your 65 profiled columns.",
            "low_coverage":"0 KG edges is expected — trigger KG construction from the main app.",
        },
    }


def _tab_assistant():
    inbox     = _q("SELECT * FROM notification_inbox LIMIT 500")
    gov_notif = _q("SELECT * FROM governance_notifications LIMIT 200")
    n_inbox   = len(inbox)
    n_gov     = len(gov_notif)
    total     = n_inbox + n_gov

    tagged  = [n for n in inbox if n.get("type") or n.get("category") or n.get("notification_type")]
    routing = round(len(tagged)/n_inbox*100,1) if n_inbox else 0.0

    bad_msg  = [n for n in inbox if not n.get("message") and not n.get("content") and not n.get("body")]
    quality  = round((n_inbox-len(bad_msg))/n_inbox*100,1) if n_inbox else 0.0

    grounded = [n for n in inbox if n.get("dataset_id") or n.get("entity_id") or n.get("run_id") or n.get("resource_id")]
    grounding= round(len(grounded)/n_inbox*100,1) if n_inbox else 0.0

    gov_rate = round(n_gov/total*100,1) if total else 0.0

    return {
        "tab":"DQ Assistant / AI Agent",
        "metrics":[
            {"id":"agent_routing_accuracy","label":"Notification Routing Accuracy","value":routing,"unit":"%",
             "status":_pct(routing) if n_inbox else "neutral",
             "formula":"notifications_with_type_tag / total_inbox × 100",
             "details":{"tagged":len(tagged),"total":n_inbox}},
            {"id":"assistant_hallucination_rate","label":"Notification Quality Rate","value":quality,"unit":"%",
             "status":_pct(quality) if n_inbox else "neutral",
             "formula":"notifications_with_content / total × 100",
             "details":{"good":n_inbox-len(bad_msg),"total":n_inbox}},
            {"id":"action_agent_success_rate","label":"Governance Action Rate","value":gov_rate,"unit":"%",
             "status":_pct(gov_rate) if total else "neutral",
             "formula":"governance_notifications / total_notifications × 100",
             "details":{"governance":n_gov,"total":total}},
            {"id":"retrieval_grounding_score","label":"Context Grounding Score","value":grounding,"unit":"%",
             "status":_pct(grounding) if n_inbox else "neutral",
             "formula":"notifications_with_dataset_reference / total × 100",
             "details":{"grounded":len(grounded),"total":n_inbox}},
            {"id":"context_retention_accuracy","label":"Context Retention Accuracy","value":round(quality*0.9,1),"unit":"%",
             "status":_pct(round(quality*0.9,1)) if n_inbox else "neutral",
             "formula":"notification_quality_rate × 0.9 — proxy for multi-turn context preservation",
             "details":{"basis":"notification_message_quality","inbox_count":n_inbox}},
        ],
        "explainability":{
            "overview":"Assistant metrics derived from notification_inbox (12) and governance_notifications (8).",
            "improvement":"Link notifications to specific dataset IDs to improve grounding score. More profiling generates more notifications.",
            "low_satisfaction":"Low grounding: 0/12 inbox notifications reference a dataset_id. Check notification generation logic.",
        },
    }


def _tab_governance():
    policies  = _q("SELECT * FROM governance_policies LIMIT 200")
    total     = len(policies)
    ai_pol    = [p for p in policies if str(p.get("source","")).lower() in ("llm","ai","suggested","generated","auto")]
    accepted  = [p for p in ai_pol if p.get("is_active") or p.get("enabled")]
    adopt     = round(len(accepted)/len(ai_pol)*100,1) if ai_pol else 0.0

    classified = int(_scalar(
        "SELECT COUNT(*) FROM column_profiles WHERE sensitivity_label IS NOT NULL AND sensitivity_label != ''"
    ))
    total_cols = int(_scalar("SELECT COUNT(*) FROM column_profiles"))
    class_acc  = round(classified/total_cols*100,1) if total_cols else 0.0

    audit_n  = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    action_n = int(_scalar("SELECT COUNT(*) FROM profiling_runs")) + int(_scalar("SELECT COUNT(*) FROM dq_rules"))
    audit_c  = round(min(100.0, audit_n/action_n*100), 1) if action_n else 0.0

    return {
        "tab":"Governance & Settings",
        "metrics":[
            {"id":"policy_adoption_rate","label":"Policy Adoption Rate","value":adopt,"unit":"%",
             "status":_pct(adopt) if ai_pol else "neutral",
             "formula":"active_ai_policies / total_ai_suggested × 100",
             "details":{"accepted":len(accepted),"suggested":len(ai_pol),"total":total}},
            {"id":"classification_accuracy","label":"Column Sensitivity Classification","value":class_acc,"unit":"%",
             "status":_pct(class_acc) if total_cols else "neutral",
             "formula":"columns_with_sensitivity_label / total_columns × 100",
             "details":{"classified":classified,"total":total_cols}},
            {"id":"audit_log_completeness","label":"Audit Log Completeness","value":audit_c,"unit":"%",
             "status":_pct(audit_c),
             "formula":"governance_audit_log_entries / (profiling_runs + dq_rules) × 100",
             "details":{"audit_entries":audit_n,"actions":action_n}},
        ],
        "explainability":{
            "overview":"Governance: 1 policy, 287 column_profiles (0 sensitivity labels), 3 audit log entries.",
            "improvement":"Add sensitivity labels to columns in the Governance tab. Generate more AI policy suggestions.",
            "low_adoption":"No AI-suggested policies found (source field not 'llm'/'ai'). Check policy generation in main app.",
        },
    }


def _tab_system():
    runs      = _q("SELECT * FROM profiling_runs")
    completed = [r for r in runs if str(r.get("status","")).lower()=="completed"]
    failed    = [r for r in runs if str(r.get("status","")).lower() in ("failed","error")]
    total     = len(runs)
    uptime    = round(len(completed)/(len(completed)+len(failed))*100,1) if (completed or failed) else 100.0

    throughput = 0.0
    if total >= 2:
        sorted_r = sorted([r for r in runs if r.get("created_at")], key=lambda x: str(x["created_at"]))
        try:
            t0 = datetime.fromisoformat(str(sorted_r[0]["created_at"]).replace("Z","+00:00").replace(" ","T"))
            t1 = datetime.fromisoformat(str(sorted_r[-1]["created_at"]).replace("Z","+00:00").replace(" ","T"))
            hours = max(1,(t1-t0).total_seconds()/3600)
            throughput = round(total/hours,2)
        except: pass

    durations=[]
    for r in completed:
        for sc,ec in [("created_at","updated_at"),("started_at","completed_at")]:
            sv,ev=r.get(sc),r.get(ec)
            if sv and ev:
                try:
                    s=datetime.fromisoformat(str(sv).replace("Z","+00:00").replace(" ","T"))
                    e=datetime.fromisoformat(str(ev).replace("Z","+00:00").replace(" ","T"))
                    ms=(e-s).total_seconds()*1000
                    if 0<ms<3600000: durations.append(ms)
                    break
                except: pass
    avg_ms=round(sum(durations)/len(durations)) if durations else 0

    return {
        "tab":"System / Platform",
        "metrics":[
            {"id":"system_uptime","label":"System Uptime","value":uptime,"unit":"%",
             "status":_pct(uptime),
             "formula":"completed_runs / (completed + failed) × 100",
             "details":{"completed":len(completed),"failed":len(failed),"total":total}},
            {"id":"api_throughput","label":"Processing Throughput","value":throughput,"unit":"runs/hr",
             "status":"healthy" if throughput>0 else "neutral",
             "formula":"total_profiling_runs / elapsed_hours",
             "details":{"total_runs":total}},
            {"id":"avg_job_duration_ms","label":"Avg Job Duration","value":avg_ms,"unit":"ms",
             "status":"healthy" if avg_ms<60000 else "warning" if avg_ms<300000 else "critical",
             "formula":"mean(run_duration_ms) across completed runs",
             "details":{"samples":len(durations)}},
        ],
        "explainability":{
            "overview":"System metrics based on profiling_runs history (35 total runs in DB).",
            "improvement":"35 runs recorded. Monitor via Render logs for failures.",
            "low_success_rate":"Check Render logs for failed profiling runs.",
        },
    }


def _tab_feedback():
    dismissed   = int(_scalar("SELECT COUNT(*) FROM governance_dismissed_suggestions"))
    active_pol  = int(_scalar(
        "SELECT COUNT(*) FROM governance_policies WHERE is_active=1 OR status='active'"
    ))
    total_sugg  = dismissed + active_pol
    accept_rate = round(active_pol/total_sugg*100,1) if total_sugg else 0.0

    audit_n     = int(_scalar("SELECT COUNT(*) FROM governance_audit_log"))
    satisfaction= round(min(100.0, audit_n/10*100),1)

    return {
        "tab":"Human Feedback",
        "metrics":[
            {"id":"ai_acceptance_rate","label":"AI Suggestion Acceptance Rate","value":accept_rate,"unit":"%",
             "status":_pct(accept_rate) if total_sugg else "neutral",
             "formula":"active_policies / (active + dismissed) × 100",
             "details":{"accepted":active_pol,"dismissed":dismissed,"total":total_sugg}},
            {"id":"analyst_satisfaction_score","label":"Platform Engagement Score","value":satisfaction,"unit":"/100",
             "status":_pct(satisfaction) if audit_n else "neutral",
             "formula":"min(audit_log_actions/10×100, 100)",
             "details":{"audit_actions":audit_n}},
        ],
        "explainability":{
            "overview":"Feedback uses governance policy acceptance (1 dismissed, 0 active) and audit log activity (3 entries).",
            "improvement":"Accept governance policies instead of dismissing. More audit activity increases engagement score.",
            "low_satisfaction":"3 audit entries → 30/100 engagement. Use Governance tab more actively.",
        },
    }


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.get("/")
@router.get("")
def get_all_metrics(dataset_id: Optional[int] = Query(None)):
    db = _get_db_file()
    fallback = lambda name: {"tab": name, "metrics": [], "explainability": {}}
    tabs = [
        _safe(_tab_global_llm,  fallback("Global AI / LLM")),
        _safe(_tab_profiling,   fallback("Profiling AI")),
        _safe(_tab_dq_scores,   fallback("DQ Scores")),
        _safe(_tab_dq_rules,    fallback("DQ Rules")),
        _safe(_tab_monitoring,  fallback("Monitoring & Trends")),
        _safe(_tab_anomalies,   fallback("Anomalies AI")),
        _safe(_tab_lineage,     fallback("Data Lineage & Impact")),
        _safe(_tab_kg,          fallback("Knowledge Graph AI")),
        _safe(_tab_assistant,   fallback("DQ Assistant / AI Agent")),
        _safe(_tab_governance,  fallback("Governance & Settings")),
        _safe(_tab_system,      fallback("System / Platform")),
        _safe(_tab_feedback,    fallback("Human Feedback")),
    ]
    return {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "dataset_id": dataset_id,
        "db_path": db,
        "tabs": tabs,
    }
