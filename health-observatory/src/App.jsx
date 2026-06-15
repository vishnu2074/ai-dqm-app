import { useState, useEffect, useCallback, useRef } from 'react'

const API_URL = '/api/health-metrics'
const REFRESH_MS = 30_000   // 30-second live polling

// ── All metrics from the backend ─────────────────────────────────────────────
const WORKING = new Set([
  // Global AI / LLM
  'hallucination_rate', 'avg_llm_latency_ms', 'response_relevance',
  'llm_output_schema_compliance_rate',
  'ai_column_description_rate', 'ai_insight_depth_score', 'llm_dataset_coverage',
  // Profiling AI
  'profiling_success_rate', 'metadata_grounding_score',
  'drift_detection_accuracy', 'avg_profiling_runtime_s',
  // DQ Scores
  'health_score_accuracy', 'rule_compliance_accuracy', 'avg_health_score',
  'health_degradation_velocity',
  // DQ Rules
  'rule_execution_success_rate', 'rule_recommendation_acceptance_rate',
  'hallucinated_rule_rate', 'rule_coverage_rate',
  // Monitoring & Trends
  'avg_runs_last_7_days', 'drift_detection_precision', 'drift_volume_trend',
  'forecast_error_rate',
  // Anomalies AI
  'anomaly_precision', 'anomaly_open_rate', 'auto_fix_success_rate',
  // Data Lineage
  'lineage_coverage', 'broken_edge_count', 'missed_dependency_rate',
  'datasets_profiled_rate',
  // Knowledge Graph AI
  'kg_build_status', 'kg_relationship_precision', 'kg_hallucinated_relationship_rate',
  // DQ Assistant
  'agent_routing_accuracy', 'notification_content_rate',
  'action_agent_success_rate', 'retrieval_grounding_score',
  'avg_notifications_per_dataset',
  // Governance & Settings
  'policy_adoption_rate', 'classification_accuracy', 'audit_log_completeness',
  // System / Platform
  'system_uptime', 'api_throughput', 'avg_job_duration_ms',
  // Human Feedback
  'ai_acceptance_rate', 'governance_activity_index',
  // Azure LLM Usage
  'azure_total_requests', 'azure_total_tokens', 'azure_token_efficiency',
  'azure_avg_latency_ms', 'azure_error_rate', 'azure_throttle_rate',
  'azure_estimated_cost_usd',
])

const S = {
  healthy:  { hex: '#10b981', glow: '0 0 8px rgba(16,185,129,0.45)',  bg: 'rgba(16,185,129,0.09)',  label: 'Healthy'  },
  warning:  { hex: '#f59e0b', glow: '0 0 8px rgba(245,158,11,0.45)',  bg: 'rgba(245,158,11,0.09)',  label: 'Warning'  },
  critical: { hex: '#ef4444', glow: '0 0 8px rgba(239,68,68,0.45)',   bg: 'rgba(239,68,68,0.09)',   label: 'Critical' },
  neutral:  { hex: '#64748b', glow: 'none',                            bg: 'rgba(100,116,139,0.09)', label: 'Neutral'  },
}

const TAB_ICONS = {
  'Global AI / LLM':          '⬡',
  'Profiling AI':              '⊙',
  'DQ Scores':                 '◈',
  'DQ Rules':                  '⊞',
  'Monitoring & Trends':       '⌖',
  'Anomalies AI':              '⚠',
  'Data Lineage & Impact':     '⊶',
  'Knowledge Graph AI':        '⬡',
  'DQ Assistant / AI Agent':   '◎',
  'Governance & Settings':     '⊛',
  'System / Platform':         '⊟',
  'Human Feedback':            '◉',
  'Azure LLM Usage':           '☁',
}

function fmt(value, unit) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value
  const n = parseFloat(value)
  if (isNaN(n)) return String(value)
  if (unit === '%')        return `${Number.isInteger(n) ? n : n.toFixed(2)}`
  if (unit === 'ms')       return `${Math.round(n).toLocaleString()}`
  if (unit === 'chars')    return `${Math.round(n)}`
  if (unit === 's')        return `${n.toFixed(2)}`
  if (unit === 'runs/day') return `${n.toFixed(2)}`
  if (unit === 'runs/hr')  return `${n.toFixed(3)}`
  if (unit === 'pts std')  return `${n.toFixed(3)}`
  if (unit === 'pts')      return `${n >= 0 ? '+' : ''}${n.toFixed(1)}`
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}

// ── Countdown ring ────────────────────────────────────────────────────────────
function CountdownRing({ msLeft, totalMs }) {
  const pct = Math.max(0, msLeft / totalMs)
  const r = 10, circ = 2 * Math.PI * r
  const dash = circ * pct
  return (
    <svg width="28" height="28" className="countdown-ring">
      <circle cx="14" cy="14" r={r} className="ring-bg" />
      <circle cx="14" cy="14" r={r} className="ring-fill"
        style={{ strokeDasharray: `${dash} ${circ}`, stroke: pct > 0.3 ? '#10b981' : '#f59e0b' }} />
      <text x="14" y="18" className="ring-text">
        {Math.ceil(msLeft / 1000)}
      </text>
    </svg>
  )
}

// ── Metric Card ───────────────────────────────────────────────────────────────
function MetricCard({ m, isNew }) {
  const [open, setOpen] = useState(false)
  const cfg = S[m.status] || S.neutral
  const val = fmt(m.value, m.unit)
  const hasData = m.value !== null && m.value !== undefined

  return (
    <div className={`card ${isNew ? 'card-new' : ''}`}
         style={{ '--sc': cfg.hex, '--sg': cfg.glow, '--sb': cfg.bg }}
         onClick={() => setOpen(o => !o)}>
      <div className="card-top">
        <span className="card-label">
          {isNew && <span className="new-badge">NEW</span>}
          {m.label}
        </span>
        <span className="pill" style={{ color: cfg.hex, background: cfg.bg }}>
          <span className="dot" style={{ background: cfg.hex, boxShadow: cfg.glow }} />
          {cfg.label}
        </span>
      </div>

      <div className="card-body">
        <span className="card-val" style={{ color: hasData ? cfg.hex : '#334155' }}>
          {val}
        </span>
        {m.unit && hasData && (
          <span className="card-unit">{m.unit}</span>
        )}
      </div>

      <div className="card-formula">{m.formula}</div>

      {open && m.details && Object.keys(m.details).length > 0 && (
        <div className="card-details">
          {Object.entries(m.details).map(([k, v]) =>
            v !== null && v !== undefined ? (
              <div key={k} className="det-row">
                <span className="det-k">{k.replace(/_/g, ' ')}</span>
                <span className="det-v">
                  {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                </span>
              </div>
            ) : null
          )}
        </div>
      )}
      <div className="card-caret">{open ? '▲' : '▼'}</div>
    </div>
  )
}

const NEW_METRICS = new Set([
  'ai_column_description_rate', 'ai_insight_depth_score', 'llm_dataset_coverage',
  'rule_compliance_accuracy', 'drift_volume_trend',
  'kg_build_status', 'kg_relationship_precision', 'kg_hallucinated_relationship_rate',
  'rule_execution_success_rate', 'rule_recommendation_acceptance_rate', 'hallucinated_rule_rate',
  'azure_total_requests', 'azure_total_tokens', 'azure_token_efficiency',
  'azure_avg_latency_ms', 'azure_error_rate', 'azure_throttle_rate',
  'azure_estimated_cost_usd',
])

// ── Summary strip ─────────────────────────────────────────────────────────────
function Summary({ tabs }) {
  const counts = { healthy: 0, warning: 0, critical: 0, neutral: 0 }
  tabs.forEach(t =>
    t.metrics.filter(m => WORKING.has(m.id))
             .forEach(m => { counts[m.status] = (counts[m.status] || 0) + 1 })
  )
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  return (
    <div className="summary">
      {(['healthy','warning','critical','neutral']).map(s =>
        counts[s] > 0 ? (
          <div key={s} className="sum-item" style={{ color: S[s].hex }}>
            <span className="sum-n">{counts[s]}</span>
            <span className="sum-l">{S[s].label}</span>
          </div>
        ) : null
      )}
      <div className="sum-item" style={{ color: '#94a3b8' }}>
        <span className="sum-n">{total}</span>
        <span className="sum-l">Active</span>
      </div>
    </div>
  )
}

function DbBadge({ path }) {
  if (!path) return null
  const ok = path.startsWith('/var/data')
  return (
    <span className={`db-badge ${ok ? 'db-ok' : 'db-warn'}`}>
      {ok ? '●' : '○'} {path}
    </span>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData]           = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [activeTab, setActiveTab] = useState(0)
  const [lastAt, setLastAt]       = useState(null)
  const [fetching, setFetching]   = useState(false)
  const [msLeft, setMsLeft]       = useState(REFRESH_MS)
  const nextFetchAt               = useRef(Date.now() + REFRESH_MS)

  const load = useCallback(async (manual = false) => {
    if (manual) setFetching(true)
    try {
      const r = await fetch(`${API_URL}?t=${Date.now()}`)   // bust cache
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setData(await r.json())
      setError(null)
      setLastAt(new Date())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      setFetching(false)
      nextFetchAt.current = Date.now() + REFRESH_MS
      setMsLeft(REFRESH_MS)
    }
  }, [])

  // Auto-refresh
  useEffect(() => {
    load()
    const interval = setInterval(load, REFRESH_MS)
    return () => clearInterval(interval)
  }, [load])

  // Countdown ticker (updates every second)
  useEffect(() => {
    const tick = setInterval(() => {
      setMsLeft(Math.max(0, nextFetchAt.current - Date.now()))
    }, 1000)
    return () => clearInterval(tick)
  }, [])

  const liveTabs = (data?.tabs ?? [])
    .filter(t => t.metrics.some(m => WORKING.has(m.id)))

  if (loading) return (
    <div className="obs">
      <div className="splash">
        <div className="loader" />
        <p>Connecting to AI DQM Observatory…</p>
      </div>
    </div>
  )

  const curTab = liveTabs[activeTab]

  return (
    <div className="obs">

      {/* ── Header ── */}
      <header className="hdr">
        <div className="hdr-l">
          <div className="hdr-icon">⬡</div>
          <div className="hdr-text">
            <h1>
              AI DQM Health Observatory
              <span className="live-badge">
                <span className="live-dot" />
                LIVE
              </span>
            </h1>
            <div className="hdr-meta">
              <DbBadge path={data?.db_path} />
              {lastAt && (
                <span className="hdr-time">
                  Updated {lastAt.toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="hdr-r">
          {data && <Summary tabs={data.tabs} />}
          <div className="refresh-group">
            <CountdownRing msLeft={msLeft} totalMs={REFRESH_MS} />
            <button
              className={`ref-btn ${fetching ? 'fetching' : ''}`}
              onClick={() => load(true)}
              disabled={fetching}
              title="Refresh now"
            >
              {fetching ? <span className="spin-icon">↻</span> : '↻'}
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="err-bar">
          ⚠ {error} — check that <code>vite.config.js</code> proxy target is correct
        </div>
      )}

      {/* ── Fetch indicator ── */}
      {fetching && <div className="fetch-bar" />}

      {/* ── Tab Nav ── */}
      <nav className="tab-nav">
        <div className="tab-scroll">
          {liveTabs.map((t, i) => {
            const wc = t.metrics.filter(m => WORKING.has(m.id)).length
            const statuses = t.metrics.filter(m => WORKING.has(m.id)).map(m => m.status)
            const worst = statuses.includes('critical') ? 'critical'
                        : statuses.includes('warning')  ? 'warning'
                        : statuses.includes('healthy')  ? 'healthy' : 'neutral'
            return (
              <button key={t.tab}
                      className={`tab-btn ${activeTab === i ? 'active' : ''}`}
                      style={activeTab === i ? { '--tc': S[worst].hex } : {}}
                      onClick={() => setActiveTab(i)}>
                <span className="tab-icon">{TAB_ICONS[t.tab] || '◆'}</span>
                <span className="tab-name">{t.tab}</span>
                <span className="tab-badge"
                      style={{ background: S[worst].bg, color: S[worst].hex }}>
                  {wc}
                </span>
              </button>
            )
          })}
        </div>
      </nav>

      {/* ── Panel ── */}
      <main className="panel">
        {curTab && (
          <>
            <div className="panel-head">
              <h2 className="panel-title">
                {TAB_ICONS[curTab.tab] || '◆'} {curTab.tab}
              </h2>
              {curTab.explainability?.overview && (
                <p className="panel-note">{curTab.explainability.overview}</p>
              )}
            </div>
            <div className="grid">
              {curTab.metrics
                .filter(m => WORKING.has(m.id))
                .map(m => (
                  <MetricCard key={m.id} m={m} isNew={NEW_METRICS.has(m.id)} />
                ))}
            </div>
          </>
        )}
      </main>

      <footer className="foot">
        AI DQM Health Observatory · {liveTabs.length} tabs · {WORKING.size} active metrics
        · auto-refresh {REFRESH_MS / 1000}s
        · source: <span className="foot-src">/api/health-metrics</span>
      </footer>
    </div>
  )
}
