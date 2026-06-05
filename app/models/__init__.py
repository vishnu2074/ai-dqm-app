from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class DataSource(Base):
    __tablename__ = "data_sources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)
    host = Column(String, nullable=True)
    port = Column(Integer, nullable=True)
    database = Column(String, nullable=True)
    username = Column(String, nullable=True)
    encrypted_password = Column(Text, nullable=True)
    connection_string = Column(Text, nullable=True)
    container_name = Column(String, nullable=True)
    ssl_mode = Column(String, nullable=True)
    status = Column(String, default="ACTIVE")
    owner = Column(String, default="USER")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    datasource_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"))
    physical_name = Column(String)
    display_name = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    profiling_runs = relationship("ProfilingRun", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    profiling_baselines = relationship("ProfilingBaseline", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    schema_history_entries = relationship("SchemaHistory", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    dq_rules = relationship("DQRule", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    dq_rule_change_logs = relationship("DQRuleChangeLog", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)
    dq_rule_runs = relationship("DQRuleRun", back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True)

class GlobalContext(Base):
    __tablename__ = "global_context"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)
    active_datasource_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    active_dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ProfilingRun(Base):
    __tablename__ = "profiling_runs"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    rows_processed = Column(Integer)
    delta_rows = Column(Integer, default=0)
    duration_ms = Column(Integer)
    checkpoint_id = Column(String, nullable=True)
    status = Column(String, default="COMPLETED")
    error_message = Column(Text, nullable=True)
    is_full_scan = Column(Boolean, default=False)
    
    # ADDED: Timing columns
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    
    # ADDED: AI-generated dataset description
    ai_summary = Column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="profiling_runs")
    column_profiles = relationship("ColumnProfile", back_populates="profiling_run", cascade="all, delete-orphan", passive_deletes=True)
    quality_checks = relationship("QualityCheck", back_populates="profiling_run", cascade="all, delete-orphan", passive_deletes=True)
    drift_records = relationship("DriftRecord", foreign_keys="DriftRecord.profiling_run_id", back_populates="profiling_run", cascade="all, delete-orphan", passive_deletes=True)
    compared_drift_records = relationship("DriftRecord", foreign_keys="DriftRecord.comparison_run_id", back_populates="comparison_run", cascade="all, delete-orphan", passive_deletes=True)
    schema_history_entries = relationship("SchemaHistory", back_populates="profiling_run", cascade="all, delete-orphan", passive_deletes=True)
    profiling_baselines = relationship("ProfilingBaseline", back_populates="profiling_run", cascade="all, delete-orphan", passive_deletes=True)

class ColumnProfile(Base):
    __tablename__ = "column_profiles"
    id = Column(Integer, primary_key=True, index=True)
    profiling_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"))
    column_name = Column(String)
    data_type = Column(String)

    completeness = Column(Float)
    uniqueness = Column(Float)
    validity = Column(Float)
    consistency = Column(Float)
    accuracy = Column(Float)
    timeliness = Column(Float, nullable=True)
    integrity = Column(Float)

    null_count = Column(Integer)
    distinct_count = Column(Integer)
    min_length = Column(Integer, nullable=True)
    max_length = Column(Integer, nullable=True)

    patterns = Column(JSON, nullable=True)
    status = Column(String)
    health_score = Column(Float)

    # ADDED: AI description and sensitivity classification
    ai_description = Column(Text, nullable=True)
    sensitivity_label = Column(String, nullable=True, default="Public")

    profiling_run = relationship("ProfilingRun", back_populates="column_profiles")

class ProfilingBaseline(Base):
    __tablename__ = "profiling_baselines"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    profiling_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"))
    column_name = Column(String)
    metric_name = Column(String)
    metric_value = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

    dataset = relationship("Dataset", back_populates="profiling_baselines")
    profiling_run = relationship("ProfilingRun", back_populates="profiling_baselines")

class QualityCheck(Base):
    __tablename__ = "temporal_checks"
    id = Column(Integer, primary_key=True, index=True)
    profiling_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"))
    column_name = Column(String)
    check_type = Column(String)
    violation_count = Column(Integer)
    severity = Column(String)
    description = Column(Text)
    status = Column(String, default="open")
    llm_root_cause = Column(Text, nullable=True)
    llm_remediation = Column(Text, nullable=True)
    resolved_by_rule_id = Column(Integer, ForeignKey("dq_rules.id", ondelete="SET NULL"), nullable=True)

    profiling_run = relationship("ProfilingRun", back_populates="quality_checks")

TemporalCheck = QualityCheck

class DriftRecord(Base):
    __tablename__ = "drift_records"
    id = Column(Integer, primary_key=True, index=True)
    profiling_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"))
    column_name = Column(String)
    drift_score = Column(Float)
    drift_type = Column(String)
    comparison_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"), nullable=True)
    
    # ADDED: Timestamp for trend analysis
    created_at = Column(String, nullable=True)

    profiling_run = relationship("ProfilingRun", foreign_keys=[profiling_run_id], back_populates="drift_records")
    comparison_run = relationship("ProfilingRun", foreign_keys=[comparison_run_id], back_populates="compared_drift_records")

class SchemaHistory(Base):
    __tablename__ = "schema_history"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    profiling_run_id = Column(Integer, ForeignKey("profiling_runs.id", ondelete="CASCADE"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    change_type = Column(String)
    column_name = Column(String)
    old_type = Column(String, nullable=True)
    new_type = Column(String, nullable=True)
    impact = Column(String)

    dataset = relationship("Dataset", back_populates="schema_history_entries")
    profiling_run = relationship("ProfilingRun", back_populates="schema_history_entries")

class DQRule(Base):
    __tablename__ = "dq_rules"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    rule_code = Column(String, index=True)
    name = Column(String)
    type = Column(String)
    column = Column(String)
    condition = Column(Text)
    severity = Column(String, default="Medium")
    status = Column(String, default="Active")
    input_mode = Column(String, default="manual", nullable=False)
    nl_text = Column(Text, nullable=True)
    regex_pattern = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    dataset = relationship("Dataset", back_populates="dq_rules")

class DQRuleChangeLog(Base):
    __tablename__ = "dq_rule_history"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    rule_code = Column(String, index=True)
    rule_name = Column(String)
    version = Column(String, default="v1.0")
    changed_by = Column(String, default="Admin")
    change_date = Column(DateTime(timezone=True), server_default=func.now())
    change_type = Column(String, default="Created")
    performance_delta = Column(String, default="N/A")

    dataset = relationship("Dataset", back_populates="dq_rule_change_logs")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_by = Column(String, default="System", nullable=False)
    description = Column(Text, nullable=True)
    parent_version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)

    dataset = relationship("Dataset", back_populates="versions")
    parent = relationship("DatasetVersion", remote_side=[id])

class DQRuleRun(Base):
    __tablename__ = "dq_rule_runs"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    input_version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, default="PREVIEW", nullable=False)
    mode = Column(String, default="flag", nullable=False)
    temp_output_path = Column(String, nullable=True)
    output_version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="dq_rule_runs")
    input_version = relationship("DatasetVersion", foreign_keys=[input_version_id])
    output_version = relationship("DatasetVersion", foreign_keys=[output_version_id])
    results = relationship("DQRuleRunResult", back_populates="run", cascade="all, delete-orphan", passive_deletes=True)

class DQRuleRunResult(Base):
    __tablename__ = "dq_rule_run_results"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("dq_rule_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_code = Column(String, nullable=False)
    rule_name = Column(String, nullable=False)
    rule_type = Column(String, nullable=False)
    column = Column(String, nullable=False)
    condition = Column(Text, nullable=True)
    pass_rate = Column(Float, default=0.0)
    violation_count = Column(Integer, default=0)
    samples_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    run = relationship("DQRuleRun", back_populates="results")

class LineageEdge(Base):
    """User-defined dataset to dataset lineage connections."""
    __tablename__ = "lineage_edges"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False, index=True)
    target = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class KnowledgeGraphEdge(Base):
    __tablename__ = "knowledge_graph_edges"
    id = Column(Integer, primary_key=True, index=True)
    source_dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    source_column = Column(String, nullable=False)
    source_dataset_name = Column(String, nullable=True)
    target_dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    target_column = Column(String, nullable=False)
    target_dataset_name = Column(String, nullable=True)
    relationship_type = Column(String, nullable=False, default="foreign_key")
    cardinality = Column(String, nullable=True)
    name_similarity = Column(Float, nullable=True)
    value_overlap = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    method = Column(String, nullable=False, default="auto")
    llm_explanation = Column(Text, nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    invalidated = Column(Boolean, default=False)

class NotificationInbox(Base):
    """Unified notification inbox for all system events."""
    __tablename__ = "notification_inbox"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, nullable=True, index=True)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, default="ALERT", nullable=False)
    category = Column(String, default="System")
    severity = Column(String, default="info")
    
    # NOTE: The actual DB table has 'dataset' TEXT column
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    dataset = Column(String, nullable=True)  # Actual column in DB

    link = Column(String, nullable=True)
    source = Column(String, nullable=True)
    view_route = Column(String, nullable=True)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    ds = relationship("Dataset")

class NotificationPreference(Base):
    __tablename__ = "notification_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    event_type = Column(String)
    enabled = Column(Boolean, default=True)
    channel = Column(String, default="in_app")

class GovernanceNotification(Base):
    __tablename__ = "governance_notifications"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    enabled = Column(Boolean, default=True)
    channel = Column(String, default="in_app")

# ADDED: Governance Audit Log model
class GovernanceAuditLog(Base):
    """Audit trail for governance actions."""
    __tablename__ = "governance_audit_log"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    details = Column(Text, nullable=True)
    dataset_id = Column(Integer, nullable=True)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)