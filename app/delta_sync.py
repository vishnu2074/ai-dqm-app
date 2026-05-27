# python-backend/app/delta_sync.py
"""
Delta mirror — called AFTER successful SQLite commits.
Each function takes the SQLite ORM object that was just committed
and writes the equivalent row to the matching Delta table.

All functions are fire-and-forget: they log failures but never
raise exceptions, so Delta errors never break the main app flow.

Usage (in a router after db.commit()):
    from app.delta_sync import sync_datasource, sync_dataset, sync_profiling_run
    sync_datasource(datasource_orm_object)
"""
from __future__ import annotations

from typing import Optional


def _exec(sql: str, params: list) -> None:
    try:
        from app.database_delta import delta_execute
        delta_execute(sql, params)
    except Exception as e:
        print(f"[delta_sync] write failed: {e}")


def sync_datasource(obj) -> None:
    """Mirror a DataSource row to Delta datasources table."""
    _exec(
        """INSERT INTO datasources
           (name, type, host, port, database, username, encrypted_password,
            connection_string, container_name, ssl_mode, status, owner)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            obj.name, obj.type, obj.host, obj.port, obj.database,
            obj.username, obj.encrypted_password, obj.connection_string,
            obj.container_name, obj.ssl_mode, obj.status, obj.owner,
        ]
    )


def sync_dataset(obj) -> None:
    """Mirror a Dataset row to Delta datasets table."""
    _exec(
        "INSERT INTO datasets (datasource_id, physical_name, display_name) VALUES (?, ?, ?)",
        [obj.datasource_id, obj.physical_name, obj.display_name]
    )


def sync_profiling_run(obj) -> None:
    """Mirror a ProfilingRun row to Delta profiling_runs table."""
    _exec(
        """INSERT INTO profiling_runs
           (dataset_id, status, run_type, total_rows, delta_rows,
            duration_seconds, started_at, completed_at, error_message, checkpoint_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            obj.dataset_id,
            obj.status or "COMPLETED",
            "FULL" if getattr(obj, "is_full_scan", False) else "INCREMENTAL",
            obj.rows_processed,
            obj.delta_rows,
            round(obj.duration_ms / 1000, 3) if obj.duration_ms else None,
            str(obj.timestamp) if obj.timestamp else None,
            str(obj.timestamp) if obj.timestamp else None,
            obj.error_message,
            obj.checkpoint_id,
        ]
    )


def sync_column_profile(obj, dataset_id: int) -> None:
    """Mirror a ColumnProfile row to Delta column_profiles table."""
    _exec(
        """INSERT INTO column_profiles
           (profiling_run_id, dataset_id, column_name, data_type,
            null_count, distinct_count, completeness, uniqueness, health_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            obj.profiling_run_id, dataset_id, obj.column_name, obj.data_type,
            obj.null_count, obj.distinct_count,
            obj.completeness, obj.uniqueness, obj.health_score,
        ]
    )


def sync_quality_check(obj, dataset_id: int = None) -> None:
    """Mirror a QualityCheck (temporal_check) row to Delta quality_checks table."""
    # dataset_id is not on the model directly — it comes through profiling_run
    _exec(
        """INSERT INTO quality_checks
           (dataset_id, profiling_run_id, check_type, column_name,
            status, severity, violation_count, message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            dataset_id,
            obj.profiling_run_id, obj.check_type, obj.column_name,
            obj.status, obj.severity, obj.violation_count,
            getattr(obj, "description", None),
        ]
    )


def sync_dq_rule(obj) -> None:
    """Mirror a DQRule row to Delta dq_rules table."""
    _exec(
        """INSERT INTO dq_rules
           (dataset_id, name, type, column_name, condition,
            severity, status, input_mode, nl_text, regex_pattern)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            obj.dataset_id, obj.name, obj.type, obj.column, obj.condition,
            obj.severity, obj.status, obj.input_mode, obj.nl_text, obj.regex_pattern,
        ]
    )


def sync_drift_record(obj) -> None:
    """Mirror a DriftRecord row to Delta drift_records table."""
    _exec(
        """INSERT INTO drift_records
           (dataset_id, profiling_run_id, column_name, drift_type, old_value, new_value)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            getattr(obj, "dataset_id", None),
            obj.profiling_run_id, obj.column_name, obj.drift_type,
            str(getattr(obj, "drift_score", "")),
            str(getattr(obj, "drift_score", "")),
        ]
    )


def sync_alert(obj) -> None:
    """Mirror an Alert row to Delta alerts table."""
    _exec(
        """INSERT INTO alerts
           (dataset_id, title, message, category, severity, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            obj.dataset_id, obj.title, obj.message,
            obj.category, obj.severity,
            getattr(obj, "status", "open"),
        ]
    )