import sqlite3
from pathlib import Path
import json

DB_PATH = "ai_dqm.db"

if not Path(DB_PATH).exists():
    print(f"ERROR: {DB_PATH} not found")
    exit(1)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def table_exists(table_name):
    cur.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name=?
    """, (table_name,))
    return cur.fetchone() is not None


def print_schema(table_name):
    print_header(f"SCHEMA :: {table_name}")

    if not table_exists(table_name):
        print("TABLE NOT FOUND")
        return

    cur.execute(f"PRAGMA table_info({table_name})")

    rows = cur.fetchall()

    if not rows:
        print("NO COLUMNS")
        return

    for r in rows:
        print(dict(r))


def run_query(title, query):
    print_header(title)

    try:
        cur.execute(query)
        rows = cur.fetchall()

        if not rows:
            print("NO ROWS")
            return

        for row in rows:
            print(dict(row))

    except Exception as e:
        print(f"ERROR: {e}")


# ==================================================================================
# FULL TABLE LIST
# ==================================================================================

run_query(
    "ALL TABLES",
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
    """
)

# ==================================================================================
# CRITICAL SCHEMAS
# ==================================================================================

print_schema("profiling_runs")
print_schema("drift_records")
print_schema("temporal_checks")

# ==================================================================================
# IMPORTANT SCHEMAS
# ==================================================================================

print_schema("governance_notifications")
print_schema("notification_inbox")
print_schema("knowledge_graph_edges")

# ==================================================================================
# STATUS DISTRIBUTIONS
# ==================================================================================

run_query(
    "TEMPORAL_CHECK STATUS DISTRIBUTION",
    """
    SELECT status, COUNT(*) as count
    FROM temporal_checks
    GROUP BY status
    """
)

run_query(
    "DRIFT_RECORDS SEVERITY DISTRIBUTION",
    """
    SELECT severity, COUNT(*) as count
    FROM drift_records
    GROUP BY severity
    """
)

run_query(
    "PROFILING_RUNS STATUS DISTRIBUTION",
    """
    SELECT status, COUNT(*) as count
    FROM profiling_runs
    GROUP BY status
    """
)

# ==================================================================================
# GOVERNANCE POLICY SOURCES
# ==================================================================================

if table_exists("governance_policies"):

    run_query(
        "GOVERNANCE_POLICY SOURCES",
        """
        SELECT source, COUNT(*) as count
        FROM governance_policies
        GROUP BY source
        """
    )

elif table_exists("ai_policies"):

    run_query(
        "AI_POLICY SOURCES",
        """
        SELECT source, COUNT(*) as count
        FROM ai_policies
        GROUP BY source
        """
    )

else:

    print_header("POLICY TABLE")
    print("Neither governance_policies nor ai_policies exists")

# ==================================================================================
# SAMPLE ROWS
# ==================================================================================

for table in [
    "profiling_runs",
    "drift_records",
    "temporal_checks",
    "governance_notifications",
    "notification_inbox",
    "knowledge_graph_edges"
]:

    if table_exists(table):

        run_query(
            f"SAMPLE ROWS :: {table}",
            f"""
            SELECT *
            FROM {table}
            LIMIT 5
            """
        )

# ==================================================================================
# FOREIGN KEYS
# ==================================================================================

for table in [
    "profiling_runs",
    "drift_records",
    "temporal_checks",
    "governance_notifications",
    "notification_inbox",
    "knowledge_graph_edges"
]:

    if table_exists(table):

        print_header(f"FOREIGN KEYS :: {table}")

        try:
            cur.execute(f"PRAGMA foreign_key_list({table})")

            rows = cur.fetchall()

            if not rows:
                print("NO FOREIGN KEYS")
            else:
                for r in rows:
                    print(dict(r))

        except Exception as e:
            print(e)

# ==================================================================================
# COLUMN PRESENCE CHECKS
# ==================================================================================

print_header("COLUMN PRESENCE CHECKS")

checks = {
    "profiling_runs": [
        "started_at",
        "completed_at",
        "duration_ms",
        "llm_call_duration_ms",
        "ai_summary",
        "status",
    ],
    "drift_records": [
        "profiling_run_id",
        "created_at",
        "severity",
    ],
    "temporal_checks": [
        "explanation",
        "status",
    ],
    "governance_notifications": [
        "action_taken",
        "enabled",
    ]
}

for table, columns in checks.items():

    if not table_exists(table):
        print(f"{table}: TABLE MISSING")
        continue

    cur.execute(f"PRAGMA table_info({table})")

    existing = {r["name"] for r in cur.fetchall()}

    print(f"\n{table}")

    for col in columns:
        print(
            f"  {col:<30} "
            f"{'YES' if col in existing else 'NO'}"
        )


# ==================================================================================
# DB INFO
# ==================================================================================

print_header("DATABASE INFO")

cur.execute("SELECT sqlite_version()")

print("SQLite Version:", cur.fetchone()[0])

print("Database File:", Path(DB_PATH).absolute())

print("Database Size:",
      f"{Path(DB_PATH).stat().st_size:,} bytes")

conn.close()

print("\nDONE")