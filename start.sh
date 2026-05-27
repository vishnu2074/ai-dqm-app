#!/usr/bin/env bash
# start.sh — Render startup script
# Set this as the startCommand in render.yaml instead of uvicorn directly.
# It copies the DB to persistent disk on first boot, then starts the server.

set -e

echo "=== AI DQM Startup ==="

# Copy DB to persistent disk if not already there
if [ ! -f "/var/data/ai_dqm.db" ]; then
    if [ -f "ai_dqm.db" ]; then
        echo "Copying ai_dqm.db to /var/data/ai_dqm.db ..."
        cp ai_dqm.db /var/data/ai_dqm.db
        echo "Done."
    else
        echo "No existing DB found — fresh DB will be created at /var/data/ai_dqm.db"
    fi
else
    echo "DB already at /var/data/ai_dqm.db"
fi

export AIDQM_DB_PATH=/var/data/ai_dqm.db

echo "Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
