"""
migrate_db.py
=============
Run this ONCE after deploying to Render to copy your existing ai_dqm.db
to the persistent disk at /var/data/ai_dqm.db.

On Render: add this as a one-time job, or call it from your startup script.

Locally you won't need this — the router finds ai_dqm.db at project root.
"""

import os
import shutil

SRC  = os.path.join(os.path.dirname(__file__), "ai_dqm.db")   # project root
DEST = "/var/data/ai_dqm.db"                                    # Render persistent disk

if os.path.exists(DEST):
    print(f"DB already at {DEST} — skipping copy.")
elif os.path.exists(SRC):
    os.makedirs("/var/data", exist_ok=True)
    shutil.copy2(SRC, DEST)
    print(f"Copied {SRC} → {DEST}  ({os.path.getsize(DEST):,} bytes)")
else:
    print(f"WARNING: source DB not found at {SRC}. A fresh DB will be created on first run.")
