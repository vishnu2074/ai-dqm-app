"""
Unified Notification Inbox — production-ready
Supports type: ALERT | ANOMALY | INCIDENT
Single inbox endpoint consumed by the frontend bell + notifications page.

FIXED: All SQL queries now use `created_at` instead of `timestamp` column.
The `timestamp` column has been removed from all operations to prevent
"no such column: timestamp" errors.
"""
from __future__ import annotations

import logging
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

notification_inbox_router = APIRouter()
logger = logging.getLogger(__name__)

VALID_TYPES      = {"ALERT", "ANOMALY", "INCIDENT"}
VALID_SEVERITIES = {"critical", "warning", "info", "low"}

# ─── DB bootstrap ─────────────────────────────────────────────────────────────

def _ensure_inbox_table() -> None:
    try:
        from app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS notification_inbox (
                    id          TEXT PRIMARY KEY,
                    user_email  TEXT,
                    title       TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    type        TEXT NOT NULL DEFAULT 'ALERT',
                    category    TEXT DEFAULT 'System',
                    severity    TEXT DEFAULT 'info',
                    source      TEXT,
                    view_route  TEXT,
                    created_at  TEXT NOT NULL,
                    is_read     INTEGER DEFAULT 0,
                    is_archived INTEGER DEFAULT 0,
                    link        TEXT,
                    dataset     TEXT
                )
            """))
            # Migration: add columns that may be missing in existing deployments
            # FIXED: Removed 'timestamp' column - using only 'created_at' now
            for col_def in [
                "ALTER TABLE notification_inbox ADD COLUMN type TEXT NOT NULL DEFAULT 'ALERT'",
                "ALTER TABLE notification_inbox ADD COLUMN source TEXT",
                "ALTER TABLE notification_inbox ADD COLUMN view_route TEXT",
                "ALTER TABLE notification_inbox ADD COLUMN created_at TEXT",
                "ALTER TABLE notification_inbox ADD COLUMN is_read INTEGER DEFAULT 0",
                "ALTER TABLE notification_inbox ADD COLUMN is_archived INTEGER DEFAULT 0",
                "ALTER TABLE notification_inbox ADD COLUMN link TEXT",
                "ALTER TABLE notification_inbox ADD COLUMN dataset TEXT",
            ]:
                try:
                    conn.execute(text(col_def))
                except Exception:
                    pass  # column already exists
            conn.commit()
    except Exception as e:
        logger.warning("[notif] Could not ensure inbox table: %s", e)


_ensure_inbox_table()

# ─── DB helpers ───────────────────────────────────────────────────────────────

def _db_insert_notification(entry: dict) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text

        now_iso = datetime.utcnow().isoformat()

        with engine.connect() as conn:
            # FIXED: Removed 'timestamp' column - using only 'created_at'
            conn.execute(text("""
                INSERT OR REPLACE INTO notification_inbox
                (id, user_email, title, message, type, category, severity,
                 source, view_route, created_at, is_read, is_archived, link, dataset)
                VALUES
                (:id, :user_email, :title, :message, :type, :category, :severity,
                 :source, :view_route, :created_at, :is_read, :is_archived, :link, :dataset)
            """), {
                "id":           entry["id"],
                "user_email":   entry.get("user_email"),
                "title":        entry["title"],
                "message":      entry["message"],
                "type":         entry.get("type", "ALERT"),
                "category":     entry.get("category", "System"),
                "severity":     entry.get("severity", "info"),
                "source":       entry.get("source"),
                "view_route":   entry.get("view_route"),
                "created_at":   entry.get("created_at", now_iso),
                "is_read":      0,
                "is_archived":  0,
                "link":         entry.get("link"),
                "dataset":      entry.get("dataset"),
            })
            conn.commit()
    except Exception as e:
        logger.error("[notif] DB insert failed: %s", e)


def _db_get_inbox(
    user: Optional[str] = None,
    limit: int = 100,
    notif_type: Optional[str] = None,
) -> list:
    try:
        from app.database import engine
        from sqlalchemy import text

        where_parts = ["is_archived = 0"]
        params: dict = {"limit": limit}

        if user:
            where_parts.append("(user_email IS NULL OR user_email = :user)")
            params["user"] = user

        if notif_type and notif_type.upper() in VALID_TYPES:
            where_parts.append("type = :type")
            params["type"] = notif_type.upper()

        where_clause = " AND ".join(where_parts)

        with engine.connect() as conn:
            # FIXED: Changed 'timestamp' to 'created_at' in SELECT and ORDER BY
            rows = conn.execute(text(f"""
                SELECT id, user_email, title, message, type, category, severity,
                       source, view_route, created_at, is_read, is_archived, link, dataset
                FROM notification_inbox
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
            """), params).fetchall()

        return [{
            "id":         r[0],
            "user_email": r[1],
            "title":      r[2],
            "message":    r[3],
            "type":       (r[4] or "ALERT").upper(),
            "category":   r[5] or "System",
            "severity":   r[6] or "info",
            "source":     r[7] or "System",
            "view_route": r[8],
            "timestamp":  r[9],  # Keep as 'timestamp' in response for frontend compatibility
            "time_ago":   _time_ago_from_iso(r[9]),
            "read":       bool(r[10]),
            "archived":   bool(r[11]),
            "link":       r[12],
            "dataset":    r[13],
            "is_new":     not bool(r[10]),
        } for r in rows]
    except Exception as e:
        logger.error("[notif] DB get inbox failed: %s", e)
        return []


def _time_ago_from_iso(ts_str: Optional[str]) -> str:
    if not ts_str:
        return "now"
    try:
        ts   = datetime.fromisoformat(ts_str)
        diff = int((datetime.utcnow() - ts).total_seconds())
        if diff < 60:    return "just now"
        if diff < 3600:  return f"{diff // 60}m ago"
        if diff < 86400: return f"{diff // 3600}h ago"
        return f"{diff // 86400}d ago"
    except Exception:
        return "now"


def _db_mark_read(notif_id: str) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("UPDATE notification_inbox SET is_read = 1 WHERE id = :id"), {"id": notif_id})
            conn.commit()
    except Exception as e:
        logger.error("[notif] DB mark read failed: %s", e)


def _db_mark_all_read(user: Optional[str] = None) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            if user:
                conn.execute(text("UPDATE notification_inbox SET is_read = 1 WHERE user_email = :user OR user_email IS NULL"), {"user": user})
            else:
                conn.execute(text("UPDATE notification_inbox SET is_read = 1"))
            conn.commit()
    except Exception as e:
        logger.error("[notif] DB mark all read failed: %s", e)


def _db_archive(notif_id: str) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("UPDATE notification_inbox SET is_archived = 1, is_read = 1 WHERE id = :id"), {"id": notif_id})
            conn.commit()
    except Exception as e:
        logger.error("[notif] DB archive failed: %s", e)


def _db_delete(notif_id: str) -> None:
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM notification_inbox WHERE id = :id"), {"id": notif_id})
            conn.commit()
    except Exception as e:
        logger.error("[notif] DB delete failed: %s", e)


def _db_get_one(notif_id: str) -> Optional[dict]:
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            # FIXED: Changed 'timestamp' to 'created_at' in SELECT
            row = conn.execute(text("""
                SELECT id, user_email, title, message, type, category, severity,
                       source, view_route, created_at, is_read, is_archived, link, dataset
                FROM notification_inbox WHERE id = :id
            """), {"id": notif_id}).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "user_email": row[1], "title": row[2], "message": row[3],
                "type": (row[4] or "ALERT").upper(), "category": row[5], "severity": row[6],
                "source": row[7], "view_route": row[8], "timestamp": row[9],  # Keep as 'timestamp' for frontend
                "read": bool(row[10]), "archived": bool(row[11]),
                "link": row[12], "dataset": row[13],
            }
    except Exception as e:
        logger.error("[notif] DB get one failed: %s", e)
        return None

# ─── System config ─────────────────────────────────────────────────────────────

def _get_system_config() -> dict:
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            rows = db.execute(text("SELECT key, value FROM governance_system_config")).fetchall()
            cfg  = {r[0]: r[1] for r in rows}
            return {
                "email_smtp_host":     cfg.get("email_smtp_host", "").strip(),
                "email_smtp_port":     int(cfg.get("email_smtp_port", 587)),
                "email_smtp_from":     cfg.get("email_smtp_from", "").strip(),
                "email_smtp_user":     cfg.get("email_smtp_user", "").strip(),
                "email_smtp_password": cfg.get("email_smtp_password", "").strip(),
                "slack_webhook_url":   cfg.get("slack_webhook_url", "").strip(),
                "sendgrid_api_key":    cfg.get("sendgrid_api_key", "").strip(),
                "mailjet_api_key":     cfg.get("mailjet_api_key", "").strip(),
                "mailjet_secret_key":  cfg.get("mailjet_secret_key", "").strip(),
            }
    except Exception as e:
        logger.warning("[notif] Could not read system config: %s", e)
        return {}

# ─── Channel preference ────────────────────────────────────────────────────────

_CATEGORY_TO_NOTIF_ID: dict = {
    "policy": "n_policy", "user": "n_user", "quality": "n_quality",
    "anomaly": "n_anomaly", "rule": "n_rule", "schema": "n_schema",
    "compliance": "n_compliance", "datasource": "n_datasource",
    "dq": "n_quality", "ai": "n_quality", "dataset": "n_datasource",
    "system": None, "incident": None,
}


def _ensure_governance_notifications_columns() -> None:
    """Add missing columns to governance_notifications if they don't exist yet."""
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            for col_def in [
                "ALTER TABLE governance_notifications ADD COLUMN recipient_email TEXT DEFAULT ''",
                "ALTER TABLE governance_notifications ADD COLUMN slack_webhook TEXT DEFAULT ''",
            ]:
                try:
                    conn.execute(text(col_def))
                    conn.commit()
                except Exception:
                    pass  # Column already exists — safe to ignore
    except Exception as e:
        logger.debug("[notif] governance_notifications column migration: %s", e)

_ensure_governance_notifications_columns()


def _get_channel_pref_from_db(category: str) -> dict:
    default = {"channel": "in_app", "enabled": True, "recipient_email": "", "slack_webhook": ""}
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        notif_id = _CATEGORY_TO_NOTIF_ID.get((category or "").lower())
        if not notif_id:
            return default
        with SessionLocal() as db:
            # Level 1: Full query with all optional columns
            try:
                row = db.execute(text(
                    "SELECT enabled, channel, recipient_email, slack_webhook "
                    "FROM governance_notifications WHERE id = :id"
                ), {"id": notif_id}).fetchone()
                if row:
                    return {"enabled": bool(row[0]), "channel": row[1] or "in_app",
                            "recipient_email": row[2] or "", "slack_webhook": row[3] or ""}
                return default
            except Exception:
                pass

            # Level 2: Without slack_webhook
            try:
                row = db.execute(text(
                    "SELECT enabled, channel, recipient_email "
                    "FROM governance_notifications WHERE id = :id"
                ), {"id": notif_id}).fetchone()
                if row:
                    return {"enabled": bool(row[0]), "channel": row[1] or "in_app",
                            "recipient_email": row[2] or "", "slack_webhook": ""}
                return default
            except Exception:
                pass

            # Level 3: Minimal — just enabled + channel (always safe)
            try:
                row = db.execute(text(
                    "SELECT enabled, channel "
                    "FROM governance_notifications WHERE id = :id"
                ), {"id": notif_id}).fetchone()
                if row:
                    return {"enabled": bool(row[0]), "channel": row[1] or "in_app",
                            "recipient_email": "", "slack_webhook": ""}
            except Exception:
                pass

        return default
    except Exception as e:
        logger.warning("[notif] Could not read channel pref for '%s': %s", category, e)
        return default

# ─── SMTP validation ───────────────────────────────────────────────────────────

def _validate_smtp_config(host: str, from_addr: str, to_address: str) -> Optional[str]:
    if not host:
        return "SMTP host is not configured. Go to Governance → Settings → System Config → Integrations."
    if "@" in host:
        return (
            f"SMTP host '{host}' looks like an email address, not a hostname. "
            f"Enter the SMTP server hostname (e.g. smtp.gmail.com, smtp.sendgrid.net) "
            f"in the SMTP Host field. Your email address goes in the 'From Address' field."
        )
    if host.startswith("http://") or host.startswith("https://"):
        return f"Remove the 'http://' prefix. SMTP Host should be just the hostname, e.g. smtp.gmail.com"
    if not from_addr:
        return "SMTP From Address is not configured. Go to Governance → Settings → System Config → Integrations."
    if not to_address:
        return "No recipient email address provided."
    return None

# ─── Email helpers ─────────────────────────────────────────────────────────────

def _send_via_mailjet(
    to_address: str, subject: str, body_text: str, body_html: str,
    system_config: dict,
) -> tuple[bool, str]:
    """
    Send email via Mailjet HTTP API.
    Requires MAILJET_API_KEY and MAILJET_SECRET_KEY env vars OR
    stored in governance_system_config as mailjet_api_key / mailjet_secret_key.
    Free tier: 200 emails/day. No SMTP needed, no App Password issues.
    """
    import json as _json, urllib.request, os
    api_key    = (system_config.get("mailjet_api_key") or os.getenv("MAILJET_API_KEY", "")).strip()
    secret_key = (system_config.get("mailjet_secret_key") or os.getenv("MAILJET_SECRET_KEY", "")).strip()
    from_addr  = system_config.get("email_smtp_from", "").strip()

    if not api_key or not secret_key or not from_addr:
        return False, "Mailjet not configured (need mailjet_api_key, mailjet_secret_key, and email_smtp_from)."

    import base64
    credentials = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()

    payload = _json.dumps({
        "Messages": [{
            "From":     {"Email": from_addr, "Name": "AI DQM"},
            "To":       [{"Email": to_address}],
            "Subject":  subject,
            "TextPart": body_text,
            "HTMLPart": body_html,
        }]
    }).encode()

    try:
        req = urllib.request.Request(
            "https://api.mailjet.com/v3.1/send",
            data=payload,
            headers={
                "Authorization":  f"Basic {credentials}",
                "Content-Type":   "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read().decode())
            if result.get("Messages") and result["Messages"][0].get("Status") == "success":
                logger.info("[notif] Mailjet sent → %s: %s", to_address, subject)
                return True, ""
            return False, f"Mailjet API error: {result}"
    except Exception as e:
        logger.error("[notif] Mailjet failed: %s", e)
        return False, f"Mailjet send failed: {e}"


def _send_via_sendgrid(
    to_address: str, subject: str, body_text: str, system_config: dict
) -> tuple[bool, str]:
    """Send email via SendGrid HTTP API."""
    import json as _json, urllib.request
    api_key   = system_config.get("sendgrid_api_key", "").strip()
    from_addr = system_config.get("email_smtp_from", "").strip()
    if not api_key or not from_addr:
        return False, "SendGrid not configured (need sendgrid_api_key and email_smtp_from)."
    payload = _json.dumps({
        "personalizations": [{"to": [{"email": to_address}]}],
        "from":    {"email": from_addr, "name": "AI DQM"},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body_text}],
    }).encode()
    try:
        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 202):
                logger.info("[notif] SendGrid sent → %s: %s", to_address, subject)
                return True, ""
        return False, "SendGrid API returned unexpected status."
    except Exception as e:
        logger.error("[notif] SendGrid failed: %s", e)
        return False, f"SendGrid send failed: {e}"


def _send_email_notif(
    to_address: str, title: str, message: str,
    link: Optional[str], system_config: dict
) -> tuple[bool, str]:
    host      = system_config.get("email_smtp_host", "").strip()
    port      = int(system_config.get("email_smtp_port", 587))
    from_addr = system_config.get("email_smtp_from", "").strip()
    smtp_user = system_config.get("email_smtp_user", "").strip()
    smtp_pass = system_config.get("email_smtp_password", "").strip()

    config_error = _validate_smtp_config(host, from_addr, to_address)
    if config_error:
        logger.warning("[notif] SMTP config error: %s", config_error)
        return False, config_error

    link_html = ""
    if link:
        link_html = (
            f'<p style="margin-top:16px;">'
            f'<a href="http://localhost:5173{link}" style="background:#7c3aed;color:white;'
            f'padding:8px 16px;border-radius:6px;text-decoration:none;font-size:14px;">'
            f"View in AI DQM →</a></p>"
        )

    body_html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:24px;">
      <div style="background:linear-gradient(to right,#7c3aed,#3b82f6);padding:20px 24px;border-radius:8px 8px 0 0;">
        <h1 style="color:white;font-size:18px;margin:0;">AI for DQM</h1>
      </div>
      <div style="border:1px solid #e5e7eb;border-top:none;padding:24px;border-radius:0 0 8px 8px;background:#fff;">
        <h2 style="color:#1f2937;font-size:16px;margin-top:0;">{title}</h2>
        <p style="color:#4b5563;font-size:14px;line-height:1.6;">{message}</p>
        {link_html}
        <hr style="margin:24px 0;border:none;border-top:1px solid #f3f4f6;">
        <p style="color:#9ca3af;font-size:12px;">
          You received this because email alerts are enabled in AI DQM Governance Settings.
        </p>
      </div>
    </div>"""

    msg             = MIMEMultipart("alternative")
    msg["Subject"]  = f"[AI DQM] {title}"
    msg["From"]     = from_addr
    msg["To"]       = to_address
    msg.attach(MIMEText(message, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        # Office365 / Outlook: must use STARTTLS on port 587
        # The smtp_user MUST be the full email address (not display name)
        if "office365" in host.lower() or "outlook" in host.lower():
            import ssl
            ctx = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo(host)
                server.starttls(context=ctx)
                server.ehlo(host)
                # Use email address as username, not display name
                login_user = smtp_user if "@" in smtp_user else from_addr
                if login_user and smtp_pass:
                    server.login(login_user, smtp_pass)
                server.sendmail(from_addr, [to_address], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                if port != 465:
                    server.starttls()
                    server.ehlo()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, [to_address], msg.as_string())
        logger.info("[notif] Email sent → %s: %s", to_address, title)
        return True, ""
    except smtplib.SMTPAuthenticationError:
        msg_err = "SMTP authentication failed. Check your SMTP username and password/API key in System Config."
        logger.error("[notif] %s", msg_err)
        return False, msg_err
    except smtplib.SMTPConnectError:
        msg_err = f"Could not connect to SMTP server '{host}:{port}'. Check the host and port in System Config."
        logger.error("[notif] %s", msg_err)
        return False, msg_err
    except smtplib.SMTPRecipientsRefused:
        msg_err = f"Recipient '{to_address}' was refused by the SMTP server."
        logger.error("[notif] %s", msg_err)
        return False, msg_err
    except smtplib.SMTPSenderRefused:
        msg_err = f"From address '{from_addr}' was refused by the SMTP server."
        logger.error("[notif] %s", msg_err)
        return False, msg_err
    except OSError as e:
        msg_err = f"Network error connecting to '{host}:{port}': {e}. Is the SMTP host correct?"
        logger.error("[notif] %s", msg_err)
        return False, msg_err
    except Exception as e:
        msg_err = f"Email send failed: {e}"
        logger.error("[notif] %s", msg_err)
        return False, msg_err


def _send_slack_notif(
    title: str, message: str, severity: str,
    link: Optional[str], system_config: dict
) -> bool:
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
        logger.error("[notif] Slack failed: %s", e)
        return False

# ─── Core push helper ─────────────────────────────────────────────────────────

def _push_in_app(
    title: str, message: str, notif_type: str, category: str, severity: str,
    link: Optional[str] = None, dataset: Optional[str] = None,
    user_email: Optional[str] = None, source: Optional[str] = None,
    view_route: Optional[str] = None,
) -> dict:
    norm_type = notif_type.upper() if notif_type.upper() in VALID_TYPES else "ALERT"
    now_iso   = datetime.utcnow().isoformat()
    # FIXED: Removed 'timestamp' key - using only 'created_at'
    entry = {
        "id":         f"ni_{uuid.uuid4().hex[:10]}",
        "user_email": user_email,
        "title":      title,
        "message":    message,
        "type":       norm_type,
        "category":   category,
        "severity":   severity,
        "source":     source or category,
        "view_route": view_route or link,
        "created_at": now_iso,
        "read":       False,
        "link":       link,
        "dataset":    dataset,
    }
    _db_insert_notification(entry)
    logger.info("[notif] Pushed [%s/%s] %s", norm_type, severity.upper(), title)
    return entry


def create_inbox_notification(
    title: str, message: str, category: str, severity: str,
    link: Optional[str] = None, dataset: Optional[str] = None,
    user_email: Optional[str] = None, notif_type: str = "ALERT",
    source: Optional[str] = None, view_route: Optional[str] = None,
) -> Optional[dict]:
    try:
        entry = _push_in_app(
            title=title, message=message, notif_type=notif_type,
            category=category, severity=severity, link=link,
            dataset=dataset, user_email=user_email,
            source=source, view_route=view_route,
        )
        pref    = _get_channel_pref_from_db(category)
        cfg     = _get_system_config()

        if not pref.get("enabled", True):
            return entry

        channel = pref.get("channel", "in_app")

        if channel == "email":
            recipient = (user_email or pref.get("recipient_email") or "").strip()
            if recipient:
                # Priority chain: Mailjet → SendGrid → SMTP
                _sent = False
                if cfg.get("mailjet_api_key") and cfg.get("mailjet_secret_key"):
                    _ok, _ = _send_via_mailjet(recipient, f"[AI DQM] {title}", message, f"<p>{message}</p>", cfg)
                    _sent = _ok
                if not _sent and cfg.get("sendgrid_api_key"):
                    _ok, _ = _send_via_sendgrid(recipient, f"[AI DQM] {title}", message, cfg)
                    _sent = _ok
                if not _sent:
                    _send_email_notif(recipient, title, message, link, cfg)
        elif channel == "slack":
            webhook_override = pref.get("slack_webhook", "").strip()
            if webhook_override:
                cfg = {**cfg, "slack_webhook_url": webhook_override}
            _send_slack_notif(title, message, severity, link, cfg)

        return entry
    except Exception as e:
        logger.error("[notif] create_inbox_notification error: %s", e)
        return None

# ─── REST endpoints ───────────────────────────────────────────────────────────

@notification_inbox_router.get("/governance/notifications/inbox")
def get_inbox(
    user: Optional[str] = None,
    limit: int = 100,
    type: Optional[str] = None,
):
    try:
        items  = _db_get_inbox(user=user, limit=limit, notif_type=type)
        unread = sum(1 for n in items if not n.get("read", False))
        return {"notifications": items, "unread_count": unread, "total": len(items)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.get("/governance/notifications/inbox/count")
def get_unread_count(user: Optional[str] = None):
    try:
        items  = _db_get_inbox(user=user, limit=500)
        unread = sum(1 for n in items if not n.get("read", False))
        return {"unread": unread}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.get("/governance/notifications")
def get_notification_prefs():
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            try:
                rows = db.execute(text(
                    "SELECT id, title, description, enabled, channel, "
                    "recipient_email, slack_webhook "
                    "FROM governance_notifications ORDER BY id"
                )).fetchall()
                return [{"id": r[0], "title": r[1], "description": r[2],
                         "enabled": bool(r[3]), "channel": r[4] or "in_app",
                         "recipient_email": r[5] or "", "slack_webhook": r[6] or ""}
                        for r in rows]
            except Exception:
                rows = db.execute(text(
                    "SELECT id, title, description, enabled, channel, recipient_email "
                    "FROM governance_notifications ORDER BY id"
                )).fetchall()
                return [{"id": r[0], "title": r[1], "description": r[2],
                         "enabled": bool(r[3]), "channel": r[4] or "in_app",
                         "recipient_email": r[5] or "", "slack_webhook": ""}
                        for r in rows]
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.put("/governance/notifications")
async def save_notification_prefs(request: Request):
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        try:
            prefs = await request.json()
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {e}"})

        if not isinstance(prefs, list):
            return JSONResponse(status_code=400, content={"detail": "Body must be a JSON array"})

        updated_count = 0
        with SessionLocal() as db:
            for pref in prefs:
                if not isinstance(pref, dict):
                    continue
                notif_id        = str(pref.get("id") or "").strip()
                if not notif_id:
                    continue
                enabled         = bool(pref.get("enabled", True))
                channel         = str(pref.get("channel") or "in_app").strip()
                recipient_email = str(pref.get("recipient_email") or "").strip()
                slack_webhook   = str(pref.get("slack_webhook") or "").strip()
                if channel not in ("email", "slack", "in_app"):
                    channel = "in_app"
                try:
                    result = db.execute(text("""
                        UPDATE governance_notifications
                        SET enabled=:enabled, channel=:channel,
                            recipient_email=:recipient_email, slack_webhook=:slack_webhook
                        WHERE id=:id
                    """), {"enabled": 1 if enabled else 0, "channel": channel,
                           "recipient_email": recipient_email, "slack_webhook": slack_webhook,
                           "id": notif_id})
                    if result.rowcount > 0:
                        updated_count += 1
                except Exception:
                    db.execute(text("""
                        UPDATE governance_notifications
                        SET enabled=:enabled, channel=:channel, recipient_email=:recipient_email
                        WHERE id=:id
                    """), {"enabled": 1 if enabled else 0, "channel": channel,
                           "recipient_email": recipient_email, "id": notif_id})
            db.commit()

        create_inbox_notification(
            title="Notification Preferences Saved",
            message=f"{updated_count} setting(s) updated successfully.",
            category="System", severity="info",
            link="/settings?tab=notifications", notif_type="ALERT", source="Settings",
        )
        return {"status": "ok", "updated": updated_count}
    except Exception as e:
        logger.error("[notif] save_notification_prefs error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.post("/governance/notifications/inbox/{notif_id}/read", status_code=204)
def mark_read(notif_id: str):
    try:
        _db_mark_read(notif_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@notification_inbox_router.post("/governance/notifications/inbox/mark-all-read", status_code=204)
async def mark_all_read(request: Request):
    try:
        try:
            body = await request.json()
            user = body.get("user")
        except Exception:
            user = None
        _db_mark_all_read(user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@notification_inbox_router.post("/governance/notifications/inbox/{notif_id}/archive", status_code=200)
def archive_notification(notif_id: str):
    try:
        _db_archive(notif_id)
        return {"status": "archived"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@notification_inbox_router.post("/governance/notifications/inbox/{notif_id}/send-email", status_code=200)
def send_email_for_notification(notif_id: str, body: dict = {}):
    try:
        notif = _db_get_one(notif_id)
        if not notif:
            raise HTTPException(status_code=404, detail="Notification not found")

        cfg        = _get_system_config()
        to_address = (body.get("to") or notif.get("user_email") or cfg.get("email_smtp_from") or "").strip()

        if not to_address:
            return JSONResponse(status_code=400, content={
                "detail": "No recipient email provided. Enter a recipient email address."
            })

        config_error = _validate_smtp_config(
            cfg.get("email_smtp_host", ""),
            cfg.get("email_smtp_from", ""),
            to_address,
        )
        if config_error:
            return JSONResponse(status_code=400, content={"detail": config_error})

        ok, error_msg = _send_email_notif(to_address, notif["title"], notif["message"], notif.get("link"), cfg)
        if ok:
            return {"status": "ok", "message": f"Email sent successfully to {to_address}"}
        return JSONResponse(status_code=400, content={"detail": error_msg})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@notification_inbox_router.delete("/governance/notifications/inbox/{notif_id}", status_code=204)
def dismiss_notification(notif_id: str):
    try:
        _db_delete(notif_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@notification_inbox_router.post("/governance/notifications/inbox", status_code=201)
def push_notification(body: dict):
    try:
        result = create_inbox_notification(
            title=str(body.get("title", "")).strip(),
            message=str(body.get("message", "")).strip(),
            category=str(body.get("category", "System")).strip(),
            severity=str(body.get("severity", "info")).strip(),
            link=body.get("link"),
            dataset=body.get("dataset"),
            user_email=body.get("user_email"),
            notif_type=str(body.get("type", "ALERT")).upper(),
            source=body.get("source"),
            view_route=body.get("view_route"),
        )
        return result if result else {"status": "dispatched"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.post("/governance/notifications/test", status_code=200)
def send_test_notification(body: dict):
    try:
        channel = str(body.get("channel", "in_app")).strip()
        cfg     = _get_system_config()

        if channel == "in_app":
            _push_in_app(
                title="Test Notification ✅",
                message="This is a test in-app notification. Your bell icon is working!",
                notif_type="ALERT", category="System", severity="info",
                link="/settings?tab=notifications", source="Settings",
            )
            return {"status": "ok", "message": "Test in-app notification created. Check the bell icon."}

        elif channel == "email":
            recipient = (body.get("recipient_email") or cfg.get("email_smtp_from") or "").strip()
            if not recipient:
                return JSONResponse(status_code=400, content={"detail": "No recipient email configured."})
            config_error = _validate_smtp_config(cfg.get("email_smtp_host", ""), cfg.get("email_smtp_from", ""), recipient)
            if config_error:
                return JSONResponse(status_code=400, content={"detail": config_error})
            ok, error_msg = _send_email_notif(recipient, "Test Email Notification ✅", "SMTP configuration is working!", "/settings?tab=notifications", cfg)
            if ok:
                _push_in_app(title="Test Email Sent ✅", message=f"Test email delivered to {recipient}",
                             notif_type="ALERT", category="System", severity="info",
                             link="/settings?tab=notifications", source="Settings")
                return {"status": "ok", "message": f"Test email sent to {recipient}"}
            return JSONResponse(status_code=400, content={"detail": error_msg})

        elif channel == "slack":
            webhook = (body.get("slack_webhook_url") or cfg.get("slack_webhook_url") or "").strip()
            if not webhook:
                return JSONResponse(status_code=400, content={"detail": "No Slack webhook URL configured."})
            cfg = {**cfg, "slack_webhook_url": webhook}
            ok  = _send_slack_notif("Test Slack Notification ✅", "Slack integration working! ✅", "info", "/settings?tab=notifications", cfg)
            if ok:
                _push_in_app(title="Test Slack Sent ✅", message="Slack webhook is working correctly.",
                             notif_type="ALERT", category="System", severity="info",
                             link="/settings?tab=notifications", source="Settings")
                return {"status": "ok", "message": "Test Slack message sent."}
            return JSONResponse(status_code=400, content={"detail": "Slack send failed. Check the webhook URL."})

        return JSONResponse(status_code=400, content={"detail": f"Unknown channel: {channel!r}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.post("/governance/incidents", status_code=201)
def create_incident(body: dict):
    try:
        result = create_inbox_notification(
            title=str(body.get("title", "Incident Reported")).strip(),
            message=str(body.get("message", "")).strip(),
            category="Incident", severity=str(body.get("severity", "warning")).strip(),
            link=body.get("link", "/incidents"), dataset=body.get("dataset"),
            notif_type="INCIDENT", source="Incident Manager",
            view_route=body.get("link", "/incidents"),
        )
        return result if result else {"status": "dispatched"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


_registered_users: list = []


@notification_inbox_router.post("/governance/users/register", status_code=200)
def register_or_update_user(body: dict):
    try:
        email = str(body.get("email", "")).lower().strip()
        if not email:
            return JSONResponse(status_code=400, content={"detail": "email is required"})
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        for u in _registered_users:
            if u["email"] == email:
                u["last_active"] = now
                u["name"]        = body.get("name", u["name"])
                return u
        new_user = {
            "id":              f"u_{uuid.uuid4().hex[:8]}",
            "email":           email,
            "name":            body.get("name", email.split("@")[0].title()),
            "role":            body.get("role", "Viewer"),
            "status":          "Active",
            "login_method":    body.get("login_method", "unknown"),
            "last_active":     now,
            "created_at":      datetime.utcnow().strftime("%Y-%m-%d"),
            "datasets_access": 0,
        }
        _registered_users.append(new_user)
        create_inbox_notification(
            title="New User Signed In",
            message=f"{new_user['name']} ({email}) signed in via {new_user['login_method']}",
            category="User", severity="info",
            link="/settings?tab=users", notif_type="ALERT", source="Auth",
        )
        return new_user
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@notification_inbox_router.get("/governance/users/registered")
def get_registered_users():
    return _registered_users