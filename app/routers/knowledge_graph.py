"""
python-backend/app/routers/knowledge_graph.py
Knowledge Graph endpoints for automatic relationship discovery

FIXED:
  - service.build_graph(db, dataset_ids) — args were swapped
  - KG edges now persisted to knowledge_graph_edges table after every build
    so health metrics (kg_build_status, kg_relationship_precision) show real data
"""
import os
from datetime import datetime
from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient
from app.database import SessionLocal
from app.services.knowledge_graph import KnowledgeGraphService

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── KG Edge Persistence ───────────────────────────────────────────────────────

def _persist_kg_edges(edges: list, folder: str = None) -> int:
    """
    Persist KG edges to knowledge_graph_edges table.
    Schema-agnostic: introspects actual columns at runtime and maps
    whatever fields the edge dicts contain to whatever columns exist.
    Non-fatal — a persistence failure never breaks the API response.
    """
    if not edges:
        return 0
    saved = 0
    try:
        from app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            # ── Discover actual table schema ──────────────────────────────────
            cols_result = conn.execute(
                text("PRAGMA table_info(knowledge_graph_edges)")
            ).fetchall()
            existing_cols = {row[1] for row in cols_result}
            if not existing_cols:
                print("[kg_persist] knowledge_graph_edges table not found — skipping")
                return 0

            # ── Clear previous auto-generated edges ───────────────────────────
            if "folder" in existing_cols and folder:
                conn.execute(
                    text("DELETE FROM knowledge_graph_edges WHERE folder = :f"),
                    {"f": folder}
                )
            elif folder:
                # Table has no folder column — clear all non-user-defined edges
                if "is_user_defined" in existing_cols:
                    conn.execute(text(
                        "DELETE FROM knowledge_graph_edges WHERE COALESCE(is_user_defined,0) = 0"
                    ))
                else:
                    conn.execute(text("DELETE FROM knowledge_graph_edges"))
            else:
                # Dataset-IDs mode: clear old dataset-mode edges
                if "is_user_defined" in existing_cols:
                    conn.execute(text(
                        "DELETE FROM knowledge_graph_edges WHERE COALESCE(is_user_defined,0) = 0"
                    ))
                else:
                    conn.execute(text("DELETE FROM knowledge_graph_edges"))

            now = datetime.utcnow().isoformat()

            for edge in edges:
                # ── Map edge fields → possible column names in priority order ──
                edge_confidence = (
                    edge.get("confidence") or edge.get("weight") or
                    edge.get("score") or 0.8
                )
                edge_source = edge.get("source") or edge.get("from", "")
                edge_target = edge.get("target") or edge.get("to", "")
                edge_type   = (
                    edge.get("relationship_type") or edge.get("type") or
                    edge.get("edge_type") or "related"
                )

                field_map = {
                    # Column name variants for source
                    "source_node":    edge_source,
                    "source":         edge_source,
                    "source_dataset": edge_source,
                    "from_node":      edge_source,
                    # Column name variants for target
                    "target_node":    edge_target,
                    "target":         edge_target,
                    "target_dataset": edge_target,
                    "to_node":        edge_target,
                    # Relationship / type
                    "relationship_type": edge_type,
                    "edge_type":         edge_type,
                    "type":              edge_type,
                    "relation":          edge_type,
                    # Confidence / weight
                    "confidence": edge_confidence,
                    "weight":     edge_confidence,
                    "score":      edge_confidence,
                    # Metadata
                    "created_at": now,
                    "folder":     folder,
                    "dataset_id": edge.get("dataset_id"),
                }

                # Only include columns that actually exist in the table
                row = {
                    col: val for col, val in field_map.items()
                    if col in existing_cols and val is not None
                }
                if not row:
                    continue

                cols_str = ", ".join(row.keys())
                vals_str = ", ".join(f":{k}" for k in row.keys())
                try:
                    conn.execute(
                        text(f"INSERT INTO knowledge_graph_edges ({cols_str}) VALUES ({vals_str})"),
                        row
                    )
                    saved += 1
                except Exception as insert_err:
                    # Single edge failure is non-fatal
                    pass

            conn.commit()
            print(f"[kg_persist] Saved {saved}/{len(edges)} edges to knowledge_graph_edges")

    except Exception as e:
        print(f"[kg_persist] Persistence failed (non-fatal): {e}")
    return saved


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/knowledge-graph")
def get_knowledge_graph(
    dataset_ids: List[int] = Body(...),
    db: Session = Depends(get_db),
):
    """Build knowledge graph for specified datasets and persist edges."""
    service = KnowledgeGraphService()
    result = service.build_graph(db, dataset_ids)  # FIXED: db first, dataset_ids second

    # Persist to DB for health metrics
    try:
        if isinstance(result, dict) and result.get("edges"):
            _persist_kg_edges(result["edges"])
    except Exception as e:
        print(f"[kg_persist] Post-build persistence error (non-fatal): {e}")

    return result


class FolderRequest(BaseModel):
    folder: str


@router.post("/knowledge-graph/folder")
def get_kg_from_folder(req: FolderRequest):
    """Build knowledge graph from Azure Blob folder and persist edges."""
    try:
        service = KnowledgeGraphService()
        result = service.build_graph_from_folder(req.folder)

        # Persist to DB for health metrics
        try:
            if isinstance(result, dict) and result.get("edges"):
                _persist_kg_edges(result["edges"], folder=req.folder)
        except Exception as e:
            print(f"[kg_persist] Post-folder-build persistence error (non-fatal): {e}")

        return result

    except Exception as e:
        err = str(e)
        # LLM credentials missing — return empty graph instead of 500
        if "Missing credentials" in err or "AZURE_OPENAI" in err or "api_key" in err:
            return {
                "nodes": [], "edges": [],
                "error": "LLM not configured — set AZURE_OPENAI_API_KEY in app.yaml"
            }
        raise


@router.get("/folders")
def list_folders():
    """
    Lists top-level folders under dqm/raw/ in Azure Blob.
    Always returns a list so folders.map() never crashes in the frontend.
    """
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name    = os.getenv("AZURE_STORAGE_CONTAINER", "intern26")

        if not connection_string:
            return []

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client    = blob_service_client.get_container_client(container_name)

        folders = set()
        blobs   = container_client.list_blobs(name_starts_with="dqm/raw/")
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) > 2 and parts[2]:
                folders.add(parts[2])

        return sorted(list(folders))

    except Exception as e:
        print(f"[FOLDER FETCH ERROR] {e}")
        return []