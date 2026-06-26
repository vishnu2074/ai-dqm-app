"""
python-backend/app/routers/knowledge_graph.py
Knowledge Graph endpoints for automatic relationship discovery

FIXED (v2):
  ROOT CAUSE OF "kg_build_status: Not Built": this router had its OWN
  _persist_kg_edges() function that:
    1. Wrote to a generic schema (source_node/from_node/source/etc.) that
       doesn't match the REAL knowledge_graph_edges table — the real table
       (defined in models/__init__.py as KnowledgeGraphEdge) has NOT NULL
       INTEGER columns source_dataset_id / target_dataset_id, and this
       function never wrote those at all — it wrote string dataset NAMES
       into whatever text-like column it could find via introspection.
    2. Ran a DELETE that wiped rows on every single build, including rows
       that app/graph/kg_engine.py's _save_edges() had ALREADY correctly
       written via the ORM (with proper INTEGER FKs) earlier in the same
       request, since service.build_graph() internally calls detect_relationships()
       which calls _save_edges().

  In short: kg_engine.py already does this correctly through the ORM.
  This router was redundantly (and incorrectly) re-persisting the same
  data with a second, schema-mismatched code path, and clobbering the
  correct rows in the process. The fix is to DELETE the duplicate
  persistence logic entirely and trust kg_engine.py's service layer,
  which is the only thing that actually understands the FK schema.

  - service.build_graph(db, dataset_ids) — args were swapped (kept fixed)
  - KG edges are now persisted ONLY by kg_engine.detect_relationships() /
    _save_edges(), called internally by KnowledgeGraphService.build_graph().
    This router no longer touches the table directly at all.
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/knowledge-graph")
def get_knowledge_graph(
    dataset_ids: List[int] = Body(...),
    db: Session = Depends(get_db),
):
    """
    Build knowledge graph for specified datasets.

    Persistence happens entirely inside KnowledgeGraphService / kg_engine —
    this endpoint no longer writes to knowledge_graph_edges directly.
    See app/graph/kg_engine.py: detect_relationships() -> _save_edges(),
    which correctly populates source_dataset_id / target_dataset_id as
    INTEGER FKs via the ORM (KnowledgeGraphEdge model), with proper
    dedup/invalidation logic already built in.
    """
    service = KnowledgeGraphService()
    result = service.build_graph(db, dataset_ids)  # FIXED: db first, dataset_ids second
    return result


class FolderRequest(BaseModel):
    folder: str


@router.post("/knowledge-graph/folder")
def get_kg_from_folder(req: FolderRequest):
    """
    Build knowledge graph from Azure Blob folder.

    Same as above — no direct table writes here. If KnowledgeGraphService's
    folder-based build path also needs row persistence, that should be
    implemented inside the service layer using the same ORM-based approach
    as kg_engine._save_edges(), not duplicated here.
    """
    try:
        service = KnowledgeGraphService()
        result = service.build_graph_from_folder(req.folder)
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
