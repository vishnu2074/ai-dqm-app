# python-backend/app/routers/knowledge_graph_team.py

import os
from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from app.database import SessionLocal
from app.services.knowledge_graph import KnowledgeGraphService

router = APIRouter()


# ── Standard db dependency — same pattern as every other router in this project ──
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/knowledge-graph")
def get_knowledge_graph(
    dataset_ids: List[int] = Body(...),   # Body(...) accepts raw JSON array [1, 2, 3]
    db: Session = Depends(get_db),        # get_db() yields — Depends works correctly
):
    service = KnowledgeGraphService()
    return service.build_graph(dataset_ids, db)


class FolderRequest(BaseModel):
    folder: str


@router.post("/knowledge-graph/folder")
def get_kg_from_folder(req: FolderRequest):
    try:
        service = KnowledgeGraphService()
        return service.build_graph_from_folder(req.folder)
    except Exception as e:
        err = str(e)
        # LLM credentials missing — return empty graph instead of 500
        if "Missing credentials" in err or "AZURE_OPENAI" in err or "api_key" in err:
            return {"nodes": [], "edges": [], "error": "LLM not configured — set AZURE_OPENAI_API_KEY in app.yaml"}
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