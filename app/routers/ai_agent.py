# python-backend/app/routers/ai_agent.py

import json
import os
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from app.database import SessionLocal
from app.services.ai_agent import AICopilotService

router = APIRouter(prefix="/ai-agent", tags=["ai-agent"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Request models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    dataset_id: Optional[int]  = None
    folder:     Optional[str]  = None
    history:    list           = []

class ConfirmRequest(BaseModel):
    action_id:  str
    dataset_id: Optional[int]  = None
    folder:     Optional[str]  = None
    confirmed:  bool

class BulkConfirmRequest(BaseModel):
    action_ids: List[str]
    dataset_id: Optional[int]  = None
    folder:     Optional[str]  = None

class FeedbackRequest(BaseModel):
    message_id: str
    rating:     int
    comment:    Optional[str]  = None
    dataset_id: Optional[int]  = None
    folder:     Optional[str]  = None

class PinRequest(BaseModel):
    dataset_id: int
    message_id: str
    content:    str

class FolderSummaryRequest(BaseModel):
    folder: str


# ─── Folder endpoints ─────────────────────────────────────────────────────────

@router.get("/folders")
def list_folders():
    """List all folders from Azure Blob Storage (dqm/raw/<folder>/)."""
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name    = os.getenv("AZURE_STORAGE_CONTAINER", "intern26")
        if not connection_string:
            return []
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client    = blob_service_client.get_container_client(container_name)
        folders = set()
        for blob in container_client.list_blobs(name_starts_with="dqm/raw/"):
            parts = blob.name.split("/")
            if len(parts) > 2 and parts[2]:
                folders.add(parts[2])
        return sorted(list(folders))
    except Exception as e:
        print(f"[AI Agent FOLDER FETCH ERROR] {e}")
        return []


@router.get("/folder-datasets/{folder_name}")
def get_folder_datasets(folder_name: str, db: Session = Depends(get_db)):
    """
    Returns all datasets registered under this folder with health/issues/rows.
    Matches datasets whose physical_name starts with dqm/raw/<folder_name>/.
    """
    service  = AICopilotService(db)
    datasets = service.get_folder_datasets(folder_name)
    return {"folder": folder_name, "datasets": datasets}


@router.post("/folder-summary")
def get_folder_summary(req: FolderSummaryRequest, db: Session = Depends(get_db)):
    """Aggregate DQ summary for all datasets in a folder."""
    return AICopilotService(db).get_folder_summary(req.folder)


@router.get("/suggestions-folder/{folder_name}")
def get_folder_suggestions(folder_name: str, db: Session = Depends(get_db)):
    """Proactive suggestions and sidebar actions for a folder."""
    return AICopilotService(db).get_folder_suggestions(folder_name)


@router.get("/history-folder/{folder_name}")
def get_folder_history(folder_name: str, db: Session = Depends(get_db)):
    """Load persisted chat history for a folder."""
    return {"history": AICopilotService(db).get_folder_history(folder_name)}


@router.delete("/memory-folder/{folder_name}")
def clear_folder_memory(folder_name: str, db: Session = Depends(get_db)):
    """Clear folder chat history."""
    return AICopilotService(db).clear_folder_memory(folder_name)


# ─── Chat (shared for dataset and folder mode) ────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.
    - Folder mode:  send folder=<name>, dataset_id=null
    - Dataset mode: send dataset_id=<id>, folder=null
    """
    service = AICopilotService(db)

    async def stream():
        try:
            async for event in service.run(
                req.message,
                req.dataset_id,
                req.history,
                folder=req.folder,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─── Action confirmation ──────────────────────────────────────────────────────

@router.post("/confirm")
def confirm_action(req: ConfirmRequest, db: Session = Depends(get_db)):
    """
    Confirm a single pending action.
    dataset_id may be null in folder mode — the action's own params carry the dataset_id.
    """
    if not req.confirmed:
        return {"status": "rejected", "message": "Action cancelled."}
    return AICopilotService(db).execute_action(req.action_id, req.dataset_id)


@router.post("/confirm-bulk")
def confirm_bulk(req: BulkConfirmRequest, db: Session = Depends(get_db)):
    """Confirm multiple pending actions at once."""
    return AICopilotService(db).execute_bulk(req.action_ids, req.dataset_id)


# ─── Suggestions (per dataset) ────────────────────────────────────────────────

@router.get("/suggestions/{dataset_id}")
def get_suggestions(dataset_id: int, db: Session = Depends(get_db)):
    return AICopilotService(db).get_proactive_suggestions(dataset_id)


# ─── History (per dataset) ────────────────────────────────────────────────────

@router.get("/history/{dataset_id}")
def get_history(dataset_id: int, db: Session = Depends(get_db)):
    return {"history": AICopilotService(db).get_history(dataset_id)}


# ─── Feedback ─────────────────────────────────────────────────────────────────

@router.post("/feedback")
def record_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    return AICopilotService(db).record_feedback(
        req.message_id, req.rating, req.comment, req.dataset_id, req.folder
    )


# ─── Pin management ───────────────────────────────────────────────────────────

@router.post("/pin")
def pin_message(req: PinRequest, db: Session = Depends(get_db)):
    return AICopilotService(db).pin_message(req.dataset_id, req.message_id, req.content)


@router.get("/pins/{dataset_id}")
def get_pins(dataset_id: int, db: Session = Depends(get_db)):
    return AICopilotService(db).get_pinned(dataset_id)


@router.delete("/pins/{dataset_id}/{message_id}")
def unpin_message(dataset_id: int, message_id: str, db: Session = Depends(get_db)):
    return AICopilotService(db).unpin_message(dataset_id, message_id)


# ─── Memory management ────────────────────────────────────────────────────────

@router.delete("/memory/{dataset_id}")
def clear_memory(dataset_id: int, db: Session = Depends(get_db)):
    return AICopilotService(db).clear_memory(dataset_id)