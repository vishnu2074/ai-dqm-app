"""
app/routers/knowledge_graph.py

KEY FIX: /knowledge-graph endpoint now calls kg_engine.detect_relationships()
directly instead of going through KnowledgeGraphService.build_graph() which
called build_column_graph() (a visual renderer that reads but NEVER writes edges).
detect_relationships() calls _save_edges() which persists to knowledge_graph_edges
with correct INTEGER FKs — this is why the table was always 0 edges.

The router still calls service.build_column_graph() AFTER detect_relationships()
to return the visual graph the frontend expects (nodes+edges for display).
"""
import os
from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient
from app.database import SessionLocal

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Folder-mode KG Persistence ────────────────────────────────────────────────

def _persist_folder_kg_edges(edges: list, folder: str) -> int:
    """
    Persist KG edges from folder-mode builds.
    Resolves string names to INTEGER dataset IDs from the datasets table.
    Non-fatal.
    """
    if not edges:
        return 0
    saved = 0
    try:
        from app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            tbl = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_graph_edges'"
            )).fetchone()
            if not tbl:
                return 0

            ds_rows = conn.execute(text(
                "SELECT id, physical_name, display_name FROM datasets"
            )).fetchall()
            name_to_id: dict = {}
            id_to_name: dict = {}
            for row in ds_rows:
                did   = row[0]
                pname = (row[1] or "").strip()
                dname = (row[2] or "").strip()
                id_to_name[did] = dname or pname
                for raw in (pname, dname):
                    if not raw:
                        continue
                    name_to_id[raw.lower()] = did
                    base = raw.replace("\\", "/").split("/")[-1].lower()
                    if base:
                        name_to_id.setdefault(base, did)
                    stem = base.rsplit(".", 1)[0]
                    if stem:
                        name_to_id.setdefault(stem, did)

            def find_id(name_str: str):
                if not name_str:
                    return None, None
                low = str(name_str).lower().strip()
                if low in name_to_id:
                    did = name_to_id[low]
                    return did, id_to_name.get(did, name_str)
                base = low.split("/")[-1]
                if base in name_to_id:
                    did = name_to_id[base]
                    return did, id_to_name.get(did, name_str)
                stem = base.rsplit(".", 1)[0]
                if stem and stem in name_to_id:
                    did = name_to_id[stem]
                    return did, id_to_name.get(did, name_str)
                for k, v in name_to_id.items():
                    if low in k or k in low:
                        return v, id_to_name.get(v, name_str)
                return None, None

            try:
                conn.execute(text(
                    "UPDATE knowledge_graph_edges SET invalidated = 1 "
                    "WHERE invalidated = 0 OR invalidated IS NULL"
                ))
            except Exception:
                pass

            skipped = 0
            for edge in edges:
                if isinstance(edge.get("source_dataset_id"), int):
                    src_id   = edge["source_dataset_id"]
                    src_name = edge.get("source_dataset_name", "")
                    src_col  = edge.get("source_column", "unknown")
                else:
                    src_name_raw = edge.get("source") or edge.get("from") or edge.get("source_dataset", "")
                    src_id, src_name = find_id(src_name_raw)
                    src_col = edge.get("source_column") or "unknown"

                if isinstance(edge.get("target_dataset_id"), int):
                    tgt_id   = edge["target_dataset_id"]
                    tgt_name = edge.get("target_dataset_name", "")
                    tgt_col  = edge.get("target_column", "unknown")
                else:
                    tgt_name_raw = edge.get("target") or edge.get("to") or edge.get("target_dataset", "")
                    tgt_id, tgt_name = find_id(tgt_name_raw)
                    tgt_col = edge.get("target_column") or "unknown"

                if src_id is None or tgt_id is None:
                    skipped += 1
                    continue
                if src_id == tgt_id:
                    continue

                rel_type    = edge.get("relationship_type") or edge.get("type") or "related"
                confidence  = float(edge.get("confidence") or edge.get("weight") or 0.5)
                explanation = edge.get("explanation") or edge.get("llm_explanation") or ""

                try:
                    conn.execute(text("""
                        INSERT INTO knowledge_graph_edges
                            (source_dataset_id, source_column, source_dataset_name,
                             target_dataset_id, target_column, target_dataset_name,
                             relationship_type, confidence, method, llm_explanation, invalidated)
                        VALUES
                            (:src_id, :src_col, :src_name,
                             :tgt_id, :tgt_col, :tgt_name,
                             :rel, :conf, :method, :expl, 0)
                    """), {
                        "src_id": src_id, "src_col": src_col, "src_name": src_name,
                        "tgt_id": tgt_id, "tgt_col": tgt_col, "tgt_name": tgt_name,
                        "rel": rel_type, "conf": confidence,
                        "method": "folder", "expl": explanation,
                    })
                    saved += 1
                except Exception as ie:
                    print(f"[kg_persist_folder] Edge insert failed: {ie}")

            conn.commit()
            print(f"[kg_persist_folder] Saved {saved}/{len(edges)} edges ({skipped} skipped)")

    except Exception as e:
        print(f"[kg_persist_folder] Persistence failed (non-fatal): {e}")
    return saved


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/knowledge-graph")
def get_knowledge_graph(
    dataset_ids: List[int] = Body(...),
    db: Session = Depends(get_db),
    force_rerun: bool = False,
):
    """
    Build knowledge graph for specified datasets.

    FIXED: Now calls detect_relationships() directly which persists edges to
    knowledge_graph_edges via _save_edges(). Previously called build_graph()
    on the service which called build_column_graph() — a visual-only function
    that READS edges but never WRITES them, hence always 0 in the DB.

    Returns the visual graph (nodes+edges) for frontend rendering.
    """
    from app.graph.kg_engine import detect_relationships, build_column_graph

    # Step 1: Run relationship detection — this persists edges to the DB
    detect_result = detect_relationships(db, dataset_ids, force_rerun=force_rerun)
    print(f"[kg_router] detect_relationships: {detect_result.get('status')} "
          f"— {detect_result.get('edges_saved', 0)} edges saved")

    # Step 2: Build visual graph from the now-populated DB edges
    visual_result = build_column_graph(db, dataset_ids)

    # Merge detect_result stats into visual_result for frontend info
    visual_result["kg_stats"] = {
        "status":              detect_result.get("status"),
        "edges_saved":         detect_result.get("edges_saved", 0),
        "relationships_found": detect_result.get("relationships_found", 0),
        "pairs_compared":      detect_result.get("pairs_compared", 0),
        "llm_calls":           detect_result.get("llm_calls", 0),
        "elapsed_ms":          detect_result.get("elapsed_ms", 0),
        "message":             detect_result.get("message", ""),
    }

    return visual_result


class FolderRequest(BaseModel):
    folder: str


@router.post("/knowledge-graph/folder")
def get_kg_from_folder(req: FolderRequest):
    """
    Build knowledge graph from Azure Blob folder.
    Folder builds operate without a DB session, so we persist edges here
    using the fixed name→id resolver.
    """
    try:
        from app.services.knowledge_graph import KnowledgeGraphService
        service = KnowledgeGraphService()
        result = service.build_graph_from_folder(req.folder)

        try:
            raw_edges = (result.get("edges") or result.get("relationships", [])
                         if isinstance(result, dict) else [])
            if raw_edges:
                _persist_folder_kg_edges(raw_edges, folder=req.folder)
        except Exception as e:
            print(f"[kg_router] Post-folder-build persistence error (non-fatal): {e}")

        return result

    except Exception as e:
        err = str(e)
        if "Missing credentials" in err or "AZURE_OPENAI" in err or "api_key" in err:
            return {
                "nodes": [], "edges": [],
                "error": "LLM not configured — set AZURE_OPENAI_API_KEY in app.yaml"
            }
        raise


@router.post("/knowledge-graph/detect")
def detect_kg_relationships(
    dataset_ids: Optional[List[int]] = Body(default=None),
    force_rerun: bool = False,
    db: Session = Depends(get_db),
):
    """
    Explicit relationship detection endpoint.
    Returns detection stats rather than the visual graph.
    Useful for triggering a fresh analysis without rebuilding the visual.
    """
    from app.graph.kg_engine import detect_relationships
    result = detect_relationships(db, dataset_ids, force_rerun=force_rerun)
    return result


@router.get("/folders")
def list_folders():
    """Lists top-level folders under dqm/raw/ in Azure Blob."""
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