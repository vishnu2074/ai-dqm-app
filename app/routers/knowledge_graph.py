"""
app/routers/knowledge_graph.py

DEFINITIVE FIX:
  /knowledge-graph now calls detect_relationships() synchronously (which persists
  edges to knowledge_graph_edges via _save_edges()), then calls build_column_graph()
  to return the visual graph WITH relationship lines drawn between columns.

  The previous service.build_graph() only called build_column_graph() which is a
  READ-only visual function — it never writes edges. That's why knowledge_graph_edges
  was always 0 and the graph showed column bubbles but no relationship lines.

  detect_relationships() has a rule-based fallback so it works even when LLM fails:
  it finds FK patterns and shared column names without any LLM call.
  After this endpoint runs, the KG visual will show relationship lines AND
  health metrics (kg_build_status, kg_entity_coverage) will show real values.
"""
import os
import threading
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

            ds_rows = conn.execute(text("SELECT id, physical_name, display_name FROM datasets")).fetchall()
            name_to_id: dict = {}
            id_to_name: dict = {}
            for row in ds_rows:
                did, pname, dname = row[0], (row[1] or "").strip(), (row[2] or "").strip()
                id_to_name[did] = dname or pname
                for raw in (pname, dname):
                    if not raw: continue
                    name_to_id[raw.lower()] = did
                    base = raw.replace("\\", "/").split("/")[-1].lower()
                    if base: name_to_id.setdefault(base, did)
                    stem = base.rsplit(".", 1)[0]
                    if stem: name_to_id.setdefault(stem, did)

            def find_id(name_str):
                if not name_str: return None, None
                low = str(name_str).lower().strip()
                for key in [low, low.split("/")[-1], low.split("/")[-1].rsplit(".", 1)[0]]:
                    if key and key in name_to_id:
                        did = name_to_id[key]
                        return did, id_to_name.get(did, name_str)
                for k, v in name_to_id.items():
                    if low in k or k in low:
                        return v, id_to_name.get(v, name_str)
                return None, None

            try:
                conn.execute(text("UPDATE knowledge_graph_edges SET invalidated = 1 WHERE invalidated = 0 OR invalidated IS NULL"))
            except Exception:
                pass

            for edge in edges:
                src_id, src_name = (edge.get("source_dataset_id"), edge.get("source_dataset_name", "")) \
                    if isinstance(edge.get("source_dataset_id"), int) \
                    else find_id(edge.get("source") or edge.get("from") or edge.get("source_dataset", ""))
                tgt_id, tgt_name = (edge.get("target_dataset_id"), edge.get("target_dataset_name", "")) \
                    if isinstance(edge.get("target_dataset_id"), int) \
                    else find_id(edge.get("target") or edge.get("to") or edge.get("target_dataset", ""))
                if src_id is None or tgt_id is None or src_id == tgt_id:
                    continue
                try:
                    conn.execute(text("""
                        INSERT INTO knowledge_graph_edges
                            (source_dataset_id, source_column, source_dataset_name,
                             target_dataset_id, target_column, target_dataset_name,
                             relationship_type, confidence, method, llm_explanation, invalidated)
                        VALUES (:si, :sc, :sn, :ti, :tc, :tn, :rel, :conf, :method, :expl, 0)
                    """), {
                        "si": src_id, "sc": edge.get("source_column", "unknown"), "sn": src_name,
                        "ti": tgt_id, "tc": edge.get("target_column", "unknown"), "tn": tgt_name,
                        "rel": edge.get("relationship_type") or "related",
                        "conf": float(edge.get("confidence") or 0.5),
                        "method": "folder", "expl": edge.get("explanation") or "",
                    })
                    saved += 1
                except Exception as ie:
                    print(f"[kg_folder] Edge insert failed: {ie}")
            conn.commit()
            print(f"[kg_folder] Saved {saved}/{len(edges)} edges")
    except Exception as e:
        print(f"[kg_folder] Persistence failed (non-fatal): {e}")
    return saved


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/knowledge-graph")
def get_knowledge_graph(
    dataset_ids: List[int] = Body(...),
    db: Session = Depends(get_db),
    force_rerun: bool = False,
):
    """
    Build KG: detect column relationships (with rule-based fallback when LLM
    unavailable) → persist edges → return visual graph with relationship lines.
    """
    from app.graph.kg_engine import detect_relationships, build_column_graph

    # Step 1: Detect & persist relationships (rule-based fallback if LLM fails)
    detect_result = detect_relationships(db, dataset_ids, force_rerun=force_rerun)
    print(f"[kg_router] detect: status={detect_result.get('status')} "
          f"edges_saved={detect_result.get('edges_saved', 0)} "
          f"relationships={detect_result.get('relationships_found', 0)}")

    # Step 2: Build visual graph — now reads the edges we just saved
    visual = build_column_graph(db, dataset_ids)

    # Attach detection stats for frontend info
    visual["kg_stats"] = {
        "status":              detect_result.get("status"),
        "edges_saved":         detect_result.get("edges_saved", 0),
        "relationships_found": detect_result.get("relationships_found", 0),
        "pairs_compared":      detect_result.get("pairs_compared", 0),
        "method":              "llm" if detect_result.get("llm_calls", 0) > 0 else "rule",
        "message":             detect_result.get("message", ""),
    }
    return visual


@router.post("/knowledge-graph/force-rebuild")
def force_rebuild_kg(
    dataset_ids: Optional[List[int]] = Body(default=None),
    db: Session = Depends(get_db),
):
    """Force full rebuild — invalidates cached edges and re-runs detection."""
    from app.graph.kg_engine import detect_relationships, build_column_graph
    detect_result = detect_relationships(db, dataset_ids, force_rerun=True)
    visual = build_column_graph(db, dataset_ids)
    visual["kg_stats"] = detect_result
    return visual


class FolderRequest(BaseModel):
    folder: str


@router.post("/knowledge-graph/folder")
def get_kg_from_folder(req: FolderRequest):
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
            print(f"[kg_router] folder persistence error (non-fatal): {e}")
        return result
    except Exception as e:
        err = str(e)
        if "Missing credentials" in err or "AZURE_OPENAI" in err or "api_key" in err:
            return {"nodes": [], "edges": [], "error": "LLM not configured"}
        raise


@router.get("/folders")
def list_folders():
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
        print(f"[FOLDER FETCH ERROR] {e}")
        return []