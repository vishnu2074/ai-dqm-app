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
    FIXED: Persist KG edges using the CORRECT KnowledgeGraphEdge model schema:
        source_dataset_id  INTEGER FK NOT NULL
        source_column      TEXT NOT NULL  (defaults to 'unknown')
        source_dataset_name TEXT
        target_dataset_id  INTEGER FK NOT NULL
        target_column      TEXT NOT NULL
        target_dataset_name TEXT
        relationship_type  TEXT NOT NULL
        confidence         FLOAT
        method             TEXT
        llm_explanation    TEXT
        invalidated        BOOLEAN (0=active, 1=superseded)

    Resolves string source/target names from the LLM output to integer
    dataset IDs using a multi-key lookup against the datasets table.
    Edges that cannot be resolved are skipped and logged.
    Non-fatal -- persistence failure never crashes the endpoint.
    """
    if not edges:
        return 0
    saved   = 0
    skipped = 0
    try:
        from app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            tbl = conn.execute(text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='knowledge_graph_edges'"
            )).fetchone()
            if not tbl:
                print("[kg_persist] knowledge_graph_edges table missing -- skipping")
                return 0

            # Build name->id lookup from all registered datasets
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
                    if raw:
                        name_to_id[raw.lower()] = did
                        base = raw.replace("\\", "/").split("/")[-1].lower()
                        if base and base not in name_to_id:
                            name_to_id[base] = did
                        stem = base.rsplit(".", 1)[0]
                        if stem and stem not in name_to_id:
                            name_to_id[stem] = did

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

            # Mark all existing active edges as invalidated before inserting new ones
            try:
                conn.execute(text(
                    "UPDATE knowledge_graph_edges "
                    "SET invalidated = 1 WHERE invalidated = 0 OR invalidated IS NULL"
                ))
            except Exception:
                pass

            for edge in edges:
                src_name    = (edge.get("source") or edge.get("from")
                               or edge.get("source_dataset") or "")
                tgt_name    = (edge.get("target") or edge.get("to")
                               or edge.get("target_dataset") or "")
                rel_type    = (edge.get("relationship_type") or edge.get("type") or "related")
                confidence  = float(edge.get("confidence") or edge.get("weight") or 0.0)
                explanation = edge.get("explanation") or edge.get("llm_explanation") or ""
                method      = edge.get("method") or ("folder" if folder else "auto")
                src_col     = edge.get("source_column") or edge.get("src_col") or "unknown"
                tgt_col     = edge.get("target_column") or edge.get("tgt_col") or "unknown"

                src_id, src_dname = find_id(src_name)
                tgt_id, tgt_dname = find_id(tgt_name)

                if src_id is None or tgt_id is None:
                    skipped += 1
                    if skipped <= 5:
                        print(f"[kg_persist] Cannot resolve '{src_name}'->'{tgt_name}' "
                              f"(src_id={src_id}, tgt_id={tgt_id}) -- skipping")
                    continue
                if src_id == tgt_id:
                    continue

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
                        "src_id":   src_id,   "src_col": src_col,  "src_name": src_dname,
                        "tgt_id":   tgt_id,   "tgt_col": tgt_col,  "tgt_name": tgt_dname,
                        "rel":      rel_type, "conf":    confidence,
                        "method":   method,   "expl":    explanation,
                    })
                    saved += 1
                except Exception as ie:
                    print(f"[kg_persist] Edge insert failed (non-fatal): {ie}")

            conn.commit()
            suffix = f" ({skipped} skipped -- unresolved names)" if skipped else ""
            print(f"[kg_persist] Saved {saved}/{len(edges)} edges to knowledge_graph_edges{suffix}")

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
