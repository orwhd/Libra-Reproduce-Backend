from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import DEFAULT_LIMITS
from .harness.coordinator import ResearchCoordinator
from .harness.main_gateway import MainAgentGateway
from .harness.scheduler import ResourceScheduler
from .schemas import (
    Artifact,
    ChatCommand,
    ChatMessagePage,
    ChatResponse,
    CreateRunRequest,
    ReportStatus,
    Run,
    RunMode,
    RunSnapshot,
    UploadReceipt,
)
from .storage.artifact_store import ArtifactStore, ArtifactValidationError
from .storage.event_store import GLOBAL_EVENT_HUB, EventStore


class ReproduceRuntimeContainer:
    def __init__(self) -> None:
        self.event_store = EventStore(hub=GLOBAL_EVENT_HUB)
        self.artifact_store = ArtifactStore()
        self.scheduler = ResourceScheduler(DEFAULT_LIMITS)
        self.coordinator = ResearchCoordinator(
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            scheduler=self.scheduler,
        )
        self.gateway = MainAgentGateway(self.event_store, self.coordinator)


RUNTIME = ReproduceRuntimeContainer()

router = APIRouter(prefix="/api", tags=["reproduce-v2"])


def get_reproduce_run_or_none(run_id: str) -> Run | None:
    return RUNTIME.event_store.get_run(run_id)


async def build_reproduce_sse_response(
    run_id: str,
    after: int | None = None,
    last_event_id: str | None = None,
) -> EventSourceResponse:
    run = RUNTIME.event_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    resume_after = 0
    if last_event_id and last_event_id.strip().isdigit():
        resume_after = max(resume_after, int(last_event_id.strip()))
    if after is not None:
        resume_after = max(resume_after, int(after))

    # Subscribe BEFORE reading backlog so zero events are missed between SQLite read and live stream
    queue = RUNTIME.event_store.hub.subscribe(run_id)
    backlog = RUNTIME.event_store.list_events(run_id, after=resume_after)
    current_meta = RUNTIME.event_store.get_run_meta(run_id) or {}
    already_terminal = current_meta.get("status") in ("completed", "failed")

    async def _gen():
        last_sent_seq = resume_after
        try:
            for ev in backlog:
                if ev.seq <= last_sent_seq:
                    continue
                last_sent_seq = ev.seq
                yield {
                    "id": str(ev.seq),
                    "event": "harness_event",
                    "data": ev.model_dump_json(),
                }

            if already_terminal:
                return

            while True:
                item = await queue.get()
                if item is None:
                    break
                if item.seq <= last_sent_seq:
                    continue
                last_sent_seq = item.seq
                yield {
                    "id": str(item.seq),
                    "event": "harness_event",
                    "data": item.model_dump_json(),
                }
                if item.type in ("run.completed", "run.failed"):
                    break
        finally:
            RUNTIME.event_store.hub.unsubscribe(run_id, queue)

    return EventSourceResponse(_gen())


@router.post("/runs", response_model=Run, status_code=201)
async def create_run_endpoint(
    req: CreateRunRequest,
    mode: RunMode = Query(default="live"),
    workspace_id: str | None = Query(default=None),
) -> Run:
    """Create a research run without starting paid LLM/tool execution (OpenAPI V2 `createRun`)."""
    return RUNTIME.event_store.create_run(
        title=req.title,
        paper_id=req.paper_id,
        mode=mode,
        workspace_id=workspace_id,
    )


@router.post("/workspaces/{workspace_id}/reproduce-run", response_model=Run)
async def bind_workspace_reproduce_run(
    workspace_id: str,
    title: str | None = Query(default=None),
    mode: RunMode = Query(default="live"),
) -> Run:
    """Idempotently retrieve or create a zero-cost Reproduce V2 Run bound to a Libra Research workspace."""
    return RUNTIME.event_store.get_or_create_workspace_run(
        workspace_id,
        title=title,
        mode=mode,
    )


@router.post("/runs/{run_id}/chat", response_model=ChatResponse)
async def submit_chat_endpoint(run_id: str, cmd: ChatCommand) -> JSONResponse:
    """Sole user-facing write path (`submitChat`); idempotent by `(run_id, request_id)`."""
    if not RUNTIME.event_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    status_code, resp = await RUNTIME.gateway.handle_chat(run_id, cmd)
    return JSONResponse(status_code=status_code, content=resp.model_dump())


@router.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
async def get_snapshot_endpoint(
    run_id: str,
    at_seq: int | None = Query(default=None, ge=0),
) -> RunSnapshot:
    snap = RUNTIME.event_store.get_snapshot(run_id, at_seq=at_seq)
    if not snap:
        raise HTTPException(status_code=404, detail="Run not found")
    return snap


@router.get("/runs/{run_id}/messages", response_model=ChatMessagePage)
async def list_run_messages_endpoint(
    run_id: str,
    ref_kind: str | None = Query(default=None),
    ref_id: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
) -> ChatMessagePage:
    if not RUNTIME.event_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNTIME.event_store.list_messages(
        run_id,
        ref_kind=ref_kind,
        ref_id=ref_id,
        cursor=cursor,
        limit=limit,
    )


@router.get("/runs/{run_id}/artifacts", response_model=list[Artifact])
async def list_run_artifacts_endpoint(run_id: str) -> list[Artifact]:
    if not RUNTIME.event_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNTIME.artifact_store.list_run_artifacts(run_id)


@router.get("/runs/{run_id}/artifacts-detail")
async def list_run_artifacts_detail_endpoint(run_id: str) -> list[dict[str, Any]]:
    if not RUNTIME.event_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNTIME.artifact_store.list_run_artifacts_rich(run_id)


@router.get("/artifacts/{artifact_id}", response_model=Artifact)
async def get_artifact_endpoint(artifact_id: str) -> Artifact:
    art = RUNTIME.artifact_store.get_artifact(artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return art


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact_endpoint(
    artifact_id: str,
    inline: bool = Query(default=False),
) -> Response:
    resolved = RUNTIME.artifact_store.resolve_artifact_bytes(artifact_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Artifact file not found")
    fpath, filename, mime_type = resolved
    disposition = "inline" if inline and mime_type in ("application/pdf", "image/png", "image/jpeg") else "attachment"
    return Response(
        content=fpath.read_bytes(),
        media_type=mime_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox",
        },
    )


@router.get("/artifacts/{artifact_id}/preview")
async def preview_artifact_text_endpoint(artifact_id: str) -> dict[str, Any]:
    """Safe read-only text/table preview for InChatDocumentViewer (never returns raw host paths)."""
    resolved = RUNTIME.artifact_store.resolve_artifact_bytes(artifact_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Artifact file not found")
    fpath, filename, mime_type = resolved
    raw = fpath.read_bytes()
    ext = fpath.suffix.lower()

    if ext in (".csv", ".tsv", ".py", ".json", ".md", ".txt"):
        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "mime_type": mime_type,
            "preview_kind": "text",
            "text": raw.decode("utf-8", errors="replace")[:80_000],
        }
    if ext == ".docx":
        import io
        import docx

        d = docx.Document(io.BytesIO(raw))
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "mime_type": mime_type,
            "preview_kind": "docx_outline",
            "text": "\n\n".join(paras)[:80_000],
        }
    if ext == ".pdf":
        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "mime_type": mime_type,
            "preview_kind": "pdf",
            "inline_url": f"/api/artifacts/{artifact_id}/download?inline=true",
        }
    if ext in (".png", ".jpg", ".jpeg"):
        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "mime_type": mime_type,
            "preview_kind": "image",
            "inline_url": f"/api/artifacts/{artifact_id}/download?inline=true",
        }
    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "mime_type": mime_type,
        "preview_kind": "download_only",
        "download_url": f"/api/artifacts/{artifact_id}/download",
    }


@router.get("/runs/{run_id}/report", response_model=ReportStatus)
async def get_report_endpoint(run_id: str) -> ReportStatus:
    rep = RUNTIME.event_store.get_report_status(run_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Run not found")
    return rep


@router.post("/runs/{run_id}/attachments", response_model=UploadReceipt, status_code=201)
async def upload_run_attachment_endpoint(
    run_id: str,
    file: UploadFile = File(...),
) -> UploadReceipt:
    """Safe attachment upload; upload alone NEVER starts agent work."""
    if not RUNTIME.event_store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    raw = await file.read()
    try:
        return RUNTIME.artifact_store.save_upload_attachment(
            run_id=run_id,
            filename=file.filename or "upload.bin",
            content=raw,
            declared_mime=file.content_type,
        )
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
