from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import DEFAULT_LIMITS
from ..harness.persona_catalog import build_initial_personas
from ..schemas import (
    CanvasProjection,
    ChatActivityMessage,
    ChatMessagePage,
    ChatResponse,
    Collaboration,
    EventSourceLiteral,
    EventSubject,
    HarnessEvent,
    Persona,
    ReportStatus,
    Run,
    RunMode,
    RunSnapshot,
    RunStatusLiteral,
    SelectedContext,
    SourceRef,
)
from ..security.tool_permissions import redact_secrets
from .db import get_reproduce_conn, jdumps, jloads, utc_now_iso


class EventSubscriberHub:
    """Per-run async SSE fan-out hub; only receives events AFTER SQLite commit."""

    def __init__(self) -> None:
        self._subs: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[HarnessEvent | None]]]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue[HarnessEvent | None]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._subs.setdefault(run_id, []).append((loop, q))
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[HarnessEvent | None]) -> None:
        items = self._subs.get(run_id, [])
        self._subs[run_id] = [(lp, item_q) for lp, item_q in items if item_q is not q]

    def publish(self, run_id: str, event: HarnessEvent | None) -> None:
        for lp, q in list(self._subs.get(run_id, [])):
            if not lp.is_closed():
                lp.call_soon_threadsafe(q.put_nowait, event)


GLOBAL_EVENT_HUB = EventSubscriberHub()


class EventStore:
    """
    Append-only transactional EventStore + deterministic Snapshot/CanvasProjection builder.
    - Guarantees `UNIQUE(run_id, seq)` starting at `seq = 1`.
    - Guarantees `(run_id, request_id)` ChatCommand idempotency.
    - Guarantees `mode='simulation'` events can NEVER enter a `mode='live'` run.
    - Replays & snapshots NEVER invoke SDK models or experiment runners.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        hub: EventSubscriberHub | None = None,
    ) -> None:
        self.db_path = db_path
        self.hub = hub or GLOBAL_EVENT_HUB

    def create_run(
        self,
        *,
        title: str | None = None,
        paper_id: str = "paper_A",
        mode: RunMode = "live",
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> Run:
        rid = run_id or f"rep_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        clean_title = (title or "Paper Reproduction Session").strip() or "Paper Reproduction Session"
        initial_resources = {
            "limits": {
                "max_active_agent_tasks": DEFAULT_LIMITS.max_active_agent_tasks,
                "max_parallel_experiments": DEFAULT_LIMITS.max_parallel_experiments,
                "max_subagent_depth": DEFAULT_LIMITS.max_subagent_depth,
                "max_debate_turns": DEFAULT_LIMITS.max_debate_turns,
                "exploration_budget_fraction": DEFAULT_LIMITS.exploration_budget_fraction,
            },
            "active_agent_tasks": 0,
            "queued_agent_tasks": 0,
            "active_experiments": 0,
            "queued_experiments": 0,
            "exploration_used_fraction": 0.0,
        }
        personas = build_initial_personas()

        with get_reproduce_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reproduce_runs (
                    run_id, title, paper_id, status, mode, workspace_id,
                    report_status, report_artifact_ids_json, missing_requirements_json,
                    resources_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'created', ?, ?, 'not_started', '[]', ?, ?, ?, ?)
                """,
                (
                    rid,
                    clean_title,
                    paper_id,
                    mode,
                    workspace_id,
                    jdumps(["verified_dataset", "executed_analysis", "evidence_gate_review", "report_file"]),
                    jdumps(initial_resources),
                    now,
                    now,
                ),
            )
            for p in personas:
                conn.execute(
                    """
                    INSERT INTO reproduce_personas (
                        run_id, persona_id, role, label, runtime_agent_id,
                        state, is_main, task_ids_json, updated_seq
                    ) VALUES (?, ?, ?, ?, NULL, 'idle', ?, '[]', 0)
                    """,
                    (rid, p.persona_id, p.role, p.label, 1 if p.is_main else 0),
                )

        return Run(
            run_id=rid,
            status="created",
            created_at=now,
            mode=mode,
            title=clean_title,
        )

    def get_run(self, run_id: str) -> Run | None:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reproduce_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        return Run(
            run_id=row["run_id"],
            status=row["status"],
            created_at=row["created_at"],
            mode=row["mode"],
            title=row["title"],
        )

    def get_run_meta(self, run_id: str) -> dict[str, Any] | None:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reproduce_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_persona_ids(self, run_id: str) -> list[str]:
        """Return all persisted persona IDs (baseline + dynamically spawned) for a run."""
        with get_reproduce_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT persona_id FROM reproduce_personas WHERE run_id = ? ORDER BY rowid ASC",
                (run_id,),
            ).fetchall()
        return [str(r["persona_id"]) for r in rows]

    def get_next_role_instance_index(self, run_id: str, role: str) -> int:
        """Return the next unused >=2 numeric suffix for dynamic spawning within `role`."""
        existing = self.list_persona_ids(run_id)
        prefix = f"persona_{role}_"
        max_idx = 1
        for pid in existing:
            if pid == f"persona_{role}":
                max_idx = max(max_idx, 1)
            elif pid.startswith(prefix):
                tail = pid[len(prefix) :]
                if tail.isdigit():
                    max_idx = max(max_idx, int(tail))
        return max_idx + 1

    def get_or_create_workspace_run(
        self,
        workspace_id: str,
        *,
        title: str | None = None,
        mode: RunMode = "live",
    ) -> Run:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reproduce_runs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT 1",
                (workspace_id,),
            ).fetchone()
        if row:
            return Run(
                run_id=row["run_id"],
                status=row["status"],
                created_at=row["created_at"],
                mode=row["mode"],
                title=row["title"],
            )
        return self.create_run(title=title, paper_id="paper_A", mode=mode, workspace_id=workspace_id)

    def get_idempotent_command(self, run_id: str, request_id: str) -> ChatResponse | None:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM reproduce_commands WHERE run_id = ? AND request_id = ?",
                (run_id, request_id),
            ).fetchone()
        if not row:
            return None
        return ChatResponse.model_validate(jloads(row["response_json"], {}))

    def record_command_atomic(
        self,
        *,
        run_id: str,
        request_id: str,
        text: str,
        selected_context: SelectedContext | None,
        attachment_ids: list[str],
        status: Literal_status,
        message: str,
        reason_code: str | None = None,
        mutate_fn: Callable[[sqlite3.Connection, str], HarnessEvent | None] | None = None,
    ) -> tuple[ChatResponse, bool, HarnessEvent | None]:
        """
        Atomically check `(run_id, request_id)` idempotency; if new, execute optional domain
        mutation + event append in the SAME SQLite transaction, commit, then broadcast SSE.
        Returns `(ChatResponse, was_duplicate, committed_event)`.
        """
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"
        emitted: HarnessEvent | None = None

        with get_reproduce_conn(self.db_path) as conn:
            existing = conn.execute(
                "SELECT response_json FROM reproduce_commands WHERE run_id = ? AND request_id = ?",
                (run_id, request_id),
            ).fetchone()
            if existing:
                return ChatResponse.model_validate(jloads(existing["response_json"], {})), True, None

            if mutate_fn is not None:
                emitted = mutate_fn(conn, command_id)

            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reproduce_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            last_seq = int(seq_row["max_seq"]) if seq_row else 0

            resp = ChatResponse(
                command_id=command_id,
                status=status,
                message=message,
                reason_code=reason_code,
                last_seq=last_seq,
            )
            conn.execute(
                """
                INSERT INTO reproduce_commands (
                    command_id, run_id, request_id, text, selected_context_json,
                    attachment_ids_json, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    run_id,
                    request_id,
                    text,
                    selected_context.model_dump_json() if selected_context else None,
                    jdumps(attachment_ids),
                    resp.model_dump_json(),
                    utc_now_iso(),
                ),
            )

        if emitted is not None:
            self.hub.publish(run_id, emitted)
        return resp, False, emitted

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        summary: str,
        actor_id: str | None = None,
        target_ids: list[str] | None = None,
        subject: EventSubject | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        mode: RunMode | None = None,
        source: EventSourceLiteral = "harness",
        state_mutator: Callable[[sqlite3.Connection, int], None] | None = None,
    ) -> HarnessEvent:
        """
        Append one `HarnessEvent` and run `state_mutator(conn, next_seq)` inside a single
        SQLite transaction; broadcast to SSE subscribers strictly after commit.
        """
        with get_reproduce_conn(self.db_path) as conn:
            ev = self.append_event_in_tx(
                conn,
                run_id,
                event_type=event_type,
                summary=summary,
                actor_id=actor_id,
                target_ids=target_ids,
                subject=subject,
                status=status,
                payload=payload,
                evidence_ids=evidence_ids,
                artifact_ids=artifact_ids,
                mode=mode,
                source=source,
                state_mutator=state_mutator,
            )
        self.hub.publish(run_id, ev)
        if event_type in ("run.completed", "run.failed"):
            self.hub.publish(run_id, None)
        return ev

    def append_event_in_tx(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        *,
        event_type: str,
        summary: str,
        actor_id: str | None = None,
        target_ids: list[str] | None = None,
        subject: EventSubject | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        mode: RunMode | None = None,
        source: EventSourceLiteral = "harness",
        state_mutator: Callable[[sqlite3.Connection, int], None] | None = None,
    ) -> HarnessEvent:
        run_row = conn.execute(
            "SELECT mode FROM reproduce_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            raise KeyError(f"Run '{run_id}' does not exist.")
        run_mode: RunMode = run_row["mode"]
        effective_mode: RunMode = mode or run_mode
        if run_mode == "live" and effective_mode == "simulation":
            raise ValueError("Refusing to mix mode='simulation' event into a LIVE scientific run.")
        if run_mode == "simulation" and effective_mode == "live":
            raise ValueError("Refusing to label event 'live' inside a SIMULATION run.")

        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reproduce_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        next_seq = int(seq_row["max_seq"]) + 1
        now = utc_now_iso()
        clean_payload = redact_secrets(payload or {})
        dedup_targets = list(dict.fromkeys(target_ids or []))
        dedup_ev = list(dict.fromkeys(evidence_ids or []))
        dedup_art = list(dict.fromkeys(artifact_ids or []))

        if state_mutator is not None:
            state_mutator(conn, next_seq)

        ev = HarnessEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            seq=next_seq,
            occurred_at=now,
            type=event_type,
            actor_id=actor_id,
            target_ids=dedup_targets,
            subject=subject,
            status=status,
            summary=str(redact_secrets(summary)),
            payload=clean_payload,
            evidence_ids=dedup_ev,
            artifact_ids=dedup_art,
            mode=effective_mode,
            source=source,
        )
        conn.execute(
            """
            INSERT INTO reproduce_events (
                run_id, seq, event_id, occurred_at, type, actor_id, target_ids_json,
                subject_json, status, summary, payload_json, evidence_ids_json,
                artifact_ids_json, mode, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ev.run_id,
                ev.seq,
                ev.event_id,
                ev.occurred_at,
                ev.type,
                ev.actor_id,
                jdumps(ev.target_ids),
                ev.subject.model_dump_json() if ev.subject else None,
                ev.status,
                ev.summary,
                jdumps(ev.payload),
                jdumps(ev.evidence_ids),
                jdumps(ev.artifact_ids),
                ev.mode,
                ev.source,
            ),
        )
        conn.execute(
            "UPDATE reproduce_runs SET updated_at = ? WHERE run_id = ?",
            (now, run_id),
        )
        return ev

    def add_chat_message(
        self,
        run_id: str,
        *,
        kind: str,
        role: str,
        text: str,
        related_refs: list[SelectedContext] | None = None,
        artifact_ids: list[str] | None = None,
        seq: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ChatActivityMessage:
        msg = ChatActivityMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            kind=kind,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            created_at=utc_now_iso(),
            text=str(redact_secrets(text)),
            related_refs=related_refs or [],
            artifact_ids=list(dict.fromkeys(artifact_ids or [])),
            seq=seq,
        )

        def _insert(c: sqlite3.Connection) -> None:
            c.execute(
                """
                INSERT INTO reproduce_messages (
                    message_id, run_id, kind, role, created_at, text,
                    related_refs_json, artifact_ids_json, seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.message_id,
                    msg.run_id,
                    msg.kind,
                    msg.role,
                    msg.created_at,
                    msg.text,
                    jdumps([r.model_dump() for r in msg.related_refs]),
                    jdumps(msg.artifact_ids),
                    msg.seq,
                ),
            )

        if conn is not None:
            _insert(conn)
        else:
            with get_reproduce_conn(self.db_path) as c:
                _insert(c)
        return msg

    def list_events(self, run_id: str, after: int = 0, up_to: int | None = None) -> list[HarnessEvent]:
        with get_reproduce_conn(self.db_path) as conn:
            if up_to is not None:
                rows = conn.execute(
                    "SELECT * FROM reproduce_events WHERE run_id = ? AND seq > ? AND seq <= ? ORDER BY seq ASC",
                    (run_id, after, up_to),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM reproduce_events WHERE run_id = ? AND seq > ? ORDER BY seq ASC",
                    (run_id, after),
                ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def list_messages(
        self,
        run_id: str,
        *,
        ref_kind: str | None = None,
        ref_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ChatMessagePage:
        start_row = int(cursor) if (cursor and cursor.isdigit()) else 0
        with get_reproduce_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM reproduce_messages WHERE run_id = ? AND row_id > ? ORDER BY row_id ASC",
                (run_id, start_row),
            ).fetchall()

        matched: list[tuple[int, ChatActivityMessage]] = []
        for r in rows:
            refs_raw = jloads(r["related_refs_json"], [])
            refs = [SelectedContext.model_validate(x) for x in refs_raw]
            art_ids: list[str] = jloads(r["artifact_ids_json"], [])
            if ref_kind and ref_id:
                hit = any(rf.kind == ref_kind and rf.id == ref_id for rf in refs)
                if not hit and ref_kind == "artifact" and ref_id in art_ids:
                    hit = True
                if not hit:
                    continue
            elif ref_kind:
                if not any(rf.kind == ref_kind for rf in refs):
                    continue
            elif ref_id:
                if not (any(rf.id == ref_id for rf in refs) or ref_id in art_ids):
                    continue

            matched.append(
                (
                    int(r["row_id"]),
                    ChatActivityMessage(
                        message_id=r["message_id"],
                        run_id=r["run_id"],
                        kind=r["kind"],
                        role=r["role"],
                        created_at=r["created_at"],
                        text=r["text"],
                        related_refs=refs,
                        artifact_ids=art_ids,
                        seq=r["seq"],
                    ),
                )
            )
            if len(matched) > limit:
                break

        has_more = len(matched) > limit
        page_slice = matched[:limit]
        next_cursor = str(page_slice[-1][0]) if (has_more and page_slice) else None
        return ChatMessagePage(
            messages=[m for _, m in page_slice],
            next_cursor=next_cursor,
        )

    def get_report_status(self, run_id: str) -> ReportStatus | None:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT report_status, report_artifact_ids_json, missing_requirements_json "
                "FROM reproduce_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return ReportStatus(
            status=row["report_status"],
            artifact_ids=jloads(row["report_artifact_ids_json"], []),
            missing_requirements=jloads(row["missing_requirements_json"], []),
        )

    def update_run_status(
        self,
        run_id: str,
        *,
        status: RunStatusLiteral | None = None,
        report_status: str | None = None,
        report_artifact_ids: list[str] | None = None,
        missing_requirements: list[str] | None = None,
        resources: dict[str, Any] | None = None,
    ) -> None:
        with get_reproduce_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM reproduce_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return
            conn.execute(
                """
                UPDATE reproduce_runs SET
                    status = ?,
                    report_status = ?,
                    report_artifact_ids_json = ?,
                    missing_requirements_json = ?,
                    resources_json = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status or row["status"],
                    report_status or row["report_status"],
                    jdumps(report_artifact_ids) if report_artifact_ids is not None else row["report_artifact_ids_json"],
                    jdumps(missing_requirements) if missing_requirements is not None else row["missing_requirements_json"],
                    jdumps(resources) if resources is not None else row["resources_json"],
                    utc_now_iso(),
                    run_id,
                ),
            )

    def get_snapshot(self, run_id: str, at_seq: int | None = None) -> RunSnapshot | None:
        """
        Pure read-only projection.
        - If `at_seq` is None or >= latest seq, reads current materialized state.
        - If `at_seq` < latest seq, deterministically folds events `1..at_seq` in memory
          WITHOUT calling any LLM, SDK, or subprocess tool!
        """
        with get_reproduce_conn(self.db_path) as conn:
            run_row = conn.execute(
                "SELECT * FROM reproduce_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not run_row:
                return None
            max_seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reproduce_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            latest_seq = int(max_seq_row["max_seq"]) if max_seq_row else 0

            if at_seq is not None and at_seq < latest_seq:
                ev_rows = conn.execute(
                    "SELECT * FROM reproduce_events WHERE run_id = ? AND seq <= ? ORDER BY seq ASC",
                    (run_id, at_seq),
                ).fetchall()
                events = [self._row_to_event(r) for r in ev_rows]
                return self._fold_historical_snapshot(run_id, run_row["mode"], at_seq, events)

            p_rows = conn.execute(
                "SELECT * FROM reproduce_personas WHERE run_id = ? ORDER BY rowid ASC", (run_id,)
            ).fetchall()
            t_rows = conn.execute(
                "SELECT * FROM reproduce_tasks WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
            ).fetchall()
            d_rows = conn.execute(
                "SELECT * FROM reproduce_debates WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
            ).fetchall()
            pod_rows = conn.execute(
                "SELECT * FROM reproduce_pods WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
            ).fetchall()
            c_rows = conn.execute(
                "SELECT * FROM reproduce_collaborations WHERE run_id = ? ORDER BY rowid ASC", (run_id,)
            ).fetchall()
            evid_rows = conn.execute(
                "SELECT * FROM reproduce_evidence WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
            ).fetchall()

        personas = [
            Persona(
                persona_id=r["persona_id"],
                role=r["role"],
                label=r["label"],
                runtime_agent_id=r["runtime_agent_id"],
                state=r["state"],
                is_main=bool(r["is_main"]),
                task_ids=jloads(r["task_ids_json"], []),
            )
            for r in p_rows
        ]
        active_collabs = [
            Collaboration(
                collaboration_id=r["collaboration_id"],
                kind=r["kind"],
                status=r["status"],
                topic=r["topic"],
                participant_persona_ids=jloads(r["participant_persona_ids_json"], []),
                ref=SourceRef.model_validate(jloads(r["ref_json"], {"kind": "collaboration", "id": r["collaboration_id"]})),
                last_event_seq=int(r["last_event_seq"]),
            )
            for r in c_rows
            if r["status"] in ("active", "ending")
        ]
        canvas = CanvasProjection(
            run_id=run_id,
            at_seq=latest_seq,
            mode=run_row["mode"],
            personas=personas,
            collaborations=active_collabs,
        )
        return RunSnapshot(
            run_id=run_id,
            at_seq=latest_seq,
            agents=[p.model_dump() for p in personas],
            tasks=[
                {
                    "id": r["task_id"],
                    "persona_id": r["persona_id"],
                    "role": r["role"],
                    "objective": r["objective"],
                    "status": r["status"],
                    "is_exploration": bool(r["is_exploration"]),
                    "artifact_ids": jloads(r["artifact_ids_json"], []),
                }
                for r in t_rows
            ],
            debates=[
                {
                    "id": r["debate_id"],
                    "claim_id": r["claim_id"],
                    "claim": r["claim"],
                    "participants": jloads(r["participants_json"], []),
                    "turns": jloads(r["turns_json"], []),
                    "proposed_experiments": jloads(r["proposed_experiments_json"], []),
                    "status": r["status"],
                    "resolution_summary": r["resolution_summary"],
                }
                for r in d_rows
            ],
            pods=[
                {
                    "id": r["pod_id"],
                    "research_question": r["research_question"],
                    "members": jloads(r["members_json"], []),
                    "budget_fraction": r["budget_fraction"],
                    "status": r["status"],
                    "outcome_summary": r["outcome_summary"],
                }
                for r in pod_rows
            ],
            evidence=[
                {
                    "id": r["evidence_id"],
                    "claim_id": r["claim_id"],
                    "dataset_manifest_id": r["dataset_manifest_id"],
                    "script_hash": r["script_hash"],
                    "result_artifact_ids": jloads(r["result_artifact_ids_json"], []),
                    "status": r["status"],
                    "reviewer_id": r["reviewer_id"],
                    "details": jloads(r["details_json"], {}),
                }
                for r in evid_rows
            ],
            resources=jloads(run_row["resources_json"], {}),
            canvas=canvas,
        )

    def _fold_historical_snapshot(
        self,
        run_id: str,
        mode: RunMode,
        target_seq: int,
        events: list[HarnessEvent],
    ) -> RunSnapshot:
        personas_map = {p.persona_id: p for p in build_initial_personas()}
        collabs_map: dict[str, tuple[int, Collaboration]] = {}
        tasks_map: dict[str, dict[str, Any]] = {}
        debates_map: dict[str, dict[str, Any]] = {}
        pods_map: dict[str, dict[str, Any]] = {}
        evidence_list: list[dict[str, Any]] = []

        for ev in events:
            pl = ev.payload
            if ev.type == "agent.spawned":
                pid = str(pl.get("persona_id") or "")
                rt_id = str(pl.get("runtime_agent_id") or ev.actor_id or "")
                if pid in personas_map:
                    personas_map[pid].runtime_agent_id = rt_id
                elif pid:
                    role_val = str(pl.get("role") or "support")
                    label_val = str(pl.get("label") or pid)
                    personas_map[pid] = Persona(
                        persona_id=pid,
                        role=role_val,
                        label=label_val,
                        runtime_agent_id=rt_id,
                        state="assigned",
                        is_main=bool(pl.get("is_main", False)),
                        task_ids=[],
                    )
            elif ev.type == "agent.status_changed":
                pid = str(pl.get("persona_id") or "")
                nxt = str(pl.get("next_state") or "idle")
                if pid in personas_map:
                    personas_map[pid].state = nxt  # type: ignore[assignment]
            elif ev.type == "task.created":
                tid = str(pl.get("task_id") or "")
                pid = str(pl.get("persona_id") or "")
                if tid:
                    tasks_map[tid] = {
                        "id": tid,
                        "persona_id": pid,
                        "role": str(pl.get("role") or ""),
                        "objective": str(pl.get("objective") or ev.summary),
                        "status": str(ev.status or "queued"),
                    }
                    if pid in personas_map and tid not in personas_map[pid].task_ids:
                        personas_map[pid].task_ids.append(tid)
            elif ev.type in ("task.started", "task.completed", "task.failed"):
                tid = str(pl.get("task_id") or (ev.subject.id if ev.subject else ""))
                if tid in tasks_map:
                    tasks_map[tid]["status"] = str(ev.status or ev.type.split(".")[-1])
            elif ev.type == "collaboration.started":
                cid = str(pl["collaboration_id"])
                rev = int(pl.get("revision", 1))
                sref = pl.get("source_ref") or {"kind": "collaboration", "id": cid}
                collabs_map[cid] = (
                    rev,
                    Collaboration(
                        collaboration_id=cid,
                        kind=pl.get("kind", "collaboration"),
                        status="active",
                        topic=str(pl.get("topic") or ev.summary),
                        participant_persona_ids=list(pl.get("participant_persona_ids") or []),
                        ref=SourceRef.model_validate(sref),
                        last_event_seq=ev.seq,
                    ),
                )
            elif ev.type == "collaboration.members_changed":
                cid = str(pl["collaboration_id"])
                rev = int(pl.get("revision", 2))
                if cid in collabs_map and rev > collabs_map[cid][0]:
                    prev_c = collabs_map[cid][1]
                    collabs_map[cid] = (
                        rev,
                        prev_c.model_copy(
                            update={
                                "participant_persona_ids": list(pl.get("participant_persona_ids") or []),
                                "last_event_seq": ev.seq,
                            }
                        ),
                    )
            elif ev.type == "collaboration.ended":
                cid = str(pl["collaboration_id"])
                collabs_map.pop(cid, None)
            elif ev.type == "debate.opened":
                did = str(pl.get("debate_id") or (ev.subject.id if ev.subject else ""))
                if did:
                    debates_map[did] = {"id": did, "claim": ev.summary, "status": "opened", "turns": []}
            elif ev.type == "debate.turn_added":
                did = str(pl.get("debate_id") or (ev.subject.id if ev.subject else ""))
                if did in debates_map:
                    debates_map[did]["turns"].append(pl)
                    debates_map[did]["status"] = "active"
            elif ev.type == "debate.closed":
                did = str(pl.get("debate_id") or (ev.subject.id if ev.subject else ""))
                if did in debates_map:
                    debates_map[did]["status"] = str(ev.status or "resolved")
            elif ev.type == "pod.started":
                pid_pod = str(pl.get("pod_id") or (ev.subject.id if ev.subject else ""))
                if pid_pod:
                    pods_map[pid_pod] = {"id": pid_pod, "research_question": ev.summary, "status": "active"}
            elif ev.type == "pod.closed":
                pid_pod = str(pl.get("pod_id") or (ev.subject.id if ev.subject else ""))
                if pid_pod in pods_map:
                    pods_map[pid_pod]["status"] = "closed"
            elif ev.type == "evidence.reviewed":
                evidence_list.append(
                    {"claim_id": pl.get("claim_id"), "status": pl.get("result"), "seq": ev.seq}
                )
            elif ev.type in ("run.completed", "run.failed"):
                collabs_map.clear()

        personas = list(personas_map.values())
        active_collabs = [c for _, c in collabs_map.values()]
        return RunSnapshot(
            run_id=run_id,
            at_seq=target_seq,
            agents=[p.model_dump() for p in personas],
            tasks=list(tasks_map.values()),
            debates=list(debates_map.values()),
            pods=list(pods_map.values()),
            evidence=evidence_list,
            resources={},
            canvas=CanvasProjection(
                run_id=run_id,
                at_seq=target_seq,
                mode=mode,
                personas=personas,
                collaborations=active_collabs,
            ),
        )

    @staticmethod
    def _row_to_event(r: Any) -> HarnessEvent:
        subj_raw = jloads(r["subject_json"], None)
        return HarnessEvent(
            event_id=r["event_id"],
            run_id=r["run_id"],
            seq=int(r["seq"]),
            occurred_at=r["occurred_at"],
            type=r["type"],
            actor_id=r["actor_id"],
            target_ids=jloads(r["target_ids_json"], []),
            subject=EventSubject.model_validate(subj_raw) if isinstance(subj_raw, dict) else None,
            status=r["status"],
            summary=r["summary"],
            payload=jloads(r["payload_json"], {}),
            evidence_ids=jloads(r["evidence_ids_json"], []),
            artifact_ids=jloads(r["artifact_ids_json"], []),
            mode=r["mode"],
            source=r["source"],
        )


Literal_status = Any
