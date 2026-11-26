from __future__ import annotations

import asyncio
import re
import sqlite3
from typing import Any

from ..schemas import (
    ChatCommand,
    ChatResponse,
    EventSubject,
    HarnessEvent,
    SelectedContext,
)
from ..storage.event_store import EventStore
from .coordinator import ResearchCoordinator

_READONLY_INQUIRY_PATTERNS = re.compile(
    r"(?:^(?:为什么|解释|说明|总结|查看|什么|有哪些|请问))|(?:^(?:why|explain|what|summarize|status|how did)\b)",
    flags=re.IGNORECASE,
)
_ACTION_TRIGGER_PATTERNS = re.compile(
    r"(?:复现|运行|重跑|执行|发起|召集|加入圆桌|五人|全员|\b(?:reproduce|run|re-run|rerun|execute|compute|convene|launch)\b)",
    flags=re.IGNORECASE,
)


_DYNAMIC_SPAWN_RE = re.compile(
    r"(?:动态调用|额外调用|并行调用|多调用|多个\s*(?:agent|分析|数据|论文|验证|支持)|再调用\s*\d+|spawn\s+\d+|spawn\s+(?:more|extra|parallel)\s+agents?)",
    flags=re.IGNORECASE,
)


class MainAgentGateway:
    """
    Sole user-facing command gateway (`POST /api/runs/{run_id}/chat`).
    - Enforces `(run_id, request_id)` idempotency.
    - Answers pure read-only inquiries (`200 answered`) from persisted Snapshot/Evidence without re-running tools.
    - Commits accepted commands atomically (`run.started`) before returning `202 accepted` and dispatching `ResearchCoordinator`.
    """

    def __init__(self, event_store: EventStore, coordinator: ResearchCoordinator) -> None:
        self.store = event_store
        self.coordinator = coordinator
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    def _spawn_bg(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    async def handle_chat(self, run_id: str, cmd: ChatCommand) -> tuple[int, ChatResponse]:
        cached = self.store.get_idempotent_command(run_id, cmd.request_id)
        if cached is not None:
            http_code = 200 if cached.status == "answered" else 202
            return http_code, cached

        run_meta = self.store.get_run_meta(run_id)
        if not run_meta:
            raise KeyError(f"Run '{run_id}' not found")

        text = cmd.text.strip()
        snap = self.store.get_snapshot(run_id)
        has_existing_results = bool(snap and (snap.evidence or snap.debates))

        # 1. Pure read-only inquiry over already-persisted evidence/debates
        if (
            has_existing_results
            and _READONLY_INQUIRY_PATTERNS.search(text)
            and not _ACTION_TRIGGER_PATTERNS.search(text)
        ):
            answer = self._synthesize_readonly_answer(text, cmd.selected_context, snap)

            def _record_inquiry(conn: sqlite3.Connection, _cid: str) -> HarnessEvent | None:
                refs = [cmd.selected_context] if cmd.selected_context else []
                self.store.add_chat_message(
                    run_id,
                    kind="message",
                    role="user",
                    text=text,
                    related_refs=refs,
                    artifact_ids=cmd.attachment_ids,
                    conn=conn,
                )
                self.store.add_chat_message(
                    run_id,
                    kind="message",
                    role="main",
                    text=answer,
                    related_refs=[SelectedContext(kind="agent", id="persona_main"), *refs],
                    conn=conn,
                )
                return None

            resp, _, _ = self.store.record_command_atomic(
                run_id=run_id,
                request_id=cmd.request_id,
                text=text,
                selected_context=cmd.selected_context,
                attachment_ids=cmd.attachment_ids,
                status="answered",
                message=answer,
                mutate_fn=_record_inquiry,
            )
            return 200, resp

        # 2. Reject over-budget explicit requests (e.g., asking for >20% exploration budget)
        if re.search(r"(?:50%|60%|80%|100%|无限预算|unlimited budget)", text, flags=re.IGNORECASE):
            msg_req = (
                "Requested exploration allocation exceeds Harness `exploration_budget_fraction = 20%` cap. "
                "Please confirm within <=20% exploration budget so the primary reproduction is protected."
            )
            resp, _, _ = self.store.record_command_atomic(
                run_id=run_id,
                request_id=cmd.request_id,
                text=text,
                selected_context=cmd.selected_context,
                attachment_ids=cmd.attachment_ids,
                status="requires_approval",
                message=msg_req,
                reason_code="EXPLORATION_BUDGET_CAP",
            )
            return 202, resp

        # 3. Follow-up dynamic multi-instance spawn or roundtable when primary run already completed
        is_dynamic_spawn_followup = bool(
            has_existing_results
            and run_meta["status"] == "completed"
            and _DYNAMIC_SPAWN_RE.search(text)
        )
        is_roundtable_followup = bool(
            not is_dynamic_spawn_followup
            and has_existing_results
            and run_meta["status"] == "completed"
            and re.search(r"(?:讨论|圆桌|五人|全员|解释.*分歧|roundtable|discuss|all agents|5-agent)", text, flags=re.IGNORECASE)
        )

        def _commit_execution(conn: sqlite3.Connection, command_id: str) -> HarnessEvent:
            refs = [cmd.selected_context] if cmd.selected_context else []
            self.store.add_chat_message(
                run_id,
                kind="message",
                role="user",
                text=text,
                related_refs=refs,
                artifact_ids=cmd.attachment_ids,
                conn=conn,
            )
            conn.execute(
                "UPDATE reproduce_runs SET status = 'running' WHERE run_id = ?",
                (run_id,),
            )
            ev = self.store.append_event_in_tx(
                conn,
                run_id,
                event_type="run.started",
                actor_id="persona_main",
                target_ids=["persona_main"],
                subject=EventSubject(kind="run", id=run_id),
                status="running",
                summary=f"Main Agent accepted command ({command_id}): {text[:90]}",
                payload={
                    "command_id": command_id,
                    "selected_context": cmd.selected_context.model_dump() if cmd.selected_context else None,
                    "attachment_ids": cmd.attachment_ids,
                },
                source="user",
            )
            return ev

        if is_dynamic_spawn_followup:
            ack_msg = "Main Agent accepted the dynamic multi-agent scaling request and is spawning specialist instances."
        elif is_roundtable_followup:
            ack_msg = "Main Agent accepted the methodological review request and is convening the research team."
        else:
            ack_msg = "Main Agent accepted the reproduction plan and dispatched Paper, Data, Analysis, and Validation under Scheduler governance."

        resp, was_dup, _ = self.store.record_command_atomic(
            run_id=run_id,
            request_id=cmd.request_id,
            text=text,
            selected_context=cmd.selected_context,
            attachment_ids=cmd.attachment_ids,
            status="accepted",
            message=ack_msg,
            mutate_fn=_commit_execution,
        )

        if not was_dup:
            if is_dynamic_spawn_followup:
                role_counts = self._infer_dynamic_spawn_counts(text)
                self._spawn_bg(
                    self.coordinator.execute_dynamic_spawn_followup(
                        run_id,
                        user_prompt=text,
                        role_counts=role_counts,
                    )
                )
            elif is_roundtable_followup:
                participants = self._infer_roundtable_members(text)
                self._spawn_bg(
                    self.coordinator.execute_team_roundtable_check(
                        run_id,
                        user_prompt=text,
                        participant_ids=participants,
                    )
                )
            else:
                self._spawn_bg(
                    self.coordinator.execute_reproduction_plan(
                        run_id,
                        user_prompt=text,
                        attachment_ids=cmd.attachment_ids,
                    )
                )

        return 202, resp

    @staticmethod
    def _infer_dynamic_spawn_counts(text: str) -> dict[str, int]:
        aliases = {
            "paper": ("paper", "论文"),
            "data": ("data", "数据"),
            "analysis": ("analysis", "分析"),
            "validation": ("validation", "验证", "检验"),
            "support": ("support", "支持", "报告"),
        }
        counts: dict[str, int] = {}
        num_match = re.search(r"(\d+)\s*(?:个|名)?", text)
        default_n = min(4, max(2, int(num_match.group(1)))) if num_match else 2
        for role, kws in aliases.items():
            if any(k in text.lower() for k in kws):
                counts[role] = default_n
        if not counts:
            counts["analysis"] = 2
        return counts

    @staticmethod
    def _infer_roundtable_members(text: str) -> list[str]:
        low = text.lower()
        if any(k in low for k in ("五人", "全员", "all", "5", "ellip", "椭圆")):
            return [
                "persona_paper",
                "persona_data",
                "persona_analysis",
                "persona_validation",
                "persona_support",
            ]
        matched: list[str] = []
        for role in ("paper", "data", "analysis", "validation", "support"):
            if role in low:
                matched.append(f"persona_{role}")
        if len(matched) >= 2:
            return matched
        return ["persona_analysis", "persona_validation", "persona_data"]

    @staticmethod
    def _synthesize_readonly_answer(
        text: str,
        selected_ctx: SelectedContext | None,
        snap: Any,
    ) -> str:
        ev_items = snap.evidence or []
        debates = snap.debates or []
        ctx_note = f" (Context: `{selected_ctx.kind}:{selected_ctx.id}`)" if selected_ctx else ""
        parts = [f"**Main Agent Read-Only Summary{ctx_note}**:"]
        if debates:
            d = debates[-1]
            parts.append(
                f"- **Debate `{d.get('id')}` (`{d.get('status')}`)**: {d.get('claim')} — {d.get('resolution_summary') or 'In active review.'}"
            )
        if ev_items:
            ev = ev_items[-1]
            gains = (ev.get("details") or {}).get("research_gains") or {}
            verified = gains.get("verified_insights") or []
            open_q = gains.get("open_questions") or []
            if verified:
                parts.append(f"- **Verified Finding**: {verified[0]}")
            if open_q:
                parts.append(f"- **Open / Unresolved Boundary**: {open_q[0]}")
        else:
            parts.append(f"- Current run is at event sequence `{snap.at_seq}` with `{len(snap.tasks)}` tracked task(s).")
        return "\n".join(parts)
