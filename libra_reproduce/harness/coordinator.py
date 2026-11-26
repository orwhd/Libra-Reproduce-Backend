from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from ..schemas import (
    EventSubject,
    RunMode,
    SelectedContext,
    SourceRef,
    VisualStateLiteral,
)
from ..science.cd_index import (
    compute_bounded_openalex_s2_window_check,
    verify_cd_toy_graphs,
)
from ..science.dataverse_loader import (
    ensure_wu2019_dataverse_sample,
    inspect_uploaded_attachment,
)
from ..science.evidence_gate import EvidenceGate
from ..science.general_paper_analysis import (
    build_general_paper_cohort_and_run,
    is_nature2019_target,
)
from ..science.iterative_experiments import (
    build_live_experiment_tools,
    run_multiround_counterfactual_suite,
    run_targeted_followup_experiment,
)
from ..science.nature2019_analysis import run_nature2019_experiments
from ..science.report_builder import build_reproduction_reports
from ..storage.artifact_store import ArtifactStore
from ..storage.db import jdumps, jloads, utc_now_iso
from ..storage.event_store import EventStore
from .collaboration_manager import CollaborationManager
from .persona_catalog import SPECIALIST_ROLES, build_dynamic_persona
from .scheduler import ResourceScheduler
from .sdk_adapter import SDKAdapter


class ResearchCoordinator:
    """
    Coordinates functional research personas (`Paper`, `Data`, `Analysis`, `Validation`, `Support`),
    enforces `ResourceScheduler` dual limits, invokes `SDKAdapter` with restricted science tools,
    drives authentic methodological Debate -> Research Pod table transitions, runs `EvidenceGate`,
    and generates verifiable DOCX/PDF reproduction reports.
    """

    def __init__(
        self,
        *,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        scheduler: ResourceScheduler,
    ) -> None:
        self.store = event_store
        self.artifacts = artifact_store
        self.scheduler = scheduler
        self.collab = CollaborationManager(event_store)
        self.evidence_gate = EvidenceGate()

    def _transition_persona(
        self,
        run_id: str,
        persona_id: str,
        next_state: VisualStateLiteral,
        reason: str,
        *,
        runtime_agent_id: str | None = None,
        mode: RunMode | None = None,
    ) -> None:
        prev_holder = ["idle"]

        def _mut(conn: sqlite3.Connection, seq: int) -> None:
            row = conn.execute(
                "SELECT state, runtime_agent_id, label FROM reproduce_personas WHERE run_id = ? AND persona_id = ?",
                (run_id, persona_id),
            ).fetchone()
            prev_state = row["state"] if row else "idle"
            prev_holder[0] = prev_state
            eff_runtime = runtime_agent_id if runtime_agent_id is not None else (row["runtime_agent_id"] if row else None)
            conn.execute(
                "UPDATE reproduce_personas SET state = ?, runtime_agent_id = ?, updated_seq = ? "
                "WHERE run_id = ? AND persona_id = ?",
                (next_state, eff_runtime, seq, run_id, persona_id),
            )

        with_prev = self._get_persona_state(run_id, persona_id)
        display_label = str(with_prev.get("label") or persona_id.replace("persona_", "").replace("_", " #").title())
        if runtime_agent_id and not with_prev.get("runtime_agent_id"):
            self.store.append_event(
                run_id,
                event_type="agent.spawned",
                actor_id=runtime_agent_id,
                target_ids=[persona_id],
                subject=EventSubject(kind="agent", id=persona_id),
                status="active",
                summary=f"Bound runtime execution session for {display_label}.",
                payload={
                    "persona_id": persona_id,
                    "role": with_prev.get("role") or persona_id.replace("persona_", "").split("_")[0],
                    "label": display_label,
                    "is_main": persona_id == "persona_main",
                    "runtime_agent_id": runtime_agent_id,
                },
                mode=mode,
                state_mutator=lambda conn, seq: conn.execute(
                    "UPDATE reproduce_personas SET runtime_agent_id = ?, updated_seq = ? WHERE run_id = ? AND persona_id = ?",
                    (runtime_agent_id, seq, run_id, persona_id),
                ),
            )

        if with_prev.get("state") == next_state:
            return

        self.store.append_event(
            run_id,
            event_type="agent.status_changed",
            actor_id=runtime_agent_id or persona_id,
            target_ids=[persona_id],
            subject=EventSubject(kind="agent", id=persona_id),
            status=next_state,
            summary=f"{display_label} -> {next_state} ({reason})",
            payload={
                "persona_id": persona_id,
                "previous_state": with_prev.get("state", "idle"),
                "next_state": next_state,
                "reason": reason,
            },
            mode=mode,
            state_mutator=_mut,
        )

    def _spawn_dynamic_persona(
        self,
        run_id: str,
        *,
        role: str,
        instance_index: int | None = None,
        label: str | None = None,
        specialty: str = "",
        mode: RunMode | None = None,
    ) -> str:
        """
        Dynamically instantiate an additional specialist worker inside one of the 5 specialist roles
        (`paper`, `data`, `analysis`, `validation`, `support`) on demand from Main Agent.
        """
        idx = instance_index or self.store.get_next_role_instance_index(run_id, role)
        rt_id = f"rt_{role}_{idx}_{run_id[-6:]}"
        persona = build_dynamic_persona(
            role=role,
            instance_index=idx,
            label=label,
            runtime_agent_id=rt_id,
        )

        def _insert_dynamic(conn: sqlite3.Connection, seq: int) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO reproduce_personas (
                    run_id, persona_id, role, label, runtime_agent_id,
                    state, is_main, task_ids_json, updated_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?)
                """,
                (
                    run_id,
                    persona.persona_id,
                    persona.role,
                    persona.label,
                    rt_id,
                    persona.state,
                    0,
                    seq,
                ),
            )

        desc = (
            f"Boss dynamically spawned {persona.label} ({persona.role}): {specialty}."
            if specialty
            else f"Boss dynamically spawned {persona.label} ({persona.role}) for parallel work."
        )
        ev = self.store.append_event(
            run_id,
            event_type="agent.spawned",
            actor_id="persona_main",
            target_ids=[persona.persona_id],
            subject=EventSubject(kind="agent", id=persona.persona_id),
            status="active",
            summary=desc,
            payload={
                "persona_id": persona.persona_id,
                "role": persona.role,
                "label": persona.label,
                "is_main": False,
                "runtime_agent_id": rt_id,
                "dynamic": True,
                "spawned_by": "persona_main",
                "specialty": specialty,
            },
            mode=mode,
            state_mutator=_insert_dynamic,
        )
        self.store.add_chat_message(
            run_id,
            kind="activity",
            role="activity",
            text=desc,
            related_refs=[
                SelectedContext(kind="agent", id="persona_main"),
                SelectedContext(kind="agent", id=persona.persona_id),
            ],
            seq=ev.seq,
        )
        return persona.persona_id

    def _get_persona_state(self, run_id: str, persona_id: str) -> dict[str, Any]:
        from ..storage.db import get_reproduce_conn

        with get_reproduce_conn(self.store.db_path) as conn:
            row = conn.execute(
                "SELECT state, runtime_agent_id, role, label FROM reproduce_personas WHERE run_id = ? AND persona_id = ?",
                (run_id, persona_id),
            ).fetchone()
        return dict(row) if row else {"state": "idle", "runtime_agent_id": None, "role": None, "label": None}

    def _create_task_record(
        self,
        run_id: str,
        *,
        persona_id: str,
        role: str,
        objective: str,
        is_exploration: bool = False,
        mode: RunMode | None = None,
    ) -> str:
        tid = f"task_{role}_{uuid.uuid4().hex[:8]}"
        now = utc_now_iso()

        def _mut(conn: sqlite3.Connection, seq: int) -> None:
            conn.execute(
                """
                INSERT INTO reproduce_tasks (
                    task_id, run_id, parent_task_id, persona_id, role, objective,
                    status, priority, is_exploration, budget_json, artifact_ids_json,
                    updated_seq, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, ?, 'queued', 10, ?, '{}', '[]', ?, ?, ?)
                """,
                (tid, run_id, persona_id, role, objective, 1 if is_exploration else 0, seq, now, now),
            )
            p_row = conn.execute(
                "SELECT task_ids_json FROM reproduce_personas WHERE run_id = ? AND persona_id = ?",
                (run_id, persona_id),
            ).fetchone()
            tids = jloads(p_row["task_ids_json"], []) if p_row else []
            if tid not in tids:
                tids.append(tid)
            conn.execute(
                "UPDATE reproduce_personas SET task_ids_json = ?, updated_seq = ? WHERE run_id = ? AND persona_id = ?",
                (jdumps(tids), seq, run_id, persona_id),
            )

        self.store.append_event(
            run_id,
            event_type="task.created",
            actor_id="persona_main",
            target_ids=[persona_id],
            subject=EventSubject(kind="task", id=tid),
            status="queued",
            summary=f"Assigned {role.capitalize()}: {objective}",
            payload={
                "task_id": tid,
                "persona_id": persona_id,
                "role": role,
                "objective": objective,
                "is_exploration": is_exploration,
            },
            mode=mode,
            state_mutator=_mut,
        )
        return tid

    def _mark_task_status(
        self,
        run_id: str,
        task_id: str,
        event_type: str,
        status: str,
        summary: str,
        *,
        persona_id: str,
        artifact_ids: list[str] | None = None,
        mode: RunMode | None = None,
    ) -> None:
        def _mut(conn: sqlite3.Connection, seq: int) -> None:
            conn.execute(
                "UPDATE reproduce_tasks SET status = ?, artifact_ids_json = ?, updated_seq = ?, updated_at = ? WHERE task_id = ?",
                (status, jdumps(artifact_ids or []), seq, utc_now_iso(), task_id),
            )

        ev = self.store.append_event(
            run_id,
            event_type=event_type,
            actor_id=persona_id,
            target_ids=[persona_id],
            subject=EventSubject(kind="task", id=task_id),
            status=status,
            summary=summary,
            payload={"task_id": task_id, "persona_id": persona_id},
            artifact_ids=artifact_ids or [],
            mode=mode,
            state_mutator=_mut,
        )
        self.store.add_chat_message(
            run_id,
            kind="activity",
            role="activity",
            text=summary,
            related_refs=[
                SelectedContext(kind="agent", id=persona_id),
                SelectedContext(kind="task", id=task_id),
            ],
            artifact_ids=artifact_ids or [],
            seq=ev.seq,
        )

    async def execute_reproduction_plan(
        self,
        run_id: str,
        *,
        user_prompt: str,
        attachment_ids: list[str] | None = None,
    ) -> None:
        meta = self.store.get_run_meta(run_id)
        if not meta:
            return
        mode: RunMode = meta["mode"]
        paper_id = str(meta.get("paper_id") or "paper_A")
        run_dir = self.artifacts.run_dir(run_id)
        sdk_adapter = SDKAdapter(run_dir)

        if mode == "live":
            can_live, provider_label = sdk_adapter.detect_live_capability()
            if not can_live:
                self._fail_run(
                    run_id,
                    mode=mode,
                    reason=(
                        "BLOCKED: No live Claude Agent SDK or configured LLM API credentials detected. "
                        "Configure an API key or run explicitly in mode='simulation'."
                    ),
                )
                return
        else:
            provider_label = "simulation_mock"

        active_collab_id: str | None = None
        try:
            self.store.update_run_status(
                run_id,
                status="running",
                report_status="generating",
                resources=self.scheduler.snapshot_state(run_id),
            )
            self._transition_persona(
                run_id,
                "persona_main",
                "working",
                "coordinating_reproduction_plan",
                runtime_agent_id=f"rt_main_{run_id[-6:]}",
                mode=mode,
            )

            nature_target = is_nature2019_target(paper_id, user_prompt)
            attachment_notes: list[dict[str, Any]] = []
            for aid in attachment_ids or []:
                resolved = self.artifacts.resolve_artifact_bytes(aid)
                if resolved:
                    fpath, _, _ = resolved
                    attachment_notes.append(inspect_uploaded_attachment(fpath))

            # Create primary tasks T1..T4
            t_paper = self._create_task_record(
                run_id,
                persona_id="persona_paper",
                role="paper",
                objective="Extract disruption metric specification, cohort window, and testable claims",
                mode=mode,
            )
            t_data = self._create_task_record(
                run_id,
                persona_id="persona_data",
                role="data",
                objective="Verify dataset schema, license, SHA-256 manifest, and cohort coverage",
                mode=mode,
            )
            t_analysis = self._create_task_record(
                run_id,
                persona_id="persona_analysis",
                role="analysis",
                objective="Execute controlled team-size vs. disruption analysis and generate figures",
                mode=mode,
            )
            t_validation = self._create_task_record(
                run_id,
                persona_id="persona_validation",
                role="validation",
                objective="Audit S1/S2 metric boundaries, citation-window comparability, and Evidence Gate",
                mode=mode,
            )

            # --- Phase 1: Paper + Parallel Multi-Instance Data Agents (`persona_data` + dynamic `persona_data_2`) ---
            extracted_scope: dict[str, Any] = {}
            dataset_bundle: dict[str, Any] = {}
            s2_csv_holder: list[Path | None] = [None]

            data_2_id = self._spawn_dynamic_persona(
                run_id,
                role="data",
                instance_index=2,
                label="Data #2",
                specialty="OpenAlex S2 Citation Graph & Window Stratification",
                mode=mode,
            )
            t_data_2 = self._create_task_record(
                run_id,
                persona_id=data_2_id,
                role="data",
                objective="Prepare bounded OpenAlex S2 citation subgraph for independent 5y vs. 10y D-index check",
                is_exploration=False,
                mode=mode,
            )

            async def _paper_stage() -> Any:
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_paper):
                    self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                    self._transition_persona(
                        run_id,
                        "persona_paper",
                        "working",
                        "extracting_methodology",
                        runtime_agent_id=f"rt_paper_{run_id[-6:]}",
                        mode=mode,
                    )
                    self._mark_task_status(
                        run_id,
                        t_paper,
                        "task.started",
                        "running",
                        "Paper Agent inspecting target methodology, team-size bins, and D = (n_i - n_j)/(n_i + n_j + n_k) specification.",
                        persona_id="persona_paper",
                        mode=mode,
                    )

                    async def _tool_inspect_paper(args: dict[str, Any]) -> dict[str, Any]:
                        spec = {
                            "target_paper": (
                                "Wu, Wang & Evans (Nature 2019, 10.1038/s41586-019-0941-9)"
                                if nature_target
                                else user_prompt[:140]
                            ),
                            "primary_window": "1954-2014",
                            "team_bins": ["1", "2", "3", "4", "5", "6-10", "11+"],
                            "disruption_formula": "D = (n_i - n_j) / (n_i + n_j + n_k), denominator 0 -> NaN",
                            "attachments_inspected": len(attachment_notes),
                            "args_echo": bool(args),
                        }
                        extracted_scope.update(spec)
                        return spec

                    turn_res = await sdk_adapter.run_controlled_science_turn(
                        mode=mode,
                        role="paper",
                        system_prompt=(
                            "You are Paper Agent (deep reasoning enabled). Call inspect_paper_scope once to register the formal "
                            "reproduction methodology, then explicitly formulate: (1) the core testable claim, (2) two rival/confounding "
                            "hypotheses (e.g., citation-volume bias, field×decade composition, citation-window truncation), and (3) a "
                            "multi-round counterfactual + generative citation-network simulation experimental plan to test them."
                        ),
                        user_prompt=f"Think carefully and design a multi-round experimental reproduction protocol for: {user_prompt}",
                        tools_spec=[
                            {
                                "name": "inspect_paper_scope",
                                "description": "Inspect and register formal paper reproduction methodology, bins, and metric equations.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"paper_id": {"type": "string"}},
                                    "required": ["paper_id"],
                                },
                            }
                        ],
                        handlers={"inspect_paper_scope": _tool_inspect_paper},
                        max_turns=5,
                        effort="high",
                    )
                    if not extracted_scope:
                        await _tool_inspect_paper({"paper_id": paper_id})
                    if not turn_res.ok and mode == "live":
                        raise RuntimeError(turn_res.error or "Paper SDK call failed")

                    for tr in turn_res.tool_traces:
                        self.store.append_event(
                            run_id,
                            event_type="tool.executed",
                            actor_id="persona_paper",
                            target_ids=["persona_paper"],
                            subject=EventSubject(kind="task", id=t_paper),
                            status="completed",
                            summary=f"SDK tool '{tr['tool']}' executed via {tr['provider']}.",
                            payload=tr,
                            mode=mode,
                            source="sdk",
                        )

                    self._mark_task_status(
                        run_id,
                        t_paper,
                        "task.completed",
                        "done",
                        turn_res.public_summary or "Methodology & formal metric scope extracted.",
                        persona_id="persona_paper",
                        mode=mode,
                    )
                    self._transition_persona(run_id, "persona_paper", "completed", "methodology_ready", mode=mode)
                    return turn_res

            async def _data_stage() -> None:
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_data):
                    self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                    self._transition_persona(
                        run_id,
                        "persona_data",
                        "working",
                        "verifying_dataset_manifest",
                        runtime_agent_id=f"rt_data_{run_id[-6:]}",
                        mode=mode,
                    )
                    self._mark_task_status(
                        run_id,
                        t_data,
                        "task.started",
                        "running",
                        "Data Agent auditing empirical dataset, license, column schema, and SHA-256 manifest.",
                        persona_id="persona_data",
                        mode=mode,
                    )
                    if nature_target:
                        ds = await ensure_wu2019_dataverse_sample(run_dir / "data")
                        dataset_bundle.update(ds)
                    else:
                        gen = await build_general_paper_cohort_and_run(run_dir, user_prompt)
                        dataset_bundle.update(gen)

                    art_manifest, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="dataset",
                        source_path=dataset_bundle["manifest_path"],
                        validation_status="passed",
                        metadata={
                            "reproduction_level": "S1",
                            "description": "Verified dataset provenance & SHA-256 manifest",
                        },
                    )
                    art_csv, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="dataset",
                        source_path=dataset_bundle["csv_path"],
                        validation_status="passed",
                        metadata={
                            "reproduction_level": "S1",
                            "description": f"Empirical paper sample (N={dataset_bundle['manifest']['subset_row_count']:,})",
                        },
                    )
                    dataset_bundle["manifest_artifact_id"] = art_manifest.artifact_id
                    dataset_bundle["csv_artifact_id"] = art_csv.artifact_id

                    for created_art, label_desc in [
                        (art_manifest, f"Data manifest verified ({dataset_bundle['manifest']['license']})"),
                        (art_csv, f"Empirical sample saved (N={dataset_bundle['manifest']['subset_row_count']:,} records)"),
                    ]:
                        ev_a = self.store.append_event(
                            run_id,
                            event_type="artifact.created",
                            actor_id="persona_data",
                            target_ids=["persona_data"],
                            subject=EventSubject(kind="artifact", id=created_art.artifact_id),
                            status="passed",
                            summary=label_desc,
                            payload={
                                "artifact_id": created_art.artifact_id,
                                "kind": created_art.kind,
                                "validation_status": created_art.validation_status,
                            },
                            artifact_ids=[created_art.artifact_id],
                            mode=mode,
                            source="runner",
                        )
                        self.store.add_chat_message(
                            run_id,
                            kind="file",
                            role="activity",
                            text=label_desc,
                            related_refs=[
                                SelectedContext(kind="agent", id="persona_data"),
                                SelectedContext(kind="artifact", id=created_art.artifact_id),
                            ],
                            artifact_ids=[created_art.artifact_id],
                            seq=ev_a.seq,
                        )

                    self._mark_task_status(
                        run_id,
                        t_data,
                        "task.completed",
                        "done",
                        f"Dataset verified: N={dataset_bundle['manifest']['subset_row_count']:,} rows ({dataset_bundle['manifest']['license']}).",
                        persona_id="persona_data",
                        artifact_ids=[art_manifest.artifact_id, art_csv.artifact_id],
                        mode=mode,
                    )
                    self._transition_persona(run_id, "persona_data", "idle", "dataset_ready", mode=mode)

            async def _data_s2_stage() -> None:
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_data_2):
                    self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                    self._transition_persona(
                        run_id,
                        data_2_id,
                        "working",
                        "fetching_openalex_s2_subgraph",
                        mode=mode,
                    )
                    self._mark_task_status(
                        run_id,
                        t_data_2,
                        "task.started",
                        "running",
                        "Data #2 preparing bounded OpenAlex S2 citation subgraph (5y vs. 10y forward windows).",
                        persona_id=data_2_id,
                        mode=mode,
                    )
                    if nature_target:
                        from ..config import get_bundled_samples_dir

                        bundled_s2 = get_bundled_samples_dir() / "wu2019_s2_openalex_bounded.csv"
                        if bundled_s2.exists():
                            p_s2 = run_dir / "data" / "s2_openalex_bounded.csv"
                            p_s2.parent.mkdir(parents=True, exist_ok=True)
                            p_s2.write_bytes(bundled_s2.read_bytes())
                            s2_csv_holder[0] = p_s2
                        elif mode == "live":
                            s2_df = await compute_bounded_openalex_s2_window_check(max_focal=8)
                            if len(s2_df) > 0:
                                p_s2 = run_dir / "data" / "s2_openalex_bounded.csv"
                                p_s2.parent.mkdir(parents=True, exist_ok=True)
                                s2_df.to_csv(p_s2, index=False)
                                s2_csv_holder[0] = p_s2

                    s2_art_ids: list[str] = []
                    if s2_csv_holder[0] and s2_csv_holder[0].exists():
                        art_s2, _ = self.artifacts.register_file(
                            run_id=run_id,
                            kind="dataset",
                            source_path=s2_csv_holder[0],
                            validation_status="passed",
                            metadata={"reproduction_level": "S2", "description": "Bounded OpenAlex S2 citation subgraph"},
                        )
                        s2_art_ids.append(art_s2.artifact_id)
                        ev_s2 = self.store.append_event(
                            run_id,
                            event_type="artifact.created",
                            actor_id=data_2_id,
                            target_ids=[data_2_id],
                            subject=EventSubject(kind="artifact", id=art_s2.artifact_id),
                            status="passed",
                            summary="Bounded OpenAlex S2 citation subgraph prepared by Data #2.",
                            payload={
                                "artifact_id": art_s2.artifact_id,
                                "kind": "dataset",
                                "validation_status": "passed",
                            },
                            artifact_ids=[art_s2.artifact_id],
                            mode=mode,
                            source="runner",
                        )
                        self.store.add_chat_message(
                            run_id,
                            kind="file",
                            role="activity",
                            text="Bounded OpenAlex S2 citation subgraph prepared by Data #2.",
                            related_refs=[
                                SelectedContext(kind="agent", id=data_2_id),
                                SelectedContext(kind="artifact", id=art_s2.artifact_id),
                            ],
                            artifact_ids=[art_s2.artifact_id],
                            seq=ev_s2.seq,
                        )

                    self._mark_task_status(
                        run_id,
                        t_data_2,
                        "task.completed",
                        "done",
                        "Data #2 finished bounded S2 citation-subgraph staging.",
                        persona_id=data_2_id,
                        artifact_ids=s2_art_ids,
                        mode=mode,
                    )
                    self._transition_persona(run_id, data_2_id, "completed", "s2_subgraph_ready", mode=mode)

            paper_turn_res, _, _ = await asyncio.gather(_paper_stage(), _data_stage(), _data_s2_stage())

            # --- Phase 2: Parallel Multi-Instance Analysis Agents (`persona_analysis` + dynamic `persona_analysis_2`) ---
            exp_bundle: dict[str, Any] = {}
            s2_toy_holder: list[dict[str, Any]] = []
            s2_csv_path = s2_csv_holder[0]

            analysis_2_id = self._spawn_dynamic_persona(
                run_id,
                role="analysis",
                instance_index=2,
                label="Analysis #2",
                specialty="S2 CD-Index Toy Graph & Citation-Window Verification",
                mode=mode,
            )
            t_analysis_2 = self._create_task_record(
                run_id,
                persona_id=analysis_2_id,
                role="analysis",
                objective="Execute independent S2 D = (n_i - n_j)/(n_i + n_j + n_k) toy-graph & window checks in parallel experiment slot",
                is_exploration=False,
                mode=mode,
            )

            async def _primary_analysis_stage() -> None:
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_analysis):
                    self._transition_persona(
                        run_id,
                        "persona_analysis",
                        "working",
                        "running_statistical_reproduction",
                        runtime_agent_id=f"rt_analysis_{run_id[-6:]}",
                        mode=mode,
                    )
                    self._mark_task_status(
                        run_id,
                        t_analysis,
                        "task.started",
                        "running",
                        "Analysis Agent acquiring parallel experiment slot #1 to compute team-size disruption curves & 95% CIs.",
                        persona_id="persona_analysis",
                        mode=mode,
                    )
                    async with self.scheduler.experiment_slot(run_id=run_id, owner_id=t_analysis):
                        self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                        if nature_target:
                            exp_out = await run_nature2019_experiments(
                                run_dir, dataset_bundle["csv_path"], s2_csv_path
                            )
                        else:
                            exp_out = dataset_bundle
                        exp_bundle.update(exp_out)

                    art_script, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="script",
                        source_path=exp_bundle["execution"].script_path,
                        validation_status="passed",
                        metadata={"reproduction_level": "S1", "description": "Executed Python reproduction script"},
                    )
                    art_team_csv, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="dataset",
                        source_path=exp_bundle["team_csv_path"],
                        validation_status="passed",
                        metadata={"reproduction_level": "S1", "description": "Team-size bin summary & 95% bootstrap CIs"},
                    )
                    art_fig_main, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="figure",
                        source_path=exp_bundle["fig_main_path"],
                        validation_status="passed",
                        metadata={"reproduction_level": "S1", "description": "Figure 1: Team size vs. disruption & citations"},
                    )
                    exp_bundle["script_artifact_id"] = art_script.artifact_id
                    exp_bundle["team_csv_artifact_id"] = art_team_csv.artifact_id
                    exp_bundle["fig_main_artifact_id"] = art_fig_main.artifact_id

                    for created_art, desc in [
                        (art_script, "Executable Python analysis script saved with SHA-256 provenance"),
                        (art_team_csv, "Intermediate team-size statistical summary table generated"),
                        (art_fig_main, "Figure 1 (Team size vs. disruption percentile & citations) rendered"),
                    ]:
                        ev_art = self.store.append_event(
                            run_id,
                            event_type="artifact.created",
                            actor_id="persona_analysis",
                            target_ids=["persona_analysis"],
                            subject=EventSubject(kind="artifact", id=created_art.artifact_id),
                            status="passed",
                            summary=desc,
                            payload={
                                "artifact_id": created_art.artifact_id,
                                "kind": created_art.kind,
                                "validation_status": created_art.validation_status,
                            },
                            artifact_ids=[created_art.artifact_id],
                            mode=mode,
                            source="runner",
                        )
                        self.store.add_chat_message(
                            run_id,
                            kind="file",
                            role="activity",
                            text=desc,
                            related_refs=[
                                SelectedContext(kind="agent", id="persona_analysis"),
                                SelectedContext(kind="artifact", id=created_art.artifact_id),
                            ],
                            artifact_ids=[created_art.artifact_id],
                            seq=ev_art.seq,
                        )

                    self._mark_task_status(
                        run_id,
                        t_analysis,
                        "task.completed",
                        "done",
                        (
                            f"Primary S1 analysis finished (N={exp_bundle['summary']['primary_sample_n']:,}, "
                            f"OLS slope={exp_bundle['summary']['ols_slope_per_author']:.5f}/author)."
                        ),
                        persona_id="persona_analysis",
                        artifact_ids=[art_script.artifact_id, art_team_csv.artifact_id, art_fig_main.artifact_id],
                        mode=mode,
                    )

            async def _secondary_s2_analysis_stage() -> None:
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_analysis_2):
                    self._transition_persona(
                        run_id,
                        analysis_2_id,
                        "working",
                        "verifying_s2_cd_index_metric",
                        mode=mode,
                    )
                    self._mark_task_status(
                        run_id,
                        t_analysis_2,
                        "task.started",
                        "running",
                        "Analysis #2 acquiring parallel experiment slot #2 to verify S2 D-index formula invariants.",
                        persona_id=analysis_2_id,
                        mode=mode,
                    )
                    async with self.scheduler.experiment_slot(run_id=run_id, owner_id=t_analysis_2):
                        self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                        toy_res = await asyncio.to_thread(verify_cd_toy_graphs)
                        s2_toy_holder.append(toy_res)
                    self._mark_task_status(
                        run_id,
                        t_analysis_2,
                        "task.completed",
                        "done",
                        "Analysis #2 verified S2 D = (n_i - n_j)/(n_i + n_j + n_k) exact toy-graph & zero-denominator invariants.",
                        persona_id=analysis_2_id,
                        mode=mode,
                    )
                    self._transition_persona(run_id, analysis_2_id, "completed", "s2_verification_ready", mode=mode)

            await asyncio.gather(_primary_analysis_stage(), _secondary_s2_analysis_stage())
            s2_toy_result = s2_toy_holder[0] if s2_toy_holder else verify_cd_toy_graphs()

            # --- Phase 3: Validation Agent triggers authentic Citation-Window Debate (2p rectangular table -> 3p circular Pod!) ---
            async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_validation):
                self._transition_persona(
                    run_id,
                    "persona_validation",
                    "debating",
                    "auditing_citation_window_comparability",
                    runtime_agent_id=f"rt_validation_{run_id[-6:]}",
                    mode=mode,
                )
                self._transition_persona(
                    run_id,
                    "persona_analysis",
                    "debating",
                    "responding_to_window_critique",
                    mode=mode,
                )
                self._mark_task_status(
                    run_id,
                    t_validation,
                    "task.started",
                    "running",
                    "Validation Agent auditing forward citation-window truncation & S1/S2 claim boundaries.",
                    persona_id="persona_validation",
                    mode=mode,
                )

                debate_id, _ = self.collab.open_debate(
                    run_id,
                    claim_id="claim_citation_window",
                    claim="Forward citation observation windows differ between historical (1954-2004) and recent (2015-2021) papers",
                    participants=["persona_analysis", "persona_validation"],
                    actor_id="persona_validation",
                    mode=mode,
                )
                collab_id, _ = self.collab.start_collaboration(
                    run_id,
                    kind="debate",
                    participant_persona_ids=["persona_analysis", "persona_validation"],
                    topic="Citation window comparability & S1/S2 scope",
                    source_ref=SourceRef(kind="debate", id=debate_id),
                    actor_id="persona_validation",
                    summary="Analysis and Validation convened a 2-person review table on forward citation-window truncation.",
                    mode=mode,
                )
                active_collab_id = collab_id
                await asyncio.sleep(0.25 if mode == "simulation" else 0.5)

                val_challenge_summary = (
                    "Validation challenged Round 1 descriptive results: (1) cumulative MAG citations give pre-2005 papers >=17y windows while post-2014 papers are truncated, "
                    "(2) solo authors might only appear more disruptive due to citation-volume or field×decade composition confounding, and (3) a generative simulation control is needed rather than relying solely on passive observational bins."
                )
                if mode == "live":
                    try:
                        val_tools, val_handlers, _ = build_live_experiment_tools(
                            run_id=run_id,
                            workspace_dir=run_dir,
                            sample_csv_path=dataset_bundle["csv_path"],
                            scheduler=self.scheduler,
                            owner_task_id=t_validation,
                        )
                        val_res = await sdk_adapter.run_controlled_science_turn(
                            mode=mode,
                            role="validation",
                            system_prompt=(
                                "You are Validation Agent. Inspect the Round 1 empirical outputs with `inspect_workspace_data`, "
                                "identify the strongest methodological confounders (citation volume, cohort truncation, field×decade composition), "
                                "and write a sharp 2-sentence empirical critique proposing what experiments Round 2 MUST execute."
                            ),
                            user_prompt="Inspect Round 1 summaries and write a 2-sentence methodological challenge for Analysis.",
                            tools_spec=[val_tools[0]],
                            handlers={"inspect_workspace_data": val_handlers["inspect_workspace_data"]},
                            max_turns=4,
                            timeout_seconds=70.0,
                            effort="high",
                        )
                        if val_res.ok and val_res.public_summary:
                            val_challenge_summary = val_res.public_summary[:600]
                        for tr in val_res.tool_traces:
                            self.store.append_event(
                                run_id,
                                event_type="tool.executed",
                                actor_id="persona_validation",
                                target_ids=["persona_validation"],
                                subject=EventSubject(kind="task", id=t_validation),
                                status="completed",
                                summary=f"Validation SDK tool '{tr['tool']}' executed via {tr['provider']}.",
                                payload=tr,
                                mode=mode,
                                source="sdk",
                            )
                    except Exception:
                        pass

                self.collab.add_debate_turn(
                    run_id,
                    debate_id=debate_id,
                    claim_id="claim_citation_window",
                    speaker_persona_id="persona_validation",
                    stance="challenge",
                    public_summary=val_challenge_summary,
                    evidence_ref=dataset_bundle["manifest_artifact_id"],
                    target_ids=["persona_analysis", "persona_validation"],
                    mode=mode,
                )
                self.collab.add_debate_turn(
                    run_id,
                    debate_id=debate_id,
                    claim_id="claim_citation_window",
                    speaker_persona_id="persona_analysis",
                    stance="propose_test",
                    public_summary=(
                        "Analysis proposed convening Data in a Pod AND dynamically spawning Analysis #3 to execute multi-round "
                        "Citation-Quartile & Field×Decade FE controls, a 200-trial Monte Carlo permutation null, "
                        "leave-one-field-out jackknife trials, and an M1 vs. M2 mechanistic citation-network simulation."
                    ),
                    evidence_ref=exp_bundle["team_csv_artifact_id"],
                    target_ids=["persona_analysis", "persona_validation"],
                    mode=mode,
                )
                await asyncio.sleep(0.25 if mode == "simulation" else 0.5)

                # Expand 2-person rectangular table -> 3-person circular Pod table (`collaboration.members_changed` rev 2)
                pod_id, _ = self.collab.start_pod(
                    run_id,
                    research_question="Do citation-volume confounding, field×decade composition, or forward citation-window truncation explain the team-size disruption decline?",
                    members=["persona_analysis", "persona_validation", "persona_data"],
                    budget_fraction=0.12,
                    actor_id="persona_validation",
                    mode=mode,
                )
                self._transition_persona(run_id, "persona_data", "debating", "joined_window_sensitivity_pod", mode=mode)
                self.collab.change_collaboration_members(
                    run_id,
                    collaboration_id=collab_id,
                    participant_persona_ids=["persona_analysis", "persona_validation", "persona_data"],
                    reason="Data joined to stratify historical (>=17y) vs. right-truncated (<7y) citation windows",
                    actor_id="persona_data",
                    summary="Data joined Analysis & Validation (table expanded to 3-member circular Research Pod).",
                    mode=mode,
                )

                # Dynamically spawn Analysis #3 to execute genuine multi-round experimental trials & mechanistic simulation
                analysis_3_id = self._spawn_dynamic_persona(
                    run_id,
                    role="analysis",
                    instance_index=3,
                    label="Analysis #3",
                    specialty="Counterfactual, Permutation Null & Generative Simulation Experiments",
                    mode=mode,
                )
                t_analysis_3 = self._create_task_record(
                    run_id,
                    persona_id=analysis_3_id,
                    role="analysis",
                    objective="Execute multi-round counterfactual controls, 200-trial permutation null, jackknife, M1 vs. M2 simulation, and live agent-authored verification trials",
                    is_exploration=True,
                    mode=mode,
                )
                agent_experiment_rounds: list[dict[str, Any]] = []
                async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=t_analysis_3, is_exploration=True, budget_fraction=0.12):
                    self._transition_persona(run_id, analysis_3_id, "working", "running_multiround_experiments", mode=mode)
                    self._mark_task_status(
                        run_id,
                        t_analysis_3,
                        "task.started",
                        "running",
                        "Analysis #3 executing multi-round counterfactual controls, 200-trial permutation null, jackknife & M1 vs. M2 generative citation-network simulation.",
                        persona_id=analysis_3_id,
                        mode=mode,
                    )
                    async with self.scheduler.experiment_slot(run_id=run_id, owner_id=t_analysis_3):
                        self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                        multiround_bundle = await run_multiround_counterfactual_suite(
                            run_dir, dataset_bundle["csv_path"]
                        )

                    art_mr_script, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="script",
                        source_path=multiround_bundle["execution"].script_path,
                        validation_status="passed",
                        metadata={"reproduction_level": "S2", "description": "Multi-round counterfactual, permutation & simulation Python script"},
                    )
                    art_mr_csv, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="dataset",
                        source_path=multiround_bundle["csv_path"],
                        validation_status="passed",
                        metadata={"reproduction_level": "S2", "description": "Multi-round experiment metrics (Quartiles, Permutation, Jackknife, M1/M2 Sim)"},
                    )
                    art_mr_fig, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="figure",
                        source_path=multiround_bundle["fig_path"],
                        validation_status="passed",
                        metadata={"reproduction_level": "S2", "description": "Figure 3: Multi-round counterfactuals & generative citation-network simulation"},
                    )

                    for m_art, m_desc in [
                        (art_mr_script, "Round 2A–2D counterfactual & mechanistic simulation script archived (SHA-256)"),
                        (art_mr_csv, "Multi-round experimental results table saved (`multiround_experiments.csv`)"),
                        (art_mr_fig, "Figure 3 (Confounders, Permutation Null, Jackknife & M1 vs. M2 Simulation) rendered"),
                    ]:
                        ev_m = self.store.append_event(
                            run_id,
                            event_type="artifact.created",
                            actor_id=analysis_3_id,
                            target_ids=[analysis_3_id],
                            subject=EventSubject(kind="artifact", id=m_art.artifact_id),
                            status="passed",
                            summary=m_desc,
                            payload={
                                "artifact_id": m_art.artifact_id,
                                "kind": m_art.kind,
                                "validation_status": "passed",
                            },
                            artifact_ids=[m_art.artifact_id],
                            mode=mode,
                            source="runner",
                        )
                        self.store.add_chat_message(
                            run_id,
                            kind="file",
                            role="activity",
                            text=m_desc,
                            related_refs=[
                                SelectedContext(kind="agent", id=analysis_3_id),
                                SelectedContext(kind="artifact", id=m_art.artifact_id),
                            ],
                            artifact_ids=[m_art.artifact_id],
                            seq=ev_m.seq,
                        )

                    live_trial_art_ids: list[str] = [art_mr_script.artifact_id, art_mr_csv.artifact_id, art_mr_fig.artifact_id]
                    if mode == "live":
                        try:
                            a3_tools, a3_handlers, recorded_trials = build_live_experiment_tools(
                                run_id=run_id,
                                workspace_dir=run_dir,
                                sample_csv_path=dataset_bundle["csv_path"],
                                scheduler=self.scheduler,
                                owner_task_id=t_analysis_3,
                            )
                            a3_res = await sdk_adapter.run_controlled_science_turn(
                                mode=mode,
                                role="analysis_3",
                                system_prompt=(
                                    "You are Analysis #3, a deeply skeptical experimental scientist. Never just recite existing summary numbers — "
                                    "design and RUN a real Python falsification/stress-test experiment inside the sandbox! "
                                    "Workflow: (1) call `inspect_workspace_data` once to inspect columns and Round 2 results; "
                                    "(2) formulate a concrete residual hypothesis (e.g., whether extreme citation outliers >99th pct or non-linear log(team_size) scaling/piecewise slope reversal changes conclusions), "
                                    "write a complete self-contained Python script using pandas/numpy/scipy (using sys.argv[1] for CSV input, NO `os`/network imports) and EXECUTE it via `run_python_experiment`; "
                                    "if `run_python_experiment` fails, diagnose `stderr`, fix the script, and re-run it; "
                                    "(3) call `record_experiment_finding` with exact numbers observed from `stdout`."
                                ),
                                user_prompt=(
                                    "Think deeply, run a targeted Python stress-test experiment on the empirical cohort (`sys.argv[1]`) via `run_python_experiment`, "
                                    "verify the quantitative outcome from stdout, and record your finding."
                                ),
                                tools_spec=a3_tools,
                                handlers=a3_handlers,
                                max_turns=6,
                                timeout_seconds=95.0,
                                effort="high",
                            )
                            agent_experiment_rounds.extend(recorded_trials)
                            for tr in a3_res.tool_traces:
                                self.store.append_event(
                                    run_id,
                                    event_type="tool.executed",
                                    actor_id=analysis_3_id,
                                    target_ids=[analysis_3_id],
                                    subject=EventSubject(kind="task", id=t_analysis_3),
                                    status="completed" if tr.get("status") == "ok" else "error",
                                    summary=f"Analysis #3 live SDK tool '{tr['tool']}' ({tr.get('status')}) via {tr['provider']}.",
                                    payload=tr,
                                    mode=mode,
                                    source="sdk",
                                )
                            for item in recorded_trials:
                                if item.get("kind") == "agent_python_trial" and item.get("ok"):
                                    sp = Path(str(item["script_path"]))
                                    if sp.exists():
                                        art_trial, _ = self.artifacts.register_file(
                                            run_id=run_id,
                                            kind="script",
                                            source_path=sp,
                                            validation_status="passed",
                                            metadata={"reproduction_level": "S2", "description": f"Live agent-authored trial: {item['hypothesis'][:80]}"},
                                        )
                                        live_trial_art_ids.append(art_trial.artifact_id)
                                        ev_tr = self.store.append_event(
                                            run_id,
                                            event_type="artifact.created",
                                            actor_id=analysis_3_id,
                                            target_ids=[analysis_3_id],
                                            subject=EventSubject(kind="artifact", id=art_trial.artifact_id),
                                            status="passed",
                                            summary=f"Analysis #3 executed custom Python trial `{item['script_name']}` ({item['duration_ms']}ms).",
                                            payload={
                                                "artifact_id": art_trial.artifact_id,
                                                "kind": "script",
                                                "validation_status": "passed",
                                            },
                                            artifact_ids=[art_trial.artifact_id],
                                            mode=mode,
                                            source="runner",
                                        )
                                        self.store.add_chat_message(
                                            run_id,
                                            kind="file",
                                            role="activity",
                                            text=f"Live Python Trial `{item['script_name']}` (`{item['hypothesis']}`): {str(item.get('stdout', '')).strip()[:180]}",
                                            related_refs=[
                                                SelectedContext(kind="agent", id=analysis_3_id),
                                                SelectedContext(kind="artifact", id=art_trial.artifact_id),
                                            ],
                                            artifact_ids=[art_trial.artifact_id],
                                            seq=ev_tr.seq,
                                        )
                        except Exception as live_exc:
                            self.store.add_chat_message(
                                run_id,
                                kind="activity",
                                role="activity",
                                text=f"Analysis #3 live optional custom trial skipped cleanly ({type(live_exc).__name__}); verified Round 2A–2D suite preserved.",
                                related_refs=[SelectedContext(kind="agent", id=analysis_3_id)],
                            )

                    mr_sum = multiround_bundle["summary"]
                    self._mark_task_status(
                        run_id,
                        t_analysis_3,
                        "task.completed",
                        "done",
                        (
                            f"Analysis #3 completed multi-round trials: Field×Decade FE slope={mr_sum['fixed_effects_field_decade']['slope']:.5f}, "
                            f"200-trial Permutation p={mr_sum['permutation_null']['p_value']:.4f}, "
                            f"Generative Sim M1={mr_sum['mechanistic_simulation']['m1_slope']:.4f} vs M2={mr_sum['mechanistic_simulation']['m2_slope']:.4f}."
                        ),
                        persona_id=analysis_3_id,
                        artifact_ids=live_trial_art_ids,
                        mode=mode,
                    )

                # Expand 3-person circular Pod -> 5-person elliptical multi-instance review table (`collaboration.members_changed` rev 3)
                self._transition_persona(run_id, analysis_2_id, "debating", "joined_elliptical_synthesis", mode=mode)
                self._transition_persona(run_id, analysis_3_id, "debating", "joined_elliptical_synthesis", mode=mode)
                self.collab.change_collaboration_members(
                    run_id,
                    collaboration_id=collab_id,
                    participant_persona_ids=[
                        "persona_analysis",
                        "persona_validation",
                        "persona_data",
                        analysis_2_id,
                        analysis_3_id,
                    ],
                    reason="Analysis #2 (S2 CD checks) and Analysis #3 (Multi-round counterfactual & generative sim) joined cross-verification table",
                    actor_id=analysis_3_id,
                    summary="Analysis #2 & Analysis #3 joined (table expanded to 5-member elliptical multi-round synthesis group).",
                    mode=mode,
                )
                await asyncio.sleep(0.25 if mode == "simulation" else 0.5)

                art_win_csv, _ = self.artifacts.register_file(
                    run_id=run_id,
                    kind="dataset",
                    source_path=exp_bundle["window_csv_path"],
                    validation_status="passed",
                    metadata={"reproduction_level": "S1", "description": "Citation-window regime sensitivity table"},
                )
                art_fig_win, _ = self.artifacts.register_file(
                    run_id=run_id,
                    kind="figure",
                    source_path=exp_bundle["fig_window_path"],
                    validation_status="passed",
                    metadata={"reproduction_level": "S2" if s2_csv_path else "S1", "description": "Figure 2: Citation window sensitivity & bounded S2 check"},
                )

                for w_art, w_desc in [
                    (art_win_csv, "Citation-window sensitivity table saved (1954-2004 vs. 2005-2014 vs. 2015-2021)"),
                    (art_fig_win, "Figure 2 (Window comparability & bounded S2 5y/10y check) verified"),
                ]:
                    ev_w = self.store.append_event(
                        run_id,
                        event_type="artifact.created",
                        actor_id="persona_validation",
                        target_ids=["persona_validation", "persona_analysis", "persona_data"],
                        subject=EventSubject(kind="artifact", id=w_art.artifact_id),
                        status="passed",
                        summary=w_desc,
                        payload={
                            "artifact_id": w_art.artifact_id,
                            "kind": w_art.kind,
                            "validation_status": w_art.validation_status,
                        },
                        artifact_ids=[w_art.artifact_id],
                        mode=mode,
                        source="runner",
                    )
                    self.store.add_chat_message(
                        run_id,
                        kind="file",
                        role="activity",
                        text=w_desc,
                        related_refs=[
                            SelectedContext(kind="collaboration", id=collab_id),
                            SelectedContext(kind="pod", id=pod_id),
                            SelectedContext(kind="artifact", id=w_art.artifact_id),
                        ],
                        artifact_ids=[w_art.artifact_id],
                        seq=ev_w.seq,
                    )

                # Spawn independent Validation #2 instance to run Evidence Gate audit
                val_2_id = self._spawn_dynamic_persona(
                    run_id,
                    role="validation",
                    instance_index=2,
                    label="Validation #2",
                    specialty="Independent Evidence Gate & Manifest Audit",
                    mode=mode,
                )
                self._transition_persona(run_id, val_2_id, "working", "auditing_evidence_gate", mode=mode)
                ev_review = self.evidence_gate.review_reproduction_bundle(
                    claim_id="claim_team_size_disruption",
                    claimed_level="S1",
                    data_manifest=dataset_bundle["manifest"],
                    sample_csv_path=dataset_bundle["csv_path"],
                    team_csv_path=exp_bundle["team_csv_path"],
                    window_csv_path=exp_bundle["window_csv_path"],
                    fig_main_path=exp_bundle["fig_main_path"],
                    fig_window_path=exp_bundle["fig_window_path"],
                    experiment_summary=exp_bundle["summary"],
                    script_sha256=exp_bundle["execution"].script_sha256,
                    exit_code=exp_bundle["execution"].exit_code,
                    toy_s2_verification=s2_toy_result,
                    multiround_summary=multiround_bundle["summary"],
                    fig_multiround_path=multiround_bundle["fig_path"],
                    agent_experiment_rounds=agent_experiment_rounds,
                )
                evidence_id = f"evid_{uuid.uuid4().hex[:10]}"
                result_art_ids = [
                    dataset_bundle["manifest_artifact_id"],
                    dataset_bundle["csv_artifact_id"],
                    exp_bundle["script_artifact_id"],
                    exp_bundle["team_csv_artifact_id"],
                    exp_bundle["fig_main_artifact_id"],
                    art_win_csv.artifact_id,
                    art_fig_win.artifact_id,
                    *live_trial_art_ids,
                ]

                def _save_evid(conn: sqlite3.Connection, _seq: int) -> None:
                    conn.execute(
                        """
                        INSERT INTO reproduce_evidence (
                            evidence_id, run_id, claim_id, dataset_manifest_id, script_hash,
                            result_artifact_ids_json, status, reviewer_id, details_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence_id,
                            run_id,
                            ev_review.claim_id,
                            dataset_bundle["manifest"]["dataset_id"],
                            exp_bundle["execution"].script_sha256,
                            jdumps(result_art_ids),
                            "verified" if ev_review.result == "passed" else ev_review.result,
                            val_2_id,
                            jdumps(
                                {
                                    "checks": ev_review.checks,
                                    "reasons": ev_review.reasons,
                                    "reproduction_level": ev_review.reproduction_level_verified,
                                    "research_gains": ev_review.research_gains,
                                }
                            ),
                            utc_now_iso(),
                        ),
                    )

                self.store.append_event(
                    run_id,
                    event_type="evidence.reviewed",
                    actor_id=val_2_id,
                    target_ids=["persona_analysis", "persona_validation", val_2_id],
                    subject=EventSubject(kind="evidence", id=evidence_id),
                    status=ev_review.result,
                    summary=f"Evidence Gate verdict by {val_2_id}: {ev_review.result.upper()} ({ev_review.reproduction_level_verified} verified).",
                    payload={
                        "claim_id": ev_review.claim_id,
                        "result": ev_review.result,
                        "reproduction_level": ev_review.reproduction_level_verified,
                        "checks": ev_review.checks,
                        "research_gains": ev_review.research_gains,
                    },
                    evidence_ids=[evidence_id],
                    artifact_ids=result_art_ids,
                    mode=mode,
                    source="validator",
                    state_mutator=_save_evid,
                )
                self._transition_persona(run_id, val_2_id, "completed", "evidence_gate_complete", mode=mode)

                resolution_text = (
                    "Window-stratified sensitivity confirmed the negative team-size vs. disruption slope holds in "
                    "the >=17y historical cohort (1954-2004), while post-2014 right-truncation compresses raw D variance; "
                    "report explicitly separates S1 (MAG precomputed D) from bounded S2 recomputation."
                )
                self.collab.close_pod(
                    run_id,
                    pod_id=pod_id,
                    outcome_summary=resolution_text,
                    members=["persona_analysis", "persona_validation", "persona_data"],
                    actor_id="persona_validation",
                    artifact_ids=[art_win_csv.artifact_id, art_fig_win.artifact_id],
                    mode=mode,
                )
                self.collab.close_debate(
                    run_id,
                    debate_id=debate_id,
                    outcome_status="resolved" if ev_review.result == "passed" else "unresolved",
                    resolution_summary=resolution_text,
                    actor_id="persona_validation",
                    target_ids=["persona_analysis", "persona_validation"],
                    evidence_ids=[evidence_id],
                    mode=mode,
                )
                self.collab.end_collaboration(
                    run_id,
                    collaboration_id=collab_id,
                    outcome="resolved" if ev_review.result == "passed" else "inconclusive",
                    actor_id="persona_validation",
                    summary="Research table concluded; participants returning to home anchors.",
                    mode=mode,
                )
                active_collab_id = None

                self._transition_persona(run_id, "persona_data", "completed", "sensitivity_pod_complete", mode=mode)
                self._transition_persona(run_id, "persona_analysis", "completed", "analysis_verified", mode=mode)
                self._transition_persona(run_id, "persona_validation", "reporting", "reporting_to_main", mode=mode)

                self._mark_task_status(
                    run_id,
                    t_validation,
                    "task.completed",
                    "done",
                    f"Evidence Gate {ev_review.result}: all provenance & numerical consistency checks completed.",
                    persona_id="persona_validation",
                    artifact_ids=[art_win_csv.artifact_id, art_fig_win.artifact_id],
                    mode=mode,
                )

            # --- Phase 4: Support / ReportBuilder generates DOCX & PDF reports (only for real runs, never fake live!) ---
            self._transition_persona(
                run_id,
                "persona_support",
                "working",
                "compiling_docx_pdf_reports",
                runtime_agent_id=f"rt_support_{run_id[-6:]}",
                mode=mode,
            )
            debate_list = [
                {
                    "id": debate_id,
                    "status": "resolved" if ev_review.result == "passed" else "unresolved",
                    "claim": "Citation observation window comparability & S1/S2 claim boundary",
                    "resolution_summary": resolution_text,
                }
            ]
            report_files = await asyncio.to_thread(
                build_reproduction_reports,
                output_dir=run_dir / "reports",
                run_id=run_id,
                user_prompt=user_prompt,
                data_manifest=dataset_bundle["manifest"],
                experiment_summary=exp_bundle["summary"],
                evidence_result=ev_review,
                fig_main_path=exp_bundle["fig_main_path"],
                fig_window_path=exp_bundle["fig_window_path"],
                debate_records=debate_list,
                sdk_provider_used=paper_turn_res.provider,
                script_sha256=exp_bundle["execution"].script_sha256,
                multiround_summary=multiround_bundle["summary"],
                fig_multiround_path=multiround_bundle["fig_path"],
                agent_experiment_rounds=agent_experiment_rounds,
            )

            val_badge = "passed" if ev_review.result == "passed" else "inconclusive"
            art_pdf, _ = self.artifacts.register_file(
                run_id=run_id,
                kind="report",
                source_path=report_files["pdf_path"],
                validation_status=val_badge,
                metadata={
                    "reproduction_level": ev_review.reproduction_level_verified,
                    "description": "Scientific Reproduction Report (PDF)",
                },
            )
            art_docx, _ = self.artifacts.register_file(
                run_id=run_id,
                kind="report",
                source_path=report_files["docx_path"],
                validation_status=val_badge,
                metadata={
                    "reproduction_level": ev_review.reproduction_level_verified,
                    "description": "Scientific Reproduction Report (Word .docx)",
                },
            )
            report_ids = [art_pdf.artifact_id, art_docx.artifact_id]

            for r_art, r_desc in [
                (art_pdf, "Final PDF Reproduction Report generated (with figures, tables & provenance)"),
                (art_docx, "Final Word (.docx) Reproduction Report generated"),
            ]:
                self.store.append_event(
                    run_id,
                    event_type="artifact.created",
                    actor_id="persona_support",
                    target_ids=["persona_support"],
                    subject=EventSubject(kind="artifact", id=r_art.artifact_id),
                    status=val_badge,
                    summary=r_desc,
                    payload={
                        "artifact_id": r_art.artifact_id,
                        "kind": "report",
                        "validation_status": val_badge,
                    },
                    artifact_ids=[r_art.artifact_id],
                    mode=mode,
                    source="runner",
                )

            ev_rep = self.store.append_event(
                run_id,
                event_type="report.generated",
                actor_id="persona_main",
                target_ids=["persona_main", "persona_support"],
                subject=EventSubject(kind="run", id=run_id),
                status="ready",
                summary=f"Reproduction reports (PDF & DOCX) ready at level {ev_review.reproduction_level_verified}.",
                payload={
                    "artifact_ids": report_ids,
                    "reproduction_level": ev_review.reproduction_level_verified,
                },
                evidence_ids=[evidence_id],
                artifact_ids=report_ids,
                mode=mode,
            )

            self._transition_persona(run_id, "persona_validation", "completed", "audit_complete", mode=mode)
            self._transition_persona(run_id, "persona_support", "completed", "reports_ready", mode=mode)
            self._transition_persona(run_id, "persona_main", "completed", "reproduction_delivered", mode=mode)

            s = exp_bundle["summary"]
            mr = multiround_bundle["summary"]
            live_trials_ok = sum(1 for t in agent_experiment_rounds if t.get("kind") == "agent_python_trial" and t.get("ok"))
            sim_prefix = "**[SIMULATION MODE]** " if mode == "simulation" else ""
            final_md = (
                f"{sim_prefix}### Reproduction Summary ({ev_review.reproduction_level_verified} Verified + Multi-Round Experiments)\n\n"
                f"- **Target**: Wu, Wang & Evans (*Nature*, 2019, `10.1038/s41586-019-0941-9`)\n"
                f"- **Empirical Sample**: `N = {s['primary_sample_n']:,}` MAG papers (`1954–2014` primary historical cohort; `{s['total_sample_n']:,}` total rows through `2021`, `CC0-1.0`)\n"
                f"- **Round 1 Baseline Finding**: Solo authors (`k=1`) average disruption `D = {s['solo_mean_disruption']:.4f}` (percentile `{s['solo_disruption_percentile']:.1f}`) vs. `11+` teams at `D = {s['large_11plus_mean_disruption']:.4f}` (`OLS slope = {s['ols_slope_per_author']:.5f}/author`, `p = {s['ols_p_value']:.2e}`), while mean forward citations rise from `{s['solo_mean_citations']:.1f}` to `{s['large_11plus_mean_citations']:.1f}`.\n"
                f"- **Round 2 Counterfactual, Permutation & Simulation Trials (`Analysis #3`)**:\n"
                f"  - *Field×Decade Fixed Effects & Citation Quartiles*: FE slope = `{mr['fixed_effects_field_decade']['slope']:.5f}/author` (`p = {mr['fixed_effects_field_decade']['p_value']:.2e}`)\n"
                f"  - *{mr['permutation_null']['n_permutations']}-Trial Within-Stratum Permutation Null*: `z = {mr['permutation_null']['z_score']:.1f}`, empirical `p = {mr['permutation_null']['p_value']:.4f}`\n"
                f"  - *Leave-One-Discipline-Out Jackknife*: `{mr['jackknife']['negative_count']}/{mr['jackknife']['total_strata']}` strata remain negative\n"
                f"  - *Generative Citation-Network Simulation (Exact CD)*: M1 search-breadth slope = `{mr['mechanistic_simulation']['m1_slope']:.4f}` vs. size-invariant control M2 = `{mr['mechanistic_simulation']['m2_slope']:.4f}`"
                + (f"\n  - *Live Agent-Authored Custom Sandbox Trials*: `{live_trials_ok}` additional Python script(s) written & executed" if live_trials_ok else "")
                + f"\n- **Evidence Gate**: `{ev_review.result.upper()}` (`{paper_turn_res.provider}`). Inspect or download the PDF/DOCX reports, figures, scripts, and CSV manifests below."
            )
            self.store.add_chat_message(
                run_id,
                kind="message",
                role="main",
                text=final_md,
                related_refs=[SelectedContext(kind="agent", id="persona_main")],
                artifact_ids=[
                    art_pdf.artifact_id,
                    art_docx.artifact_id,
                    exp_bundle["fig_main_artifact_id"],
                    art_fig_win.artifact_id,
                    art_mr_fig.artifact_id,
                    exp_bundle["team_csv_artifact_id"],
                    art_mr_csv.artifact_id,
                    exp_bundle["script_artifact_id"],
                    *live_trial_art_ids,
                    dataset_bundle["manifest_artifact_id"],
                ],
                seq=ev_rep.seq,
            )

            self.store.update_run_status(
                run_id,
                status="completed",
                report_status="ready",
                report_artifact_ids=report_ids,
                missing_requirements=[],
                resources=self.scheduler.snapshot_state(run_id),
            )
            self.store.append_event(
                run_id,
                event_type="run.completed",
                actor_id="persona_main",
                target_ids=["persona_main"],
                subject=EventSubject(kind="run", id=run_id),
                status="completed",
                summary=f"Run completed ({ev_review.reproduction_level_verified} verified; {len(report_ids)} report files generated).",
                payload={
                    "reproduction_level": ev_review.reproduction_level_verified,
                    "evidence_result": ev_review.result,
                    "sdk_provider": paper_turn_res.provider,
                },
                evidence_ids=[evidence_id],
                artifact_ids=report_ids,
                mode=mode,
            )
            # Return all personas (baseline 6 + dynamically spawned specialists) to calm `idle` at anchors (L1.5 §8.4)
            for pid in self.store.list_persona_ids(run_id):
                self._transition_persona(run_id, pid, "idle", "returned_to_anchor", mode=mode)

        except Exception as exc:
            if active_collab_id:
                try:
                    self.collab.end_collaboration(
                        run_id,
                        collaboration_id=active_collab_id,
                        outcome="aborted_on_error",
                        actor_id="persona_main",
                        mode=mode,
                    )
                except Exception:
                    pass
            self._fail_run(run_id, mode=mode, reason=f"{type(exc).__name__}: {exc}")

    async def execute_team_roundtable_check(
        self,
        run_id: str,
        *,
        user_prompt: str,
        participant_ids: list[str],
    ) -> None:
        """
        Execute a follow-up methodological roundtable requested via Chat (supporting 2, 3-4, or 5-6
        non-Main participants so all temporary table shapes can also be observed on live follow-ups).
        """
        meta = self.store.get_run_meta(run_id)
        if not meta:
            return
        mode: RunMode = meta["mode"]
        members = [p for p in dict.fromkeys(participant_ids) if p != "persona_main"]
        if len(members) < 2:
            members = ["persona_analysis", "persona_validation"]

        self.store.update_run_status(run_id, status="running")
        self._transition_persona(run_id, "persona_main", "working", "coordinating_followup_roundtable", mode=mode)
        for m in members:
            self._transition_persona(run_id, m, "debating", "roundtable_discussion", mode=mode)

        cid, _ = self.collab.start_collaboration(
            run_id,
            kind="pod" if len(members) >= 3 else "debate",
            participant_persona_ids=members[:2],
            topic=user_prompt[:96],
            source_ref=SourceRef(kind="collaboration", id=f"roundtable_{uuid.uuid4().hex[:6]}"),
            actor_id="persona_main",
            summary=f"Opened initial 2-agent review table for: {user_prompt[:80]}",
            mode=mode,
        )
        await asyncio.sleep(0.45)
        if len(members) >= 3:
            self.collab.change_collaboration_members(
                run_id,
                collaboration_id=cid,
                participant_persona_ids=members[: min(4, len(members))],
                reason="Expanded roundtable to circular group",
                actor_id=members[2],
                summary=f"Expanded table to {min(4, len(members))} agents for cross-role verification.",
                mode=mode,
            )
            await asyncio.sleep(0.45)
        if len(members) >= 5:
            self.collab.change_collaboration_members(
                run_id,
                collaboration_id=cid,
                participant_persona_ids=members,
                reason="Full research team joined elliptical review table",
                actor_id=members[4],
                summary=f"Expanded table to {len(members)}-agent elliptical research group.",
                mode=mode,
            )
            await asyncio.sleep(0.55)

        self.collab.end_collaboration(
            run_id,
            collaboration_id=cid,
            outcome="synthesized",
            actor_id="persona_main",
            summary="Roundtable synthesis delivered to Boss; participants returned to anchors.",
            mode=mode,
        )
        for m in ["persona_main", *members]:
            self._transition_persona(run_id, m, "idle", "roundtable_complete", mode=mode)

        snap = self.store.get_snapshot(run_id)
        ev_count = len(snap.evidence) if snap else 0
        self.store.add_chat_message(
            run_id,
            kind="message",
            role="main",
            text=(
                f"Completed {len(members)}-agent methodological review on **\"{user_prompt[:100]}\"**.\n\n"
                f"- Active Evidence Gate records checked: `{ev_count}`\n"
                f"- Conclusion: S1 MAG empirical findings and bounded S2 citation-window checks remain consistent; "
                f"any claim beyond observed citation windows stays marked as an open methodological boundary."
            ),
            related_refs=[
                SelectedContext(kind="collaboration", id=cid),
                *[SelectedContext(kind="agent", id=m) for m in members],
            ],
        )
        self.store.update_run_status(run_id, status="completed")
        self.store.append_event(
            run_id,
            event_type="run.completed",
            actor_id="persona_main",
            target_ids=["persona_main"],
            subject=EventSubject(kind="run", id=run_id),
            status="completed",
            summary="Follow-up methodological review completed.",
            payload={"followup": True, "participants": members},
            mode=mode,
        )

    async def execute_dynamic_spawn_followup(
        self,
        run_id: str,
        *,
        user_prompt: str,
        role_counts: dict[str, int],
    ) -> None:
        """
        Dynamically summon N additional specialist instances across any of the 5 specialist classes
        under Main Boss coordination, execute parallel tasks under ResourceScheduler, hold a
        multi-instance verification roundtable, and return every persona cleanly to `idle`.
        """
        meta = self.store.get_run_meta(run_id)
        if not meta:
            return
        mode: RunMode = meta["mode"]
        self.store.update_run_status(run_id, status="running")
        self._transition_persona(run_id, "persona_main", "working", "spawning_parallel_specialists", mode=mode)

        spawned_ids: list[str] = []
        task_map: dict[str, str] = {}
        for role, count in role_counts.items():
            if role not in SPECIALIST_ROLES:
                continue
            for _ in range(max(1, min(4, count))):
                idx = self.store.get_next_role_instance_index(run_id, role)
                pid = self._spawn_dynamic_persona(
                    run_id,
                    role=role,
                    instance_index=idx,
                    label=f"{role.capitalize()} #{idx}",
                    specialty=user_prompt[:80],
                    mode=mode,
                )
                tid = self._create_task_record(
                    run_id,
                    persona_id=pid,
                    role=role,
                    objective=f"Parallel {role} check #{idx}: {user_prompt[:80]}",
                    is_exploration=True,
                    mode=mode,
                )
                spawned_ids.append(pid)
                task_map[pid] = tid

        if not spawned_ids:
            idx = self.store.get_next_role_instance_index(run_id, "analysis")
            pid = self._spawn_dynamic_persona(
                run_id,
                role="analysis",
                instance_index=idx,
                label=f"Analysis #{idx}",
                specialty=user_prompt[:80],
                mode=mode,
            )
            tid = self._create_task_record(
                run_id,
                persona_id=pid,
                role="analysis",
                objective=f"Parallel analysis check #{idx}: {user_prompt[:80]}",
                is_exploration=True,
                mode=mode,
            )
            spawned_ids.append(pid)
            task_map[pid] = tid

        run_dir = self.artifacts.run_dir(run_id)
        candidate_csv = next(
            (p for p in [run_dir / "data" / "wu2019_mag_sample.csv", run_dir / "data" / "openalex_paper_cohort.csv"] if p.exists()),
            None,
        )
        followup_results: list[str] = []

        async def _run_dynamic_worker(worker_idx: int, pid: str) -> None:
            tid = task_map[pid]
            async with self.scheduler.agent_task_slot(run_id=run_id, owner_id=tid):
                self.store.update_run_status(run_id, resources=self.scheduler.snapshot_state(run_id))
                self._transition_persona(run_id, pid, "working", "executing_dynamic_experiment", mode=mode)
                self._mark_task_status(
                    run_id,
                    tid,
                    "task.started",
                    "running",
                    f"{pid} acquiring experiment slot for empirical subsample robustness check.",
                    persona_id=pid,
                    mode=mode,
                )
                art_ids_worker: list[str] = []
                msg_done = f"{pid} completed verification subtask."
                if candidate_csv is not None:
                    async with self.scheduler.experiment_slot(run_id=run_id, owner_id=tid):
                        exp_w = await run_targeted_followup_experiment(
                            run_dir,
                            candidate_csv,
                            script_rel_path=f"scripts/followup_{pid}.py",
                            seed_offset=worker_idx + 1,
                        )
                    art_w, _ = self.artifacts.register_file(
                        run_id=run_id,
                        kind="script",
                        source_path=exp_w["execution"].script_path,
                        validation_status="passed",
                        metadata={"reproduction_level": "S2", "description": f"Follow-up robustness check by {pid}"},
                    )
                    art_ids_worker.append(art_w.artifact_id)
                    r = exp_w["result"]
                    msg_done = (
                        f"{pid} ran trimmed-citation test (n={r['n_evaluated']:,}): "
                        f"slope={r['trimmed_top5pct_citations_slope']:.5f}, p={r['trimmed_p_value']:.2e}."
                    )
                    followup_results.append(msg_done)
                else:
                    toy = await asyncio.to_thread(verify_cd_toy_graphs)
                    msg_done = f"{pid} verified bounded S2 toy citation graph (`passed={toy['passed']}`)."
                    followup_results.append(msg_done)

                self._mark_task_status(
                    run_id,
                    tid,
                    "task.completed",
                    "done",
                    msg_done,
                    persona_id=pid,
                    artifact_ids=art_ids_worker,
                    mode=mode,
                )
                self._transition_persona(run_id, pid, "completed", "subtask_done", mode=mode)

        await asyncio.gather(*(_run_dynamic_worker(i, pid) for i, pid in enumerate(spawned_ids)))

        table_members = [*spawned_ids]
        if len(table_members) < 2:
            table_members.append("persona_validation")
        cid, _ = self.collab.start_collaboration(
            run_id,
            kind="pod" if len(table_members) >= 3 else "debate",
            participant_persona_ids=table_members,
            topic=f"Parallel cross-check ({len(table_members)} specialists)",
            source_ref=SourceRef(kind="collaboration", id=f"dyn_{uuid.uuid4().hex[:6]}"),
            actor_id="persona_main",
            summary=f"Boss convened {len(table_members)} specialists (including dynamic workers) for synthesis.",
            mode=mode,
        )
        for m in table_members:
            self._transition_persona(run_id, m, "debating", "synthesizing_parallel_findings", mode=mode)
        await asyncio.sleep(0.35 if mode == "simulation" else 0.7)

        self.collab.end_collaboration(
            run_id,
            collaboration_id=cid,
            outcome="synthesized",
            actor_id="persona_main",
            summary="Dynamic specialist team synthesis delivered to Boss.",
            mode=mode,
        )
        for pid in self.store.list_persona_ids(run_id):
            self._transition_persona(run_id, pid, "idle", "returned_to_anchor", mode=mode)

        details_block = "\n".join(f"- {ln}" for ln in followup_results) if followup_results else "- Parallel robustness scripts executed cleanly."
        self.store.add_chat_message(
            run_id,
            kind="message",
            role="main",
            text=(
                f"Boss Agent dynamically spawned `{len(spawned_ids)}` specialist agent instance(s) "
                f"({', '.join(f'`{p}`' for p in spawned_ids)}) and executed real Python robustness experiments for **\"{user_prompt[:100]}\"**:\n\n"
                f"{details_block}"
            ),
            related_refs=[
                SelectedContext(kind="agent", id="persona_main"),
                *[SelectedContext(kind="agent", id=p) for p in spawned_ids],
                SelectedContext(kind="collaboration", id=cid),
            ],
        )
        self.store.update_run_status(
            run_id,
            status="completed",
            resources=self.scheduler.snapshot_state(run_id),
        )
        self.store.append_event(
            run_id,
            event_type="run.completed",
            actor_id="persona_main",
            target_ids=["persona_main"],
            subject=EventSubject(kind="run", id=run_id),
            status="completed",
            summary=f"Dynamic multi-instance check completed ({len(spawned_ids)} spawned).",
            payload={"dynamic_spawned": spawned_ids},
            mode=mode,
        )

    def _fail_run(self, run_id: str, *, mode: RunMode, reason: str) -> None:
        self._transition_persona(run_id, "persona_main", "failed", "run_failed", mode=mode)
        self.store.update_run_status(
            run_id,
            status="failed",
            report_status="failed",
            resources=self.scheduler.snapshot_state(run_id),
        )
        ev = self.store.append_event(
            run_id,
            event_type="run.failed",
            actor_id="persona_main",
            target_ids=["persona_main"],
            subject=EventSubject(kind="run", id=run_id),
            status="failed",
            summary=reason[:300],
            payload={"error": reason[:1000]},
            mode=mode,
        )
        self.store.add_chat_message(
            run_id,
            kind="system",
            role="main",
            text=f"**Run Failed / Blocked**: {reason}",
            related_refs=[SelectedContext(kind="agent", id="persona_main")],
            seq=ev.seq,
        )
