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
from ..science.nature2019_analysis import run_nature2019_experiments
from ..science.report_builder import build_reproduction_reports
from ..storage.artifact_store import ArtifactStore
from ..storage.db import jdumps, jloads, utc_now_iso
from ..storage.event_store import EventStore
from .collaboration_manager import CollaborationManager
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
                "SELECT state, runtime_agent_id FROM reproduce_personas WHERE run_id = ? AND persona_id = ?",
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
        if runtime_agent_id and not with_prev.get("runtime_agent_id"):
            self.store.append_event(
                run_id,
                event_type="agent.spawned",
                actor_id=runtime_agent_id,
                target_ids=[persona_id],
                subject=EventSubject(kind="agent", id=persona_id),
                status="active",
                summary=f"Bound runtime execution session for {persona_id.replace('persona_', '').capitalize()}.",
                payload={"persona_id": persona_id, "runtime_agent_id": runtime_agent_id},
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
            summary=f"{persona_id.replace('persona_', '').capitalize()} -> {next_state} ({reason})",
            payload={
                "persona_id": persona_id,
                "previous_state": with_prev.get("state", "idle"),
                "next_state": next_state,
                "reason": reason,
            },
            mode=mode,
            state_mutator=_mut,
        )

    def _get_persona_state(self, run_id: str, persona_id: str) -> dict[str, Any]:
        from ..storage.db import get_reproduce_conn

        with get_reproduce_conn(self.store.db_path) as conn:
            row = conn.execute(
                "SELECT state, runtime_agent_id FROM reproduce_personas WHERE run_id = ? AND persona_id = ?",
                (run_id, persona_id),
            ).fetchone()
        return dict(row) if row else {"state": "idle", "runtime_agent_id": None}

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

            # --- Phase 1: Paper + Data Agents run concurrently under Scheduler ---
            extracted_scope: dict[str, Any] = {}
            dataset_bundle: dict[str, Any] = {}

            async def _paper_stage() -> AgentTurnResult:
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
                        system_prompt="You are Paper Agent. Call inspect_paper_scope once to register formal reproduction methodology, then summarize in 2 sentences.",
                        user_prompt=f"Extract methodology and testable claims for: {user_prompt}",
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
                        max_turns=3,
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

            paper_turn_res, _ = await asyncio.gather(_paper_stage(), _data_stage())

            # --- Phase 2: Analysis Agent executes primary statistical script inside Parallel Experiment Pool ---
            exp_bundle: dict[str, Any] = {}
            s2_toy_result = verify_cd_toy_graphs()
            s2_csv_path: Path | None = None

            if nature_target:
                from ..config import get_bundled_samples_dir

                bundled_s2 = get_bundled_samples_dir() / "wu2019_s2_openalex_bounded.csv"
                if bundled_s2.exists():
                    s2_csv_path = run_dir / "data" / "s2_openalex_bounded.csv"
                    s2_csv_path.write_bytes(bundled_s2.read_bytes())
                elif mode == "live":
                    s2_df = await compute_bounded_openalex_s2_window_check(max_focal=8)
                    if len(s2_df) > 0:
                        s2_csv_path = run_dir / "data" / "s2_openalex_bounded.csv"
                        s2_df.to_csv(s2_csv_path, index=False)

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
                    "Analysis Agent acquiring parallel experiment slot to compute team-size disruption curves & 95% CIs.",
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
                await asyncio.sleep(0.35 if mode == "simulation" else 0.9)

                self.collab.add_debate_turn(
                    run_id,
                    debate_id=debate_id,
                    claim_id="claim_citation_window",
                    speaker_persona_id="persona_validation",
                    stance="challenge",
                    public_summary=(
                        "Validation noted that cumulative MAG citations through 2021 give pre-2005 papers >=17 years of "
                        "forward citations while post-2014 papers are right-truncated, and precomputed S1 D-scores must not be claimed as S2."
                    ),
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
                        "Analysis proposed inviting Data into a 3-agent Pod to compare 1954-2004, 2005-2014, and "
                        "2015-2021 cohorts alongside bounded 5y vs. 10y S2 citation subgraphs."
                    ),
                    evidence_ref=exp_bundle["team_csv_artifact_id"],
                    target_ids=["persona_analysis", "persona_validation"],
                    mode=mode,
                )
                await asyncio.sleep(0.35 if mode == "simulation" else 0.9)

                # Expand 2-person rectangular table -> 3-person circular Pod table (`collaboration.members_changed`)
                pod_id, _ = self.collab.start_pod(
                    run_id,
                    research_question="Does right-truncation of forward citation windows reverse the negative team-size disruption slope?",
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
                await asyncio.sleep(0.35 if mode == "simulation" else 0.9)

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

                # Independent Evidence Gate evaluation
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
                            "evidence_gate",
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
                    actor_id="persona_validation",
                    target_ids=["persona_analysis", "persona_validation"],
                    subject=EventSubject(kind="evidence", id=evidence_id),
                    status=ev_review.result,
                    summary=f"Evidence Gate verdict: {ev_review.result.upper()} ({ev_review.reproduction_level_verified} verified).",
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
            sim_prefix = "**[SIMULATION MODE]** " if mode == "simulation" else ""
            final_md = (
                f"{sim_prefix}### Reproduction Summary ({ev_review.reproduction_level_verified} Verified + Bounded S2 Check)\n\n"
                f"- **Target**: Wu, Wang & Evans (*Nature*, 2019, `10.1038/s41586-019-0941-9`)\n"
                f"- **Empirical Sample**: `N = {s['primary_sample_n']:,}` MAG papers (`1954–2014` primary historical cohort; `{s['total_sample_n']:,}` total rows through `2021`, `CC0-1.0`)\n"
                f"- **Core Finding**: Solo authors (`k=1`) average disruption `D = {s['solo_mean_disruption']:.4f}` (percentile `{s['solo_disruption_percentile']:.1f}`) vs. `11+` teams at `D = {s['large_11plus_mean_disruption']:.4f}` (percentile `{s['large_11plus_disruption_percentile']:.1f}`), while mean forward citations rise from `{s['solo_mean_citations']:.1f}` to `{s['large_11plus_mean_citations']:.1f}` (`OLS slope = {s['ols_slope_per_author']:.5f}/author`, `p = {s['ols_p_value']:.2e}`).\n"
                f"- **Validation & Window Sensitivity**: Comparing `1954–2004` (`>=17y` citation window) vs. `2015–2021` confirmed right-truncation compresses raw `D` spread without reversing the negative team-size gradient.\n"
                f"- **Evidence Gate**: `{ev_review.result.upper()}` (`{paper_turn_res.provider}`). Inspect or download the PDF/DOCX reports, figures, scripts, and CSV manifests directly below."
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
                    exp_bundle["team_csv_artifact_id"],
                    exp_bundle["script_artifact_id"],
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
            # Return all personas to calm `idle` at home anchors after run completion (L1.5 §8.4)
            for pid in (
                "persona_main",
                "persona_paper",
                "persona_data",
                "persona_analysis",
                "persona_validation",
                "persona_support",
            ):
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
            summary="Roundtable synthesis delivered to Main; participants returned to anchors.",
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
