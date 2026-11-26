from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import validate

from libra_reproduce.config import HarnessResourceLimits, REPO_ROOT
from libra_reproduce.harness.scheduler import ResourceDeniedError, ResourceScheduler
from libra_reproduce.main import app as standalone_reproduce_app
from libra_reproduce.science.cd_index import compute_cd_index, verify_cd_toy_graphs
from libra_reproduce.science.evidence_gate import EvidenceGate
from libra_reproduce.security.experiment_runner import SafeExperimentRunner
from libra_reproduce.security.tool_permissions import (
    ToolPermissionDeniedError,
    ToolPermissionGate,
)
from libra_reproduce.storage.db import init_reproduce_db

CONTRACTS_DIR = REPO_ROOT / "contracts"


def _load_schema(name: str) -> dict:
    return json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_scheduler_dual_pools_and_release_on_failure() -> None:
    """HAR-01 & HAR-02: Agent task pool (4) and parallel experiment pool (2) are strictly independent."""
    init_reproduce_db()
    sched = ResourceScheduler(
        HarnessResourceLimits(
            max_active_agent_tasks=4,
            max_parallel_experiments=2,
            max_subagent_depth=2,
            max_debate_turns=6,
            exploration_budget_fraction=0.20,
            max_task_retries=2,
            experiment_timeout_seconds=30,
        )
    )

    leases = []
    for i in range(4):
        lease, outcome, _ = await sched.try_acquire_agent_task(
            run_id="run_sched", owner_id=f"ag_{i}", wait_if_full=False
        )
        assert lease is not None
        assert outcome == "granted"
        leases.append(lease)

    fifth, out5, _ = await sched.try_acquire_agent_task(
        run_id="run_sched", owner_id="ag_5", wait_if_full=False
    )
    assert fifth is None
    assert out5 == "queued"

    # Parallel experiment pool (2) operates independently of active agent tasks (4)
    exp1, eout1, _ = await sched.try_acquire_experiment(
        run_id="run_sched", owner_id="exp_1", wait_if_full=False
    )
    exp2, eout2, _ = await sched.try_acquire_experiment(
        run_id="run_sched", owner_id="exp_2", wait_if_full=False
    )
    exp3, eout3, _ = await sched.try_acquire_experiment(
        run_id="run_sched", owner_id="exp_3", wait_if_full=False
    )
    assert exp1 is not None and eout1 == "granted"
    assert exp2 is not None and eout2 == "granted"
    assert exp3 is None and eout3 == "queued"

    await sched.release_experiment(exp1.lease_id)
    await sched.release_experiment(exp2.lease_id)
    for ls in leases:
        await sched.release_agent_task(ls.lease_id)

    with pytest.raises(RuntimeError):
        async with sched.agent_task_slot(run_id="run_sched", owner_id="boom"):
            assert sched.snapshot_state("run_sched")["active_agent_tasks"] == 1
            raise RuntimeError("simulated agent crash")
    assert sched.snapshot_state("run_sched")["active_agent_tasks"] == 0

    with pytest.raises(ResourceDeniedError):
        async with sched.agent_task_slot(
            run_id="run_sched",
            owner_id="pod_over",
            is_exploration=True,
            budget_fraction=0.35,
        ):
            pass


def test_security_permissions_and_ast_sandbox(tmp_path: Path) -> None:
    """HAR-03: Built-in shell/write tools, path traversal, network imports, and os.environ are blocked."""
    gate = ToolPermissionGate(tmp_path / "run_box")
    with pytest.raises(ToolPermissionDeniedError):
        gate.authorize_tool_call("Bash", {"command": "cat ~/.ssh/id_rsa"})
    with pytest.raises(ToolPermissionDeniedError):
        gate.authorize_tool_call("inspect_paper_scope", {"file_path": "../../etc/passwd"})

    gate.authorize_tool_call(
        "inspect_paper_scope",
        {"paper_id": "paper_A", "api_key": "FAKE_SECRET_FOR_REDACTION_TEST"},
    )
    assert gate.audit_log[-1].sanitized_args["api_key"] == "[REDACTED]"

    runner = SafeExperimentRunner(tmp_path / "run_box", timeout_seconds=10)
    bad_env = runner.run_sync("bad_env.py", "import os\nprint(os.environ)\n")
    assert not bad_env.ok and bad_env.exit_code == -2

    bad_net = runner.run_sync("bad_net.py", "import urllib.request\n")
    assert not bad_net.ok and bad_net.exit_code == -2


def test_science_s2_cd_toy_graph_and_evidence_gate(tmp_path: Path) -> None:
    """SCI-02 & SCI-03: Toy citation graph hand-calculated checks + Evidence Gate consensus/overclaim rejection."""
    toy = verify_cd_toy_graphs()
    assert toy["passed"] is True

    years = {"r": 1990, "p": 2000, "c": 2003}
    assert compute_cd_index("p", years, [("p", "r"), ("c", "p")]).disruption == pytest.approx(1.0)
    assert compute_cd_index("p", years, [("p", "r"), ("c", "p"), ("c", "r")]).disruption == pytest.approx(-1.0)
    assert compute_cd_index("p", years, [("p", "r"), ("c", "r")]).disruption == pytest.approx(0.0)

    gate = EvidenceGate()
    consensus_res = gate.review_consensus_only("claim_1", ["persona_analysis", "persona_validation"])
    assert consensus_res.result == "failed"

    fake_csv = tmp_path / "s.csv"
    fake_csv.write_text("team_bin,sample_n,mean_disruption\n1,10,0.1\n", encoding="utf-8")
    overclaim = gate.review_reproduction_bundle(
        claim_id="claim_over",
        claimed_level="S2",
        data_manifest={
            "source_doi_or_url": "https://doi.org/10.7910/DVN/JPWNNK",
            "license": "CC0-1.0",
            "file_sha256": hashlib.sha256(fake_csv.read_bytes()).hexdigest(),
            "metric_origin": "precomputed_by_authors",
            "primary_1954_2014_row_count": 10,
        },
        sample_csv_path=fake_csv,
        team_csv_path=fake_csv,
        window_csv_path=fake_csv,
        fig_main_path=None,
        fig_window_path=None,
        experiment_summary={"primary_sample_n": 10, "solo_mean_disruption": 0.1},
        script_sha256="a" * 64,
        exit_code=0,
    )
    assert overclaim.result == "failed"
    assert overclaim.checks["reproduction_level_honest"] is False


@pytest.mark.asyncio
async def test_standalone_openapi_v2_simulation_idempotency_and_sse_resume() -> None:
    """API-01..API-05: Standalone ASGI OpenAPI V2 contract, idempotency, SSE resume, replay, and reports."""
    init_reproduce_db()
    event_schema = _load_schema("event.schema.json")
    collab_schema = _load_schema("collaboration-payload.schema.json")
    canvas_schema = _load_schema("canvas-projection.schema.json")

    transport = httpx.ASGITransport(app=standalone_reproduce_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_health = await client.get("/api/health")
        assert r_health.status_code == 200

        r_create = await client.post(
            "/api/runs?mode=simulation",
            json={"title": "Standalone Nature 2019 Test", "paper_id": "paper_A"},
        )
        assert r_create.status_code == 201
        run_id = r_create.json()["run_id"]

        snap0 = (await client.get(f"/api/runs/{run_id}/snapshot")).json()
        assert snap0["at_seq"] == 0
        validate(snap0["canvas"], canvas_schema)
        assert len(snap0["canvas"]["personas"]) == 6
        assert all(p["runtime_agent_id"] is None and p["state"] == "idle" for p in snap0["canvas"]["personas"])

        chat_payload = {
            "request_id": "req-standalone-001",
            "text": "How does team size influence disruptive innovation vs developmental research?",
            "selected_context": None,
            "attachment_ids": [],
        }
        c1 = await client.post(f"/api/runs/{run_id}/chat", json=chat_payload)
        c2 = await client.post(f"/api/runs/{run_id}/chat", json=chat_payload)
        assert c1.status_code == 202 and c2.status_code == 202
        assert c1.json()["command_id"] == c2.json()["command_id"]

        for _ in range(180):
            cur = (await client.get(f"/api/runs/{run_id}")).json()
            if cur["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.15)
        assert cur["status"] == "completed"

        sse_full = await client.get(f"/api/runs/{run_id}/events")
        events_all = [
            json.loads(line[len("data:") :].strip())
            for line in sse_full.text.splitlines()
            if line.startswith("data:")
        ]
        assert len(events_all) >= 12
        for ev in events_all:
            validate(ev, event_schema)
            if ev["type"].startswith("collaboration."):
                validate(ev, collab_schema)

        sse_resumed = await client.get(f"/api/runs/{run_id}/events?after=5")
        resumed_events = [
            json.loads(line[len("data:") :].strip())
            for line in sse_resumed.text.splitlines()
            if line.startswith("data:")
        ]
        assert [e["seq"] for e in resumed_events] == list(range(6, len(events_all) + 1))

        rep_status = (await client.get(f"/api/runs/{run_id}/report")).json()
        assert rep_status["status"] == "ready"
        assert len(rep_status["artifact_ids"]) == 2

        seq_before_read = (await client.get(f"/api/runs/{run_id}/snapshot")).json()["at_seq"]
        inq = await client.post(
            f"/api/runs/{run_id}/chat",
            json={
                "request_id": "req-standalone-readonly-002",
                "text": "Explain the citation window sensitivity conclusion",
                "selected_context": {"kind": "agent", "id": "persona_validation"},
            },
        )
        assert inq.status_code == 200
        assert inq.json()["status"] == "answered"
        seq_after_read = (await client.get(f"/api/runs/{run_id}/snapshot")).json()["at_seq"]
        assert seq_after_read == seq_before_read
