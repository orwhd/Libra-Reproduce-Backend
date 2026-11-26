# Libra Reproduce Backend · Implementation Status

**Updated**: 2026-11-27  
**Scope**: WHALES Lab Coding / Design Test — **Project 2 Pure-Backend Deliverable** (`Headless CLI` + `OpenAPI 3.1 V2 REST/SSE` + `Part II Option A Architecture Figure`).

---

## 1. Status Matrix (`DONE` / `PARTIAL`)

| Acceptance ID | Requirement | Status | Verification & Code Path |
| --- | --- | --- | --- |
| **STANDALONE** | Pure-backend headless CLI + standalone OpenAPI 3.1 V2 FastAPI+SSE server with zero frontend coupling | **DONE** | `libra_reproduce/main.py`, `libra_reproduce/cli.py`, `scripts/reproduce_nature2019.py`, `requirements.txt` |
| **API-01..API-05** | OpenAPI 3.1 V2 routes, `(run_id, request_id)` idempotency, transactional SQLite `EventStore` (`UNIQUE(run_id, seq)`), SSE `after=N` / `Last-Event-ID`, zero-execution `GET /snapshot?at_seq=N` | **DONE** | `libra_reproduce/api.py`, `libra_reproduce/storage/event_store.py`, `scripts/validate_contracts.py` (`27` refs), `tests/test_standalone_reproduce.py` |
| **UI-02 / BOSS** | Decoupled 6-persona catalog (`Boss`, `Paper`, `Data`, `Analysis`, `Validation`, `Support`) + dynamic multi-instance specialist scaling (`persona_{role}_{index}`) | **DONE** | `libra_reproduce/harness/persona_catalog.py`, `libra_reproduce/harness/coordinator.py` |
| **HAR-01..HAR-03** | Dual-pool `ResourceScheduler` (`<=4` active agent tasks, `<=2` parallel experiments, `<=20%` exploration budget), `ToolPermissionGate` (`can_use_tool` + path jail + redaction), AST/env-stripped `SafeExperimentRunner` | **DONE** | `libra_reproduce/harness/scheduler.py`, `libra_reproduce/security/tool_permissions.py`, `libra_reproduce/security/experiment_runner.py` |
| **SCI-01..SCI-04** | Empirical Wu, Wang & Evans (*Nature*, 2019) reproduction (`S1` on Harvard Dataverse `CC0` MAG cohort `N=60,120`, bounded `S2` $D$-index + window sensitivity, Round 2 counterfactual suite + live agent falsification scripts), independent `EvidenceGate`, readable `.pdf` & `.docx` reports | **DONE** | `libra_reproduce/science/*`, `outputs/nature2019_live_verified/*` |
| **DESIGN-A** | Pure-backend paper-style system architecture diagram (`PDF` + 300-DPI `PNG`) | **DONE** | `design/generate_project2_figure.py`, `design/project2_system_architecture.pdf`, `design/project2_system_architecture.png` |
| **SCOPE-PARTIAL** | Full proprietary Web of Science raw-graph `S2` recomputation & USPTO/GitHub cross-corpus replication | **PARTIAL (Documented)** | Primary cohort scoped honestly to `S1`; bounded OpenAlex + toy citation graphs cover `S2`; USPTO/GitHub & missing `Multi-Institution` column recorded in reports & `README.md` |

---

## 2. Latest Verified Live Run (`rep_2212a8f80c7f`)

Executed via headless CLI using live `claude-agent-sdk==0.2.189`:

```bash
uv run python -m libra_reproduce.cli run --paper-id paper_A --mode live --output-dir outputs/nature2019_live_verified
```

- **Status**: `completed` (`106` events, `16` verified artifacts, `EvidenceGate = PASSED`, `3`-page PDF & DOCX reports).
- **Primary S1 Cohort (`1954–2014`, $N = 45,335$)**:
  - Solo (`k=1`, `n=10,150`): mean $D = +0.014013$, percentile `54.33`, mean citations `23.98`.
  - Teams (`k>=11`, `n=449`): mean $D = +0.000364$, percentile `41.18`, mean citations `68.10`.
  - OLS slope: `-0.00187` / author ($p = 6.89 \times 10^{-40}$); preserved after 19-field demeaning.
- **Round 2 & Live Agent Falsification Suite**:
  - `Field × Decade` FWL Fixed Effects slope: `-0.00077` ($p = 1.48 \times 10^{-6}$).
  - Citation-quartile negative slopes: `Q1 -0.001363`, `Q2 -0.001182`, `Q3 -0.001445`, `Q4 -0.002782`.
  - Within-stratum permutation null (`n_perm=200`): `z = -6.369`, `p_perm = 0.00498`.
  - Leave-one-discipline-out jackknife: `7/7` negative (`[-0.002171, -0.001261]`).
  - Generative DAG simulation: `M1` slope `-0.065404` ($p = 6.98 \times 10^{-77}$) vs. `M2` control `-0.00074` ($p = 0.800$).
  - Live agent-authored scripts executed cleanly in sandbox: `schema_probe.py`, `functional_form_and_outlier_falsification.py`, `falsification_v2.py`.
