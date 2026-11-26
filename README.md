# Libra Reproduce Backend — Standalone Multi-Agent Scientific Paper Reproduction Harness

**WHALES Lab Coding / Design Test — Project 2 Pure-Backend Deliverable (OpenAPI 3.1 V2 + Headless CLI)**

`libra-reproduce-backend` is a standalone, **harness-engineered multi-agent backend** for rigorous empirical scientific paper reproduction. It implements the complete **OpenAPI 3.1 V2 HTTP + resumable SSE contract** (`contracts/openapi.yaml`) so it can operate **either** as a pure headless Python CLI / REST+SSE microservice **or** as the backend engine for the Libra Research web frontend (`Research Chat` + `Living Research Canvas`).

---

## Architecture Figure & Documentation

- **System Architecture Figure (Design Test Option A)**: [design/project2_system_architecture.pdf](design/project2_system_architecture.pdf) ([PNG preview](design/project2_system_architecture.png))
- **Implementation Plan & Status**: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) · [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) · [docs/decisions/ADR-V2-01-to-07.md](docs/decisions/ADR-V2-01-to-07.md)
- **Machine-Readable Contracts**: [contracts/openapi.yaml](contracts/openapi.yaml) · [contracts/event.schema.json](contracts/event.schema.json) · [contracts/canvas-projection.schema.json](contracts/canvas-projection.schema.json) · [contracts/collaboration-payload.schema.json](contracts/collaboration-payload.schema.json)

---

## 1. Multi-Agent Harness Architecture

1. **`MainAgentGateway` (`libra_reproduce/harness/main_gateway.py`)**: Sole user command entrypoint (`POST /api/runs/{run_id}/chat`) enforcing `request_id` idempotency, distinguishing zero-cost read-only inquiries (`200 OK`, `status: "answered"`) from orchestrated reproduction runs (`202 Accepted`), and synthesizing verified findings.
2. **Decoupled 6-Persona Roster (`libra_reproduce/harness/persona_catalog.py`)**: Separates stable logical personas (`persona_main`, `persona_paper`, `persona_data`, `persona_analysis`, `persona_validation`, `persona_support`) from on-demand SDK runtime execution sessions (`runtime_agent_id`).
3. **Controlled `SDKAdapter` (`libra_reproduce/harness/sdk_adapter.py`)**:
   - Primary live driver: **`claude-agent-sdk==0.2.189`** (`ClaudeSDKClient`) exposing restricted domain MCP tools (`create_sdk_mcp_server("science", ...)`), blocking unrestricted built-in `Bash`/`Write`/`Edit`/`Read`/`Task`/`Agent` tools, and recording secret-redacted `tool.executed` traces.
   - Portable live fallback: OpenAI-compatible tool-calling harness (`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`) governed by the exact same `ToolPermissionGate` and `ResourceScheduler`.
   - Deterministic CI fixture mode: `mode="simulation"` (explicitly tagged `simulation` on every event and snapshot; never labeled `live`).
4. **Dual-Pool `ResourceScheduler` (`libra_reproduce/harness/scheduler.py`)**: Independently enforces `max_active_agent_tasks = 4`, `max_parallel_experiments = 2`, and `exploration_budget_fraction <= 0.20`, with guaranteed lease release on exception or cancellation.
5. **Isolated `SafeExperimentRunner` (`libra_reproduce/security/experiment_runner.py`)**: Validates Python scripts via AST policy checks (blocking network, process spawning, and `os.environ` access) and executes statistical reproduction scripts inside `python -I` subprocesses with stripped environments (`0` API keys) and wall-clock timeouts.
6. **Methodological Debate -> Sensitivity Pod -> `EvidenceGate` (`libra_reproduce/harness/collaboration_manager.py`, `libra_reproduce/science/evidence_gate.py`)**:
   - `Validation` challenges forward citation-window truncation and `S1` vs. `S2` claim boundaries (`collaboration.started`), expanding with `Data` and `Analysis` into a 3-agent sensitivity Pod (`collaboration.members_changed`) before `EvidenceGate` audits dataset SHA-256 manifests, row conservation, figure-vs-table numeric agreement, and S2 toy-graph invariants (`passed | failed | inconclusive`).
7. **Transactional SQLite `EventStore` & `ReportBuilder` (`libra_reproduce/storage/event_store.py`, `libra_reproduce/science/report_builder.py`)**:
   - Persists monotonic `(run_id, seq)` `HarnessEvent` streams supporting `Last-Event-ID` / `?after=N` SSE resume and zero-execution historical snapshots (`GET /api/runs/{run_id}/snapshot?at_seq=N`).
   - Generates publication-ready **PDF (`reproduction_report.pdf`)** and **Word (`reproduction_report.docx`)** reports embedding real figures, statistical tables, Research Gains (`Verified`, `Ruled Out`, `Open Questions`, `Method Notes`), and reproduction instructions.

---

## 2. Empirical Benchmark: Wu, Wang & Evans (*Nature*, 2019)

- **Target Paper**: Lingfei Wu, Dashun Wang & James A. Evans (2019), *"Large teams develop and small teams disrupt science and technology"*, *Nature*, 566, 378–382 (`https://doi.org/10.1038/s41586-019-0941-9`).
- **Empirical Dataset (`data/samples/`)**:
  - Stratified sample (`N = 60,120` rows; `45,335` in the `1954–2014` primary cohort) drawn via HTTP `206 Range` blocks from the authors' Harvard Dataverse release (`https://doi.org/10.7910/DVN/JPWNNK`, File `6925921` `Aggregated_20210521.txt`, `CC0-1.0` Public Domain), plus bounded OpenAlex `S2` citation subgraphs (`wu2019_s2_openalex_bounded.csv`).
  - Pre-verified live output bundle included in [`outputs/nature2019_live_verified/`](outputs/nature2019_live_verified/).

---

## 3. Quick Start (Pure Backend)

### 1. Install with `uv`

```bash
uv sync
```

### 2. Configure Credentials (Optional for Simulation; Required for Live Mode)

```bash
cp .env.example .env
# Option A: Log into Claude Code CLI (`claude auth status`) or set ANTHROPIC_API_KEY in .env
# Option B: Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL for OpenAI-compatible live tool execution
```

### 3. One-Command Headless Reproduction (CLI)

```bash
# Live multi-agent reproduction (writes events.jsonl, CSVs, PNG figures, PDF & DOCX reports)
uv run python scripts/reproduce_nature2019.py

# Or via full CLI flags:
uv run python -m libra_reproduce.cli run \
  --paper-id paper_A \
  --mode live \
  --output-dir outputs/latest_run

# Deterministic offline simulation mode (no API key required):
uv run python scripts/reproduce_nature2019.py --simulation
```

### 4. Start Standalone OpenAPI V2 HTTP + SSE Server

```bash
uv run uvicorn libra_reproduce.main:app --host 127.0.0.1 --port 8001
```

Example pure-HTTP workflow against the running server:

```bash
# 1. Create a reproduction run (returns 6 idle personas, 0 paid calls)
RUN_ID=$(curl -s -X POST "http://127.0.0.1:8001/api/runs?mode=live" \
  -H "Content-Type: application/json" \
  -d '{"title":"Nature 2019 Reproduction","paper_id":"paper_A"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")

# 2. Submit research command to Main Agent
curl -s -X POST "http://127.0.0.1:8001/api/runs/$RUN_ID/chat" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"req_001","text":"Reproduce how team size affects disruptive innovation vs developmental citation impact."}'

# 3. Stream ordered HarnessEvents via SSE (supports ?after=N or Last-Event-ID)
curl -N "http://127.0.0.1:8001/api/runs/$RUN_ID/events?after=0"

# 4. Inspect read-only snapshot & download generated PDF/DOCX reports
curl -s "http://127.0.0.1:8001/api/runs/$RUN_ID/snapshot" | head -c 600
curl -s "http://127.0.0.1:8001/api/runs/$RUN_ID/report"
```

---

## 4. Automated Contract & Pytest Verification

```bash
# Validate OpenAPI 3.1 + JSON Schema 2020-12 contracts and examples
uv run python scripts/validate_contracts.py

# Run standalone unit, sandbox security, S2 CD-index, Evidence Gate & ASGI integration tests
uv run pytest -v
```
