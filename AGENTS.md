# AGENTS.md — Libra Reproduce Backend (Standalone Project 2)

1. **Scientific Integrity**: Never claim S2/S3 reproduction when analyzing precomputed S1 metrics; never output mock/simulation results as `mode="live"`.
2. **Sole Write Gateway**: All user-triggered state mutations flow through `POST /api/runs/{run_id}/chat` (or `libra_reproduce.cli` invoking `MainAgentGateway`).
3. **Dual-Pool Resource Governance**: `ResourceScheduler` enforces `max_active_agent_tasks=4` and `max_parallel_experiments=2` independently; `ClaudeSDKClient` runs with `disallowed_tools` + `ToolPermissionGate` + `SafeExperimentRunner`.
4. **Independent Evidence Gate**: Agent agreement never marks a claim verified; `EvidenceGate` checks SHA-256 manifests, row conservation, script exit code, valid PNG headers, and S1/S2 honesty before `ReportBuilder` emits PDF/DOCX reports.
5. **Secret Safety**: Never log, persist in events/reports, or commit API keys.
