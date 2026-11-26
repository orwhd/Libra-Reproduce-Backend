# Libra Reproduce Backend

Backend multi-agent LLM system for scientific paper reproduction.

## Design

![Project 2 System Architecture](design/project2_system_architecture.png)

- Vector formats: [PDF](design/project2_system_architecture.pdf) · [SVG](design/project2_system_architecture.svg)

## Target Paper & Dataset

- **Paper**: Lingfei Wu, Dashun Wang & James A. Evans (2019), *"Large teams develop and small teams disrupt science and technology"*, *Nature*, 566, 378–382 ([article](https://www.nature.com/articles/s41586-019-0941-9)).
- **Example Prompt**: *"How does team size influence the balance between disruptive innovation and developmental research in science and technology?"*
- **Dataset**: A documented sample from the authors' public Harvard Dataverse release (`CC0-1.0`) stored under `data/samples/`, together with a bounded [OpenAlex](https://openalex.org/) citation-neighborhood subset for direct disruption-index recomputation. See [`data/README.md`](data/README.md) for data provenance.

## Implementation Overview

- **Multi-agent workflow**: A coordinating agent (Boss) receives the research prompt and delegates work across specialized agents: Paper (extracting claims and methodology), Data (loading and auditing datasets), Analysis (writing and executing statistical and robustness scripts), Validation (reviewing claims, running follow-up checks, and verifying evidence), and Support (compiling the final report).
- **Harness & runtime**: Built on the Claude Agent SDK with an OpenAI-compatible fallback. The harness enforces tool permissions, bounded task concurrency, and SQLite event logging for reproducible run inspection.
- **Sandboxed execution**: Agent-generated Python experiment scripts run in isolated subprocesses with static safety checks, stripped credentials, and timeouts.
- **Verification & reporting**: A validation step checks artifact integrity, sample consistency, and alignment between claims and data provenance before generating PDF and Word (`.docx`) reports with embedded figures and tables.

## Reproduced Outputs

A completed reproduction bundle is included in [`outputs/nature2019_live_verified/`](outputs/nature2019_live_verified/), covering the core team-size vs. disruption analysis, citation-window sensitivity, and follow-up robustness checks:

- **Reports**: [`reproduction_report.pdf`](outputs/nature2019_live_verified/reproduction_report.pdf), [`reproduction_report.docx`](outputs/nature2019_live_verified/reproduction_report.docx)
- **Figures**: [`figure_team_size_disruption.png`](outputs/nature2019_live_verified/figure_team_size_disruption.png), [`figure_window_sensitivity.png`](outputs/nature2019_live_verified/figure_window_sensitivity.png), [`figure_multiround_experiments.png`](outputs/nature2019_live_verified/figure_multiround_experiments.png)
- **Scripts**: [`reproduce_nature2019_experiment.py`](outputs/nature2019_live_verified/reproduce_nature2019_experiment.py), [`exp_round2_counterfactual_simulation.py`](outputs/nature2019_live_verified/exp_round2_counterfactual_simulation.py), plus additional agent-generated check scripts in the same directory
- **Intermediate data & manifests**: Intermediate CSV datasets, data/artifact manifests, and run trace (`events.jsonl`) in [`outputs/nature2019_live_verified/`](outputs/nature2019_live_verified/)

## Setup & Reproduction Instructions

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Install Dependencies

```bash
git clone https://github.com/orwhd/Libra-Reproduce-Backend.git
cd Libra-Reproduce-Backend
uv sync
```

Or with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Credentials

```bash
cp .env.example .env
```

Configure one LLM provider in `.env`:
- **Claude Agent SDK**: authenticate `claude` CLI (`claude auth status`) or set `ANTHROPIC_API_KEY`.
- **OpenAI-compatible API**: set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.

### Run Full Multi-Agent Reproduction (CLI)

```bash
uv run python scripts/reproduce_nature2019.py
```

### Re-Run Saved Experiment Scripts Directly

```bash
mkdir -p /tmp/recheck_s1 /tmp/recheck_r2
uv run python outputs/nature2019_live_verified/reproduce_nature2019_experiment.py \
  outputs/nature2019_live_verified/wu2019_mag_sample.csv /tmp/recheck_s1
uv run python outputs/nature2019_live_verified/exp_round2_counterfactual_simulation.py \
  outputs/nature2019_live_verified/wu2019_mag_sample.csv /tmp/recheck_r2
```

### Run as an HTTP API Server (Optional)

```bash
uv run uvicorn libra_reproduce.main:app --host 127.0.0.1 --port 8001
```

## Status & Technical Decisions

### What Works
- End-to-end backend multi-agent reproduction pipeline (CLI and HTTP API) covering claim extraction, dataset auditing, sandboxed Python experimentation, validation checks, and PDF/DOCX report generation.
- Reproducible artifact bundle for Wu, Wang & Evans (2019) with reports, figures, analysis scripts, intermediate CSV data, and manifests.

### What Is Incomplete
- The main empirical analysis uses a stratified sample of the authors' precomputed MAG dataset rather than recomputing disruption scores over the full proprietary Web of Science citation graph; direct graph-level recomputation is demonstrated on a bounded OpenAlex subset.
- Secondary corpora in the original paper (USPTO patents and GitHub repositories) are not reproduced.
- Multi-institution subgroup analysis is omitted because the `Multi-Institution` column referenced in the Dataverse description is absent from the distributed data file.

### Main Technical Decisions
- **Provenance-checked reporting**: The validation step distinguishes between analyses over author-precomputed scores and direct citation-graph recomputation so reported conclusions stay within the verified scope of the data.
- **Isolated script execution**: Generated analysis scripts run in credential-stripped subprocesses with static safety checks to prevent side effects and secret leakage.
- **Event-logged backend architecture**: Recording runs, events, and artifacts in SQLite keeps the reproduction harness decoupled from any frontend and supports auditable output bundles.
