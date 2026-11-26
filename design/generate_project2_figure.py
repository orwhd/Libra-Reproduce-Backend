#!/usr/bin/env python3
"""
Generate WHALES Lab Part II Design Test (Option A) Paper-Style System & Workflow Figure
for the Pure-Backend Project 2 Deliverable:
  - design/project2_system_architecture.pdf (vector)
  - design/project2_system_architecture.png (300 DPI)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "#ffffff",
    ec: str = "#334155",
    title_color: str = "#0f172a",
    sub_color: str = "#475569",
    lw: float = 1.35,
    title_size: float = 9.0,
    sub_size: float = 7.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.003,rounding_size=0.010",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.74,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=4,
            family="DejaVu Sans",
        )
        ax.text(
            x + w / 2,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=sub_color,
            linespacing=1.28,
            zorder=4,
            family="DejaVu Sans",
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=4,
            family="DejaVu Sans",
        )


def draw_region(ax, x: float, y: float, w: float, h: float, label: str, fc: str, ec: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.15,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.010,
        y + h - 0.012,
        label,
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=ec,
        zorder=2,
        family="DejaVu Sans",
    )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = "#334155",
    ls: str = "-",
    lx: float | None = None,
    ly: float | None = None,
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        color=color,
        linewidth=1.35,
        linestyle=ls,
        zorder=5,
    )
    ax.add_patch(arr)
    if label:
        mx = lx if lx is not None else (start[0] + end[0]) / 2
        my = ly if ly is not None else (start[1] + end[1]) / 2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.16", "fc": "#ffffff", "ec": "none", "alpha": 0.94},
            zorder=6,
            family="DejaVu Sans",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.5, 9.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(
        0.5,
        0.968,
        "Libra Reproduce Backend: Harness-Governed Multi-Agent Scientific Discovery",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#0f172a",
        family="DejaVu Sans",
    )
    ax.text(
        0.5,
        0.942,
        "WHALES Lab · Project 2 Architecture (Option A) — Pure-Backend CLI/REST+SSE, Dual-Pool Scheduler, Sandboxed Multi-Round Science & Evidence Gate",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#475569",
        family="DejaVu Sans",
    )

    # Region A: Pure-Backend Entrypoints & Open Datasets
    draw_region(ax, 0.012, 0.085, 0.214, 0.825, "A. Backend Entrypoints & Data", "#f8fafc", "#2563eb")
    draw_box(
        ax, 0.025, 0.720, 0.188, 0.145,
        "Headless CLI Runner",
        "scripts/reproduce_nature2019.py\npython -m libra_reproduce.cli run\n• --mode live | simulation\n• Exports PDF, DOCX, CSV, PNG",
        fc="#eff6ff", ec="#2563eb", title_color="#1d4ed8",
    )
    draw_box(
        ax, 0.025, 0.525, 0.188, 0.150,
        "OpenAPI 3.1 REST + SSE",
        "libra_reproduce.main:app\n• POST /api/runs/{id}/chat\n• (run_id, request_id) idempotent\n• SSE ?after=N / Last-Event-ID",
        fc="#ffffff", ec="#1d4ed8", title_color="#1e40af",
    )
    draw_box(
        ax, 0.025, 0.320, 0.188, 0.160,
        "Open Empirical Cohort (S1)",
        "Harvard Dataverse CC0-1.0\n(Wu, Wang & Evans, Nature 2019)\n• N = 60,120 stratified MAG rows\n• 1954–2014 + 2015–2021 cohorts",
        fc="#ecfdf5", ec="#059669", title_color="#047857",
    )
    draw_box(
        ax, 0.025, 0.115, 0.188, 0.160,
        "Bounded Graph & Schemas",
        "• OpenAlex S2 subgraph (5y/10y)\n• SHA-256 data_manifest.json\n• Draft 2020-12 JSON Schemas\n• Read-only ?at_seq=N snapshots",
        fc="#f5f3ff", ec="#7c3aed", title_color="#6d28d9",
    )

    # Region B: Multi-Agent Governance Harness & Sandboxed Execution
    draw_region(ax, 0.254, 0.315, 0.486, 0.595, "B. Multi-Agent Governance Harness & Isolated Execution", "#fbfdff", "#0f172a")
    draw_box(
        ax, 0.270, 0.740, 0.212, 0.125,
        "BossAgentGateway & Coordinator",
        "Classifies 200 answered vs.\n202 accepted; dynamically scales\nBoss / Paper / Data / Analysis / Val",
        fc="#fffbeb", ec="#d97706", title_color="#b45309",
    )
    # 6 persona role dots clear of the vertical Dispatch arrow (x=0.376)
    for idx_c, col in enumerate(["#FFD24A", "#FF9FD8", "#7CC8FF", "#4FF5C8", "#FF9A6B", "#B9A6FF"]):
        circ = Circle((0.402 + idx_c * 0.017, 0.712), 0.006, facecolor=col, edgecolor="#1b1b1f", linewidth=0.95, zorder=5)
        ax.add_patch(circ)
    draw_box(
        ax, 0.514, 0.740, 0.212, 0.125,
        "Dual-Pool ResourceScheduler",
        "Pool 1: Active Agent Tasks (<=4)\nPool 2: Parallel Experiments (<=2)\nExploration Pod budget (<=20%)",
        fc="#fef2f2", ec="#dc2626", title_color="#b91c1c",
    )
    draw_box(
        ax, 0.270, 0.545, 0.212, 0.135,
        "Controlled SDKAdapter",
        "ClaudeSDKClient (0.2.189) MCP\n+ portable OpenAI tool harness\nToolPermissionGate blocks Bash/Edit",
        fc="#eff6ff", ec="#1d4ed8", title_color="#1e40af",
    )
    draw_box(
        ax, 0.514, 0.545, 0.212, 0.135,
        "SafeExperimentRunner (-I / -E)",
        "AST static check (no network/env)\nStripped env (0 API keys) + POSIX\nrlimits & wall-clock timeout",
        fc="#1e293b", ec="#0f172a", title_color="#f8fafc", sub_color="#cbd5e1",
    )
    draw_box(
        ax, 0.270, 0.340, 0.212, 0.145,
        "Empirical Science (Round 1)",
        "• S1 team size k=1 vs k>=11 & OLS\n• Within-19-discipline demeaning\n• S2 D=(ni-nj)/(ni+nj+nk) checks",
        fc="#ecfdf5", ec="#059669", title_color="#047857",
    )
    draw_box(
        ax, 0.514, 0.340, 0.212, 0.145,
        "Debate -> Pod (Round 2)",
        "• FWL Field x Decade Fixed Effects\n• Citation-quartile & Jackknife tests\n• Permutation Null (n=200) & DAG sim",
        fc="#fff7ed", ec="#ea580c", title_color="#c2410c",
    )

    # Region C: Append-Only EventStore
    draw_region(ax, 0.254, 0.085, 0.486, 0.195, "C. Transactional SQLite EventStore & Provenance Ledger", "#f8fafc", "#475569")
    draw_box(
        ax, 0.270, 0.115, 0.214, 0.115,
        "Append-Only Event Ledger",
        "Single-tx state + HarnessEvent\nUNIQUE(run_id, seq); secret-redacted\ntool.executed audit traces",
        fc="#ffffff", ec="#334155",
    )
    draw_box(
        ax, 0.510, 0.115, 0.216, 0.115,
        "Scoped ArtifactStore",
        "Run-jailed SHA-256 registry;\nsafe download URLs; zero host path\nleakage across API responses",
        fc="#ffffff", ec="#059669",
    )

    # Region D: Independent Verification & Scientific Deliverables
    draw_region(ax, 0.768, 0.085, 0.220, 0.825, "D. Verification & Deliverables", "#fbfaf6", "#7c3aed")
    draw_box(
        ax, 0.782, 0.665, 0.192, 0.195,
        "Independent EvidenceGate",
        "• Rejects consensus-only claims\n• Audits SHA-256 dataset hashes\n• Verifies N conservation & PNGs\n• Enforces honest S1 vs S2 labels",
        fc="#f5f3ff", ec="#7c3aed", title_color="#6d28d9",
    )
    draw_box(
        ax, 0.782, 0.395, 0.192, 0.225,
        "ReportBuilder (Support Agent)",
        "Synthesizes publication artifacts:\n• reproduction_report.pdf\n• reproduction_report.docx\n• 4 Research Gain buckets\n  (Verified / Ruled Out / Open / Note)",
        fc="#ecfdf5", ec="#059669", title_color="#047857",
    )
    draw_box(
        ax, 0.782, 0.115, 0.192, 0.230,
        "Reproducible Bundle",
        "outputs/nature2019_live_verified/\n• Python experiment scripts\n• Intermediate CSVs & manifests\n• 300-DPI PNG figures\n• events.jsonl execution trace",
        fc="#ffffff", ec="#334155",
    )

    # Flow Arrows
    draw_arrow(ax, (0.213, 0.795), (0.270, 0.795), "Prompt / Run", color="#2563eb", ly=0.818)
    draw_arrow(ax, (0.213, 0.600), (0.270, 0.765), "ChatCmd", color="#1d4ed8", lx=0.240, ly=0.695)
    draw_arrow(ax, (0.213, 0.405), (0.270, 0.405), "CC0 MAG / S2", color="#059669", ly=0.427)

    draw_arrow(ax, (0.482, 0.802), (0.514, 0.802), "Lease", color="#dc2626", ly=0.824)
    draw_arrow(ax, (0.376, 0.740), (0.376, 0.680), "Dispatch", color="#0f172a", lx=0.340)
    draw_arrow(ax, (0.482, 0.612), (0.514, 0.612), "MCP Tool", color="#1d4ed8", ly=0.633)
    draw_arrow(ax, (0.376, 0.545), (0.376, 0.485), "Round 1", color="#059669", lx=0.340)
    draw_arrow(ax, (0.620, 0.545), (0.620, 0.485), "Round 2 Pod", color="#c2410c", lx=0.664)
    draw_arrow(ax, (0.482, 0.412), (0.514, 0.412), "Challenge", color="#ea580c", ly=0.433)

    draw_arrow(ax, (0.376, 0.340), (0.376, 0.230), "Commit Events", color="#334155", lx=0.326)
    draw_arrow(ax, (0.620, 0.340), (0.620, 0.230), "Artifacts + SHA", color="#059669", lx=0.672)

    draw_arrow(ax, (0.726, 0.412), (0.782, 0.740), "Audit Gate", color="#7c3aed", lx=0.752, ly=0.585)
    draw_arrow(ax, (0.878, 0.665), (0.878, 0.620), "PASSED", color="#059669", lx=0.910)
    draw_arrow(ax, (0.878, 0.395), (0.878, 0.345), "Export", color="#334155", lx=0.905)
    draw_arrow(ax, (0.726, 0.172), (0.782, 0.172), "Ledger", color="#475569", ly=0.194)

    # Bottom orthogonal SSE / Snapshot feedback path
    pts_nav = [(0.782, 0.135), (0.750, 0.135), (0.750, 0.045), (0.119, 0.045), (0.119, 0.115)]
    ax.plot([p[0] for p in pts_nav[:-1]], [p[1] for p in pts_nav[:-1]], color="#64748b", lw=1.3, ls="--", zorder=5)
    ax.add_patch(
        FancyArrowPatch(
            pts_nav[-2],
            pts_nav[-1],
            arrowstyle="-|>,head_length=5.5,head_width=3.2",
            color="#64748b",
            linewidth=1.3,
            linestyle="--",
            zorder=5,
        )
    )
    ax.text(
        0.450,
        0.045,
        "Resumable SSE Event Stream & Deterministic Replay Snapshots (Zero Re-Execution Read Path)",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#475569",
        bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "none", "alpha": 0.96},
        zorder=6,
        family="DejaVu Sans",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "project2_system_architecture.pdf"
    png_path = OUT_DIR / "project2_system_architecture.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"Generated {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
