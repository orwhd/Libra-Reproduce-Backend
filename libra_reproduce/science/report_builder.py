from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docx
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .evidence_gate import EvidenceReviewResult


def build_reproduction_reports(
    *,
    output_dir: Path,
    run_id: str,
    user_prompt: str,
    data_manifest: dict[str, Any],
    experiment_summary: dict[str, Any],
    evidence_result: EvidenceReviewResult,
    fig_main_path: Path,
    fig_window_path: Path,
    debate_records: list[dict[str, Any]],
    sdk_provider_used: str,
    script_sha256: str,
) -> dict[str, Path]:
    """
    Generate both Word (.docx) and PDF (.pdf) scientific reproduction reports from verified
    on-disk artifacts, manifests, figures, and Evidence Gate findings.
    Independent of any frontend or browser rendering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    level_badge = (
        f"{evidence_result.reproduction_level_verified} (Author MAG Precomputed D Reanalysis) "
        "+ Bounded S2 Citation-Graph Verification"
    )

    docx_path = output_dir / "reproduction_report.docx"
    pdf_path = output_dir / "reproduction_report.pdf"

    _write_docx_report(
        docx_path=docx_path,
        run_id=run_id,
        generated_ts=generated_ts,
        level_badge=level_badge,
        user_prompt=user_prompt,
        data_manifest=data_manifest,
        experiment_summary=experiment_summary,
        evidence_result=evidence_result,
        fig_main_path=fig_main_path,
        fig_window_path=fig_window_path,
        debate_records=debate_records,
        sdk_provider_used=sdk_provider_used,
        script_sha256=script_sha256,
    )
    _write_pdf_report(
        pdf_path=pdf_path,
        run_id=run_id,
        generated_ts=generated_ts,
        level_badge=level_badge,
        user_prompt=user_prompt,
        data_manifest=data_manifest,
        experiment_summary=experiment_summary,
        evidence_result=evidence_result,
        fig_main_path=fig_main_path,
        fig_window_path=fig_window_path,
        debate_records=debate_records,
        sdk_provider_used=sdk_provider_used,
        script_sha256=script_sha256,
    )
    return {"docx_path": docx_path, "pdf_path": pdf_path}


def _write_docx_report(
    *,
    docx_path: Path,
    run_id: str,
    generated_ts: str,
    level_badge: str,
    user_prompt: str,
    data_manifest: dict[str, Any],
    experiment_summary: dict[str, Any],
    evidence_result: EvidenceReviewResult,
    fig_main_path: Path,
    fig_window_path: Path,
    debate_records: list[dict[str, Any]],
    sdk_provider_used: str,
    script_sha256: str,
) -> None:
    doc = docx.Document()
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(10.5)

    title = doc.add_heading(
        "Scientific Paper Reproduction Report: Large Teams Develop and Small Teams Disrupt Science and Technology",
        level=1,
    )
    for run in title.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)

    meta_p = doc.add_paragraph()
    meta_p.add_run("Target Paper: ").bold = True
    meta_p.add_run(
        "Wu, L., Wang, D., & Evans, J. A. (2019). Large teams develop and small teams disrupt science and technology. "
        "Nature, 566(7744), 378-382. https://doi.org/10.1038/s41586-019-0941-9\n"
    )
    meta_p.add_run("Run ID: ").bold = True
    meta_p.add_run(f"{run_id}   |   ")
    meta_p.add_run("Generated: ").bold = True
    meta_p.add_run(f"{generated_ts}\n")
    meta_p.add_run("Verified Reproduction Scope: ").bold = True
    meta_p.add_run(f"{level_badge}\n")
    meta_p.add_run("Evidence Gate Status: ").bold = True
    meta_p.add_run(f"{evidence_result.result.upper()}   |   Harness Provider: {sdk_provider_used}\n")
    meta_p.add_run("Research Question: ").bold = True
    meta_p.add_run(user_prompt)

    doc.add_heading("1. Executive Summary & Core Reproduction Verdict", level=2)
    doc.add_paragraph(
        f"Using a controlled multi-agent scientific reproduction harness with independent Evidence Gate verification, "
        f"we evaluated the central empirical claim of Wu, Wang & Evans (Nature 2019) on an audited sample of "
        f"N = {experiment_summary['primary_sample_n']:,} scholarly articles (1954-2014 primary historical window; "
        f"{experiment_summary['total_sample_n']:,} total records through 2021) drawn from the authors' official "
        f"Harvard Dataverse Microsoft Academic Graph (MAG) replication repository (doi:10.7910/DVN/JPWNNK, CC0-1.0)."
    )
    doc.add_paragraph(
        f"Consistent with Nature 2019 Fig. 2, disruption declines monotonically with team size while forward citation "
        f"impact increases: solo authors (k=1) achieve a mean disruption score of {experiment_summary['solo_mean_disruption']:.4f} "
        f"(mean percentile {experiment_summary['solo_disruption_percentile']:.1f}) and {experiment_summary['solo_mean_citations']:.1f} "
        f"mean citations, whereas teams of 11+ authors fall to {experiment_summary['large_11plus_mean_disruption']:.4f} "
        f"(mean percentile {experiment_summary['large_11plus_disruption_percentile']:.1f}) while averaging "
        f"{experiment_summary['large_11plus_mean_citations']:.1f} citations (OLS slope = {experiment_summary['ols_slope_per_author']:.5f} "
        f"per author, p = {experiment_summary['ols_p_value']:.2e}; Spearman r = {experiment_summary['spearman_team_vs_disruption_r']:.4f})."
    )

    doc.add_heading("2. Data Provenance, Schema Verification & S1/S2 Scope", level=2)
    prov_items = [
        f"Primary Dataset (S1): {data_manifest['source_doi_or_url']} ({data_manifest['upstream_filename']}, License: {data_manifest['license']}).",
        f"Dataset SHA-256: {data_manifest['file_sha256']} | Analysis Script SHA-256: {script_sha256}.",
        f"Empirical Schema Audit: {data_manifest['schema_verification_note']}",
        f"Sampling Protocol: 20 deterministic 128 KiB HTTP Range slices across chronological offsets (1954-2021), dropping partial boundary lines ({data_manifest['malformed_boundary_rows_dropped']} boundary lines dropped) and deduplicating MAGPaperId.",
        "S1 vs. S2 Distinction: Main MAG statistics are classified strictly as S1 (independent statistical reanalysis of author-published MAG disruption scores). Independent graph-level recomputation D = (n_i - n_j)/(n_i + n_j + n_k) is verified separately via S2 unit toy graphs and a bounded OpenAlex citation neighborhood.",
    ]
    for item in prov_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("3. Quantitative Results by Team Size (1954-2014 Primary Cohort)", level=2)
    bins = experiment_summary.get("bins", [])
    table = doc.add_table(rows=1, cols=7)
    table.style = "Light Grid Accent 1"
    headers = [
        "Team Size",
        "Sample N",
        "Mean D",
        "95% Bootstrap CI",
        "Disruption Pct.",
        "Field-Demeaned D",
        "Mean Citations",
    ]
    for idx, h in enumerate(headers):
        table.rows[0].cells[idx].text = h

    for b in bins:
        row_cells = table.add_row().cells
        row_cells[0].text = str(b["team_bin"])
        row_cells[1].text = f"{int(b['sample_n']):,}"
        row_cells[2].text = f"{b['mean_disruption']:.5f}"
        row_cells[3].text = f"[{b['ci95_low']:.5f}, {b['ci95_high']:.5f}]"
        row_cells[4].text = f"{b['mean_disruption_percentile']:.1f}"
        row_cells[5].text = f"{b['field_demeaned_disruption']:.5f}"
        row_cells[6].text = f"{b['mean_citations']:.2f}"

    doc.add_heading("4. Reproduced Figures & Citation-Window Sensitivity", level=2)
    if fig_main_path.exists():
        doc.add_picture(str(fig_main_path), width=Inches(6.2))
        doc.add_paragraph(
            "Figure 1: (a) Mean disruption percentile (95% bootstrap CI) vs. mean forward citations across team sizes 1 to 11+ "
            "in the 1954-2014 MAG sample. (b) Within-discipline raw disruption trajectories across the top 5 sampled MAG fields."
        )
    if fig_window_path.exists():
        doc.add_picture(str(fig_window_path), width=Inches(6.2))
        doc.add_paragraph(
            "Figure 2: (a) Observation-window comparability check contrasting 1954-2004 (>=17y forward window), 2005-2014, "
            "and right-truncated 2015-2021 cohorts. (b) Bounded S2 independent citation-graph recomputation (5-year vs. 10-year window)."
        )

    doc.add_heading("5. Multi-Agent Validation Debate & Research Gain", level=2)
    for drec in debate_records:
        doc.add_paragraph(
            f"Debate [{drec.get('id', 'debate_01')}] ({drec.get('status', 'resolved')}): {drec.get('claim', '')} — "
            f"{drec.get('resolution_summary', '')}",
            style="List Bullet",
        )

    for cat_title, key in [
        ("Verified Insights", "verified_insights"),
        ("Ruled-Out Explanations", "ruled_out_explanations"),
        ("Open Questions & Limitations", "open_questions"),
        ("Methodological Lessons", "methodology_lessons"),
    ]:
        items = evidence_result.research_gains.get(key, [])
        if items:
            p = doc.add_paragraph()
            p.add_run(f"{cat_title}:").bold = True
            for it in items:
                doc.add_paragraph(it, style="List Bullet")

    doc.add_heading("6. Standalone Reproduction Instructions", level=2)
    doc.add_paragraph(
        "To regenerate all datasets, checksums, figures, Evidence Gate audits, and DOCX/PDF reports purely from the backend CLI:\n"
        "  cd backend && uv run python -m libra_reproduce.cli run --paper-id paper_A "
        "--prompt \"How does team size influence the balance between disruptive innovation and developmental research in science and technology?\""
    )
    doc.save(str(docx_path))


def _write_pdf_report(
    *,
    pdf_path: Path,
    run_id: str,
    generated_ts: str,
    level_badge: str,
    user_prompt: str,
    data_manifest: dict[str, Any],
    experiment_summary: dict[str, Any],
    evidence_result: EvidenceReviewResult,
    fig_main_path: Path,
    fig_window_path: Path,
    debate_records: list[dict[str, Any]],
    sdk_provider_used: str,
    script_sha256: str,
) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RepTitle",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "RepH2",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "RepBody",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5,
    )
    caption_style = ParagraphStyle(
        "RepCap",
        parent=body_style,
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )

    story: list[Any] = []
    story.append(
        Paragraph(
            "Paper Reproduction Report: Large Teams Develop and Small Teams Disrupt Science and Technology",
            title_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Target Paper:</b> Wu, Wang &amp; Evans (<i>Nature</i>, 2019, DOI: 10.1038/s41586-019-0941-9)<br/>"
            f"<b>Run ID:</b> {run_id} &nbsp;|&nbsp; <b>Generated:</b> {generated_ts}<br/>"
            f"<b>Reproduction Level:</b> {level_badge}<br/>"
            f"<b>Evidence Gate Verdict:</b> {evidence_result.result.upper()} &nbsp;|&nbsp; "
            f"<b>Harness Engine:</b> {sdk_provider_used}<br/>"
            f"<b>Query:</b> {user_prompt}",
            body_style,
        )
    )

    story.append(Paragraph("1. Executive Summary", h2_style))
    story.append(
        Paragraph(
            f"We empirically evaluated Wu, Wang &amp; Evans (2019) on an audited CC0-1.0 sample of "
            f"<b>N = {experiment_summary['primary_sample_n']:,}</b> MAG papers (1954-2014 primary cohort; "
            f"{experiment_summary['total_sample_n']:,} total rows through 2021) from Harvard Dataverse "
            f"(doi:10.7910/DVN/JPWNNK). Solo authors (k=1) exhibit higher disruption "
            f"(mean D = {experiment_summary['solo_mean_disruption']:.4f}, percentile {experiment_summary['solo_disruption_percentile']:.1f}) "
            f"than 11+ teams (mean D = {experiment_summary['large_11plus_mean_disruption']:.4f}, percentile "
            f"{experiment_summary['large_11plus_disruption_percentile']:.1f}), whereas forward citations increase "
            f"from {experiment_summary['solo_mean_citations']:.1f} to {experiment_summary['large_11plus_mean_citations']:.1f} "
            f"(OLS slope = {experiment_summary['ols_slope_per_author']:.5f}/author, p = {experiment_summary['ols_p_value']:.2e}).",
            body_style,
        )
    )

    story.append(Paragraph("2. Data Provenance &amp; Quantitative Summary", h2_style))
    story.append(
        Paragraph(
            f"<b>Dataset SHA-256:</b> <font name='Courier' size='8'>{data_manifest['file_sha256']}</font><br/>"
            f"<b>Script SHA-256:</b> <font name='Courier' size='8'>{script_sha256}</font><br/>"
            f"<b>Schema Audit:</b> {data_manifest['schema_verification_note']}",
            body_style,
        )
    )

    table_data = [["Team Bin", "Sample N", "Mean D", "95% CI", "Pct.", "Field-Dem. D", "Mean Cites"]]
    for b in experiment_summary.get("bins", []):
        table_data.append(
            [
                str(b["team_bin"]),
                f"{int(b['sample_n']):,}",
                f"{b['mean_disruption']:.5f}",
                f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}]",
                f"{b['mean_disruption_percentile']:.1f}",
                f"{b['field_demeaned_disruption']:.5f}",
                f"{b['mean_citations']:.2f}",
            ]
        )

    tbl = Table(table_data, colWidths=[0.75 * inch, 0.85 * inch, 0.85 * inch, 1.45 * inch, 0.65 * inch, 0.95 * inch, 0.85 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Reproduced Figures &amp; Sensitivity Checks", h2_style))
    if fig_main_path.exists():
        story.append(RLImage(str(fig_main_path), width=6.4 * inch, height=2.57 * inch))
        story.append(
            Paragraph(
                "Figure 1: Team size vs. disruption percentile and citations (1954-2014 MAG sample, S1) and within-field trajectories.",
                caption_style,
            )
        )
    if fig_window_path.exists():
        story.append(RLImage(str(fig_window_path), width=6.4 * inch, height=2.46 * inch))
        story.append(
            Paragraph(
                "Figure 2: Citation observation window sensitivity (1954-2004 vs. 2005-2014 vs. 2015-2021) and bounded S2 recomputation.",
                caption_style,
            )
        )

    story.append(Paragraph("4. Validation Debate, Limitations &amp; Reproduction Command", h2_style))
    for drec in debate_records:
        story.append(
            Paragraph(
                f"• <b>Debate ({drec.get('status', 'resolved')}):</b> {drec.get('claim', '')} — {drec.get('resolution_summary', '')}",
                body_style,
            )
        )
    for lim in data_manifest.get("known_biases", []):
        story.append(Paragraph(f"• <b>Limitation:</b> {lim}", body_style))

    story.append(
        Paragraph(
            "<b>CLI Reproduction Command:</b> <font name='Courier' size='8'>"
            "uv run python -m libra_reproduce.cli run --paper-id paper_A --prompt "
            "\"How does team size influence the balance between disruptive innovation and developmental research?\""
            "</font>",
            body_style,
        )
    )

    doc_pdf = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    doc_pdf.build(story)
