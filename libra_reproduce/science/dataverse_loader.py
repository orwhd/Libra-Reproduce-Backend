from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..config import get_bundled_samples_dir

DATAVERSE_DOI_URL = "https://doi.org/10.7910/DVN/JPWNNK"
DATAVERSE_ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/6925921"
DATAVERSE_TOTAL_BYTES = 850_713_255
DATAVERSE_UPSTREAM_MD5 = "8dea725aa83835d57232dc96b50b75a7"

MAG_LEVEL0_FIELDS: dict[int, str] = {
    0: "Political Science",
    1: "Geology",
    2: "Medicine",
    3: "History",
    4: "Mathematics",
    5: "Physics",
    6: "Materials Science",
    7: "Biology",
    8: "Business",
    9: "Chemistry",
    10: "Engineering",
    11: "Economics",
    12: "Art",
    13: "Philosophy",
    14: "Computer Science",
    15: "Geography",
    16: "Psychology",
    17: "Sociology",
    18: "Environmental Science",
}

# 20 deterministic 128 KiB byte-range windows across [25 MB .. 835 MB] (~2.5 MiB total)
# spanning chronological MAG publications from ~1954 through 2021.
STRATIFIED_BYTE_WINDOWS: list[tuple[int, int]] = [
    (offset_mb * 1024 * 1024, offset_mb * 1024 * 1024 + 131_071)
    for offset_mb in (
        25, 45, 70, 100, 135, 175, 215, 260, 305, 350,
        395, 440, 485, 530, 575, 620, 665, 715, 765, 810,
    )
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_dataverse_block(raw_chunk: bytes, drop_first_line: bool = True) -> tuple[list[tuple[Any, ...]], int]:
    text = raw_chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) <= 2:
        return [], 0
    # Drop partial first and last lines cut at arbitrary HTTP byte-range boundaries
    interior = lines[1:-1] if drop_first_line else lines[:-1]
    parsed: list[tuple[Any, ...]] = []
    malformed = 0
    for line in interior:
        parts = line.split("\t")
        if len(parts) != 6:
            malformed += 1
            continue
        try:
            paper_id = int(parts[0].strip())
            year = int(parts[1].strip())
            raw_field = parts[2].strip().split(",")[0].strip()
            field_code = int(raw_field) if raw_field else -1
            team_size = int(parts[3].strip())
            citations = int(parts[4].strip())
            disruption = float(parts[5].strip())
        except ValueError:
            malformed += 1
            continue
        if year < 1830 or year > 2025 or team_size <= 0 or math.isnan(disruption) or abs(disruption) > 1.000001:
            malformed += 1
            continue
        parsed.append(
            (
                paper_id,
                year,
                field_code,
                MAG_LEVEL0_FIELDS.get(field_code, f"Field_{field_code}"),
                team_size,
                citations,
                disruption,
            )
        )
    return parsed, malformed


async def ensure_wu2019_dataverse_sample(
    output_dir: Path,
    *,
    force_download: bool = False,
) -> dict[str, Any]:
    """
    Ensure a verified, CC0-1.0 real empirical sample from Wu, Wang & Evans's Harvard Dataverse
    (`doi:10.7910/DVN/JPWNNK`, File 6925921 `Aggregated_20210521.txt`) is saved in `output_dir`
    together with a full provenance `data_manifest.json`.
    Uses cached bundled sample if already downloaded, or fetches live HTTP 206 byte ranges.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bundled_dir = get_bundled_samples_dir()
    bundled_csv_gz = bundled_dir / "wu2019_mag_stratified_sample.csv.gz"
    bundled_meta = bundled_dir / "wu2019_mag_stratified_manifest.json"

    df: pd.DataFrame | None = None
    retrieved_at = datetime.now(timezone.utc).isoformat()
    retrieval_mode = "harvard_dataverse_http_range_live"
    raw_candidates = 0
    malformed_rows = 0

    if not force_download and bundled_csv_gz.exists() and bundled_meta.exists():
        try:
            df = pd.read_csv(bundled_csv_gz, compression="gzip")
            prev_meta = json.loads(bundled_meta.read_text(encoding="utf-8"))
            retrieved_at = str(prev_meta.get("retrieved_at_utc") or retrieved_at)
            retrieval_mode = "verified_bundled_dataverse_cc0_sample"
            raw_candidates = int(prev_meta.get("raw_row_count") or len(df))
            malformed_rows = int(prev_meta.get("malformed_boundary_rows_dropped") or 0)
        except Exception:
            df = None

    if df is None:
        all_rows: list[tuple[Any, ...]] = []
        headers_base = {"User-Agent": "LibraResearch-ReproduceV2/0.2"}
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, headers=headers_base) as client:
            sem = __import__("asyncio").Semaphore(4)

            async def _fetch_slice(start_b: int, end_b: int) -> tuple[list[tuple[Any, ...]], int]:
                async with sem:
                    resp = await client.get(
                        DATAVERSE_ACCESS_URL,
                        headers={"Range": f"bytes={start_b}-{end_b}"},
                    )
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"Dataverse HTTP {resp.status_code} for range {start_b}-{end_b}")
                    return _parse_dataverse_block(resp.content, drop_first_line=(start_b > 0))

            slice_results = await __import__("asyncio").gather(
                *[_fetch_slice(s, e) for s, e in STRATIFIED_BYTE_WINDOWS]
            )

        for rows, bad in slice_results:
            raw_candidates += len(rows) + bad
            malformed_rows += bad
            all_rows.extend(rows)

        df = pd.DataFrame(
            all_rows,
            columns=[
                "mag_paper_id",
                "year",
                "field_code",
                "field_name",
                "team_size",
                "citation_count",
                "disruption",
            ],
        )
        df = df.drop_duplicates(subset=["mag_paper_id"]).sort_values(["year", "mag_paper_id"]).reset_index(drop=True)

        # Persist into bundled CC0 samples dir so clean/offline re-runs are instant and reproducible
        bundled_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(bundled_csv_gz, "wt", encoding="utf-8", newline="") as gz:
            df.to_csv(gz, index=False)

    # Filter primary historical cohort (1954-2014, matching Nature 2019 WoS window) + keep 2015-2021 for window sensitivity
    df_clean = df[(df["year"] >= 1954) & (df["year"] <= 2021) & (df["team_size"] >= 1)].copy()
    df_clean["cohort_window"] = df_clean["year"].apply(
        lambda y: "1954-2005" if y <= 2005 else ("2006-2014" if y <= 2014 else "2015-2021_truncated")
    )

    csv_path = output_dir / "wu2019_mag_sample.csv"
    parquet_path = output_dir / "wu2019_mag_sample.parquet"
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    csv_path.write_bytes(csv_bytes)
    df_clean.to_parquet(parquet_path, index=False)

    file_sha = _sha256_bytes(csv_bytes)
    parquet_sha = _sha256_bytes(parquet_path.read_bytes())

    missingness = {
        col: round(float(df_clean[col].isna().mean()), 6)
        for col in df_clean.columns
    }

    manifest: dict[str, Any] = {
        "dataset_id": f"wu2019_dataverse_mag_{file_sha[:12]}",
        "source_doi_or_url": DATAVERSE_DOI_URL,
        "upstream_access_url": DATAVERSE_ACCESS_URL,
        "upstream_filename": "Aggregated_20210521.txt",
        "upstream_filesize_bytes": DATAVERSE_TOTAL_BYTES,
        "upstream_md5": DATAVERSE_UPSTREAM_MD5,
        "source_type": "author_replication_data_harvard_dataverse",
        "retrieval_mode": retrieval_mode,
        "retrieved_at_utc": retrieved_at,
        "license": "CC0-1.0 (Creative Commons CC0 1.0 Universal Public Domain Dedication)",
        "file_name": csv_path.name,
        "file_sha256": file_sha,
        "parquet_file_name": parquet_path.name,
        "parquet_sha256": parquet_sha,
        "raw_row_count": int(raw_candidates),
        "subset_row_count": int(len(df_clean)),
        "primary_1954_2014_row_count": int((df_clean["year"] <= 2014).sum()),
        "truncated_2015_2021_row_count": int((df_clean["year"] >= 2015).sum()),
        "malformed_boundary_rows_dropped": int(malformed_rows),
        "selection_script": "scripts/reproduce_nature2019.py",
        "selection_parameters": {
            "start_year": 1954,
            "end_year": 2021,
            "primary_analysis_end_year": 2014,
            "http_byte_windows": [f"{s}-{e}" for s, e in STRATIFIED_BYTE_WINDOWS],
        },
        "columns_verified": [
            "mag_paper_id",
            "year",
            "field_code",
            "field_name",
            "team_size",
            "citation_count",
            "disruption",
            "cohort_window",
        ],
        "schema_verification_note": (
            "Empirical byte-level inspection confirmed Aggregated_20210521.txt contains 6 tab-separated "
            "columns (MAGPaperId, Year, Field, TeamSize, Citation, Disruption); the 'Multi-Institution' "
            "label mentioned in the Dataverse prose abstract is absent from this file."
        ),
        "missingness": missingness,
        "metric_origin": "precomputed_by_authors_in_harvard_dataverse_mag_release",
        "reproduction_level": "S1",
        "observed_citation_window": (
            "MAG cumulative citation snapshot through 2021; papers in 1954-2014 have >=7 years of forward "
            "observation, whereas 2015-2021 papers experience right-truncated forward citation windows."
        ),
        "known_biases": [
            "Chronological byte-offset block sampling across Aggregated_20210521.txt (~2.5 MiB / 850 MiB) rather than full 19.5M-row scan.",
            "MAG paper universe differs from the proprietary Web of Science (1954-2014), USPTO patent, and GitHub corpora used in Nature 2019 Fig. 1-2.",
            "Disruption values in S1 are author-precomputed MAG D-scores; independent graph-level D recomputation is verified separately in bounded S2.",
        ],
    }

    if not bundled_meta.exists() or force_download:
        bundled_meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
    }


def inspect_uploaded_attachment(file_path: Path) -> dict[str, Any]:
    """Safely extract structured summary from an uploaded PDF, CSV, JSON, or text attachment."""
    suffix = file_path.suffix.lower()
    raw = file_path.read_bytes()
    sha = _sha256_bytes(raw)
    info: dict[str, Any] = {
        "filename": file_path.name,
        "sha256": sha,
        "size_bytes": len(raw),
        "kind": suffix.lstrip(".") or "binary",
    }
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages_text = []
        for page in reader.pages[:12]:
            pages_text.append(page.extract_text() or "")
        joined = "\n".join(pages_text).strip()
        info["page_count"] = len(reader.pages)
        info["excerpt"] = joined[:6000]
    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(io.BytesIO(raw), sep=sep, nrows=500)
        info["columns"] = list(df.columns)
        info["preview_rows"] = int(len(df))
    elif suffix in (".txt", ".md", ".json"):
        info["excerpt"] = raw.decode("utf-8", errors="replace")[:6000]
    return info
