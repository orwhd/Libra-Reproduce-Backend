# Data Provenance & Sampling Documentation (Project 2)

## 1. Primary Empirical Dataset (S1 — Author Replication Release)
- **Target Paper**: Wu, L., Wang, D., & Evans, J. A. (2019). *Large teams develop and small teams disrupt science and technology*. **Nature**, 566, 378–382. `https://doi.org/10.1038/s41586-019-0941-9`
- **Upstream Repository**: Harvard Dataverse — `https://doi.org/10.7910/DVN/JPWNNK` (Version 4.0, File ID `6925921`: `Aggregated_20210521.txt`, `850,713,255` bytes, MD5 `8dea725aa83835d57232dc96b50b75a7`).
- **License**: **CC0 1.0 Universal (Public Domain Dedication)** — `http://creativecommons.org/publicdomain/zero/1.0` (verified directly via Harvard Dataverse metadata API; redistribution of derived subsets is explicitly permitted).
- **Verified Column Schema (6 Tab-Separated Columns)**:
  1. `mag_paper_id` (`int64`) — Microsoft Academic Graph Paper ID
  2. `year` (`int32`) — Publication year (`1830–2021`)
  3. `field_code` (`int32`) — MAG Level-0 Field-of-Study code (`0..18`, mapped to `field_name`)
  4. `team_size` (`int32`) — Distinct author count ($\ge 1$)
  5. `citation_count` (`int32`) — Cumulative forward citations in MAG snapshot
  6. `disruption` (`float64`) — Author-precomputed Funk & Owen-Smith disruption index $D \in [-1, 1]$
  *(Note: Although the Dataverse prose abstract mentions "Multi-Institution or not", empirical byte inspection confirms `Aggregated_20210521.txt` contains 6 columns without that flag.)*
- **Deterministic Stratified Byte-Window Protocol**:
  - `libra_reproduce/science/dataverse_loader.py` fetches 20 deterministic `128 KiB` HTTP `206 Partial Content` slices across byte offsets `[25 MiB .. 810 MiB]`, discards partial boundary lines at slice edges, deduplicates `mag_paper_id`, and retains `N = 60,120` papers (`1954–2021`), including `45,335` papers in the primary `1954–2014` historical window and `14,785` papers in `2015–2021` for right-truncation sensitivity analysis.
  - Cached in `data/samples/wu2019_mag_stratified_sample.csv.gz` for deterministic offline execution; pass `--force-download` or delete the cached file to re-stream directly from Harvard Dataverse.

## 2. Bounded S2 Citation-Graph Dataset (`wu2019_s2_openalex_bounded.csv`)
- **Source**: OpenAlex Works API (`https://api.openalex.org/works`, CC0 1.0 Open Metadata).
- **Purpose**: Exercises independent graph-level recomputation $D = (n_i - n_j)/(n_i + n_j + n_k)$ comparing 5-year (`D_5y`) vs. 10-year (`D_10y`) forward citation observation windows alongside deterministic unit toy graphs (`SCI-02`).
