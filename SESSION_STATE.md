# IndicSafe — Session State

Last updated: 2026-10-08 · Build-guide Phase 1 complete, **not committed** · next: guide Phase 2 (Transformation Engine)

## Objective

Build a reproducible, provenance-rich dataset generator for an Indian multilingual, transliterated and code-mixed LLM safety benchmark (labels SAFE / UNSAFE / AMBIGUOUS).

- The spec is `docs/IndicSafe_Master_Claude_Build_Prompt.txt`.
- The full Phase 0 design is in `docs/phase0_design.md`.
- Work one phase at a time, and stop for review after each.

## Implemented (Phase 0/1)

- **Configs:** YAML files validated by pydantic (`backend/config.py`). Languages, sources, categories and thresholds are not hard-coded in Python.
- **Source registry** (`configs/sources.yaml`): all 15 raw zips, each with its role, format, field mapping, SHA-256 checksum and licence status (all `UNVERIFIED`).
- **Read-only zip readers.** Malformed rows are reported with their line numbers. Bare `NaN` values are converted to `null`.
- **Seed Manager:**
  - Imports and validates rows. It checks length, control characters, U+FFFD, mojibake, and that the measured script matches the declared one.
  - Marks exact duplicates, including across sources.
  - Optionally imports a team-written seed CSV (`--manual-csv`).
  - Selects the pilot deterministically and stratified by category, with a word-Jaccard guard against templated near-duplicates.
  - Exports JSONL and CSV plus a manifest.
- **Provenance:**
  - Each import checks the raw-file checksums and stops on a mismatch.
  - Run IDs look like `IMPORT_<ts>_<config fingerprint>`.
  - Manifests record config hashes, random seed, versions, environment, git commit and a `dirty` flag.
- **Pilot:** `data/pilot/pilot_seeds_v0.1-pilot-seeds.{jsonl,csv,manifest.json}`, 30 seeds:
  - 12 NicheHazardQA + 8 data_for_hub + 10 dataset_10k.
  - Provisional labels: 20 UNSAFE, 10 SAFE.
- **Seed pool:** `data/processed/seed_pool.jsonl` (git-ignored, rebuildable).
  - nichehazardqa: 388 valid.
  - data_for_hub: 1,938 valid, 22 duplicates.
  - dataset_10k: 617 valid (English factual and Indian questions only).
- **Tests:** 86 pass.
  - They run on a fixture mini-project built in a temp directory, plus checksum checks on the real files.
  - A mutation check confirmed the tests catch broken duplicate detection and a broken diversity guard.

## Key decisions

- **Raw data stays zipped** in `data/raw/` and is never extracted or modified. All 15 checksums were verified unchanged after the work.
- **`intended_label` is provisional and source-derived.** `final_label` is set only by human annotation; the schema rejects it at import.
- **Nothing is deleted.** Rejected and duplicate rows are kept with `rejection_reasons` or `duplicate_of`.
- **Seed IDs are stable:** `S-<PREFIX>-<source ref>` (e.g. `S-NHQA-366`), or `S-MAN-<hash>` for team-written seeds.
- **data_for_hub:**
  - Only the `question` field is imported; the conversations (which contain harmful model output) are not.
  - Its `topic` field is an academic subject, so these seeds get category `unassigned`.
- **dataset_10k:**
  - English factual and Indian questions are the benign controls.
  - Its Hindi/Marathi parallel rows are candidate reference translations; whether they are human or machine translations is unknown.
- **The LID/MLI/MT/NER/POS/TN files are support data only**, never seeds or labels. They are Hindi–English only and share one sentence pool. LID test overlaps LID train (4,120 of 5,000 sentences), so validators must be evaluated on a de-duplicated held-out subset.
- **MVP languages (decided 2026-10-08):** en (reference) plus **hi**, **mr** and **gu**, each in native script (Deva / Gujr) and Latn, at code-mix levels L1 and L2 with English. That is 16 variants per seed. Native-speaker annotators are confirmed for hi and mr; Gujarati annotators are not yet confirmed.
- **Taxonomy frozen as v1.0** (2026-10-08): 14 categories plus `unassigned`. Any change needs a new `taxonomy_version`.
- **Translation (decided 2026-10-08):** only through a provider interface. The default is local open-source MT (IndicTrans2 is the first candidate); an LLM adapter is optional and off. No LLM API is hard-coded. Phase 2 evaluates translation on a small pilot before the production model is chosen. Configured in `configs/generation.yaml` → `translation`.
- **Code-mix measure:** `code_mix_ratio = secondary / (primary + secondary)` language-tagged tokens. CMI is reported too.
  - Bands: L0 < 0.05 ≤ L1 < 0.20 ≤ L2 < 0.35 ≤ L3 ≤ 0.50.
- **QC thresholds** are design values in `configs/generation.yaml`, still to be calibrated.
- **Do not use `docs/Topic Approval Presentation.pdf`** (user instruction).

## Files created / modified

- **New:**
  - `configs/{sources,languages,taxonomy,generation}.yaml`
  - `backend/{__init__,config}.py`
  - `generator/{__init__,schemas,source_readers,text_utils,provenance,seed_manager}.py`
  - `scripts/import_seeds.py`
  - `tests/{__init__,conftest,test_config,test_text_utils,test_seed_manager,test_provenance,test_real_sources}.py`
  - `README.md`, `docs/phase0_design.md`, `requirements.txt` (pydantic, PyYAML, pytest), `pytest.ini`, `SESSION_STATE.md`
- **Modified:** `.env.example` (placeholders only, no keys) and `.gitignore` (adds `data/processed/`, `*.tmp`).
- **Outputs:** `data/pilot/*` (meant to be committed) and `data/processed/*` (ignored).
- **Deleted by the user** (shown in `git status`): `docs/BE_Major_Project_Summary_for_Claude.pdf` and `docs/Topic Approval Presentation.pdf`.

## Known issues / limitations

- **Source labels and categories are often wrong.** For example, `S-NHQA-366` (phishing) maps to `dangerous_instructions` but is really `cyber_misuse`, and `S-DFH-1127` looks benign. Human review must fix these; nothing is relabelled automatically.
- **All seeds are English-origin** so far.
- **There is no language detector yet.** Script detection alone cannot tell hi from mr, since both use Devanagari.
- **There is no Marathi–English or Gujarati–English code-mix gold data** in `data/raw/`, and no romanised Gujarati references. The team has named candidate external sources: IndicGuard, SurakshaEval, L3Cube-MeCorpus/MeHate, Bhasha SFT (Soket AI Labs) and AIKosh. Each must be inspected for mr-en/gu-en content, licence, task fit and benchmark overlap before use (`docs/phase0_design.md` §14).
- **The thresholds and code-mix bands are untested starting values.**
- **Licences:** the team confirmed all raw files are open and free to use (`OPEN_TEAM_CONFIRMED` in `configs/sources.yaml`). Exact licence names and citations are still needed for the dataset card. The 42 MB of raw zips are committed to git.
- **No database yet:** storage is JSONL files plus manifests.
- **Duplicate prompts in a manual CSV get the same `S-MAN-<hash>` ID.** They are rejected as `duplicate_seed_id` rather than marked DUPLICATE.
- **Manually written code-mixed seeds:** not now (team decision); possibly later.

## Phase plan (build-guide order, guide §9)

| # | Phase | Status |
|---|---|---|
| 1 | Seed Manager + provenance | **Done**; all 9 guide-§7 checks re-verified on 2026-10-08 |
| 2 | Transformation Engine: interface, translation adapter (IndicTrans2 default, optional LLM), paraphrase and transliteration interfaces, `parent_prompt_id`/`seed_id` lineage, validation hooks | **Next** (use the guide §10 prompt) |
| 3 | Language/script layer | |
| 4 | Code-mixing engine | |
| 5 | QC pipeline | |
| 6 | Generation jobs | |
| 7 | Human review | |
| 8 | Dataset release (versioning, seed-level splits, exports) | |
| 9 | Benchmark | |
| 10 | Classifier | |
| 11 | Secondary factuality | |

The SQLite store, exporters and splitter from the design doc's §12 are built inside these phases when first needed. They don't become separate phases.

## Exact next steps

1. Inspect the pilot: `data/pilot/pilot_seeds_v0.1-pilot-seeds.csv`.
2. Commit the milestone in PowerShell:

   ```powershell
   git add .; git commit -m "phase 0 project audit and seed manager"
   ```

   This also commits the two PDF deletions; run `git restore docs/` first if they were accidental.
3. Start a **new Claude Code chat** and paste the build guide's §10 prompt (Phase 2, Transformation Engine). Begin it with: "Read SESSION_STATE.md first."
4. Still open on the team side:
   - Gujarati annotators.
   - Code-mix reference sources for mr and gu, inspected in Phase 4 (code-mixing).

## Commands

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\import_seeds.py      # --no-pilot | --manual-csv <csv> | --force
.\.venv\Scripts\python.exe -m pytest
```
