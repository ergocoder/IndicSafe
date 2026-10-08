# IndicSafe — Session State

Last updated: 2026-10-08 · Phase 2b + 3 committed (b6822ba). Pilot adjudication done; **pilot v0.2 built and committed** · next: rerun pilot translations on v0.2, native-speaker review of the pilot translations, then Phase 4

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

## Implemented (Phase 2 — Transformation Engine)

- **Modules:** `generator/transformation_engine.py` (abstract `Transformation`, engine, provider registry, validation hooks, lineage check/trace, run export), `generator/{translation,transliteration,paraphrase}.py` (adapter ABCs + transformations). Test doubles in `tests/fakes.py` (generation_method `mock`).
- **Records** (`generator/schemas.py`, frozen): `VariantRecord` (design §5.2 fields; root = identity copy of a VALID seed with `parent_prompt_id=None`; every child has `parent_prompt_id`, `seed_id`, `seed_version`, `lineage` = ancestor ids) and `TransformationRecord` (spec §11 + resolved parameters, provider/version/model, request fingerprint, derived seed, raw output, hook results, status SUCCEEDED / VALIDATION_FAILED / ERROR).
- **Deterministic ids:** `T-<sha256(canonical request)[:16]>`, `P-<seed suffix>-<same hash>`. Request = type + parent id + parent content hash + resolved params + provider name/version/model. Run id and wall clock are excluded. Per-request `derived_seed` = hash(random_seed, request).
- **Validation hooks** (configured per type in `generation.yaml` → `transformation_engine.validation_hooks`): non_empty, text_integrity, expected_script (qc.script thresholds), differs_from_parent (`condition_not_realised`), length_ratio (WARN). Failed children are kept but can't be parents unless `allow_failed_parent=True`.
- **Providers** are built from config via `register_provider` / `build_provider`. **No real adapter is implemented:** `indictrans2` (configured default) raises `ProviderUnavailableError` until its adapter exists; LLM adapters are disabled; transliteration has no default provider yet.
- **Not built (by instruction):** LLM generation, code-mixing engine, language ID, semantic/label-consistency QC, batch jobs, CLI script, SQLite.
- **Tests:** 140 pass (54 new). Mutation checks confirmed the tests catch removed hooks, a constant derived seed, and dropping the parent content hash from ids.

## Implemented (pilot review layer, before Phase 2b)

- **Inputs** (`data/pilot/reviews/`): the two annotator CSVs (bhargavi, swasthik), the team's `adjudication_worksheet_v0.1.csv` + `adjudication_rules_draft.md` (not created by Claude), and the new `reviews_v0.1.yaml` (paths, annotators, review dates) and `category_mapping_v0.1.csv`.
- **Module** `generator/review_import.py`, CLI `scripts/import_reviews.py`.
  - Reads only `seed_id`, `content_hash`, `my_label`, `my_category`, `note`. All other columns are ignored (bhargavi's Excel-stripped `source_reference`, swasthik's blank `import_run_id` header).
  - Matches on seed_id, requires every pilot seed exactly once, and stops on any content_hash mismatch. Also checks the v0.1 JSONL against its manifest checksum.
  - **Category mapping:** an exact taxonomy `category_id` passes through. Anything else must be in `category_mapping_v0.1.csv` (keys compared case-insensitively with whitespace collapsed; types `display_name` / `typo` / `multiple`). Blank → `BLANK_CATEGORY`, not in the table → `UNMAPPED_CATEGORY`, multi-category cell → `MULTIPLE_CATEGORIES`. All three stay unresolved (category_id null) and send the seed to adjudication. Typos are mapped but flagged `TYPO_MAPPED`.
  - Raw label/category/note, annotator, review date (given by the team: swasthik 2026-10-03, bhargavi 2026-10-07; not in the files), source file and sha256 are kept per annotation.
- **Outputs:** `review_layer_v0.1.jsonl` (one record per seed) and `review_report_v0.1.json` (agreement, flagged cells, every distinct raw category value and how it was resolved, worksheet check, input checksums).
- **Results on the real files:**
  - Label: 23/30 = 76.67 %, Cohen's κ = 0.584. Category (27 seeds where both cells resolved): 25/27, κ = 0.903.
  - Flags: S-NHQA-250 swasthik multiple categories; S-NHQA-253 and S-DFH-307 bhargavi blank category; typo-mapped: S-NHQA-6 (swasthik), S-DFH-1281 (both).
  - 8 seeds need adjudication (7 label, 1 category-only: S-DFH-307). **The worksheet matches exactly:** same 8 seeds, same `issue`, labels, categories, notes, prompts and source labels.
- **`--build`** writes `pilot_seeds_v0.2-pilot-seeds.{jsonl,csv,manifest.json}` only when every worksheet row has `adjudicated_label` (a valid label), `adjudicated_category` (an exact category_id), `adjudicated_by` and `rationale`, and the worksheet still matches the review layer. Otherwise it exits 2 and lists what is missing. **Built 2026-10-08:** `data/pilot/pilot_seeds_v0.2-pilot-seeds.{jsonl,csv,manifest.json}`, 30 seeds: 22 `human_agreed` + 8 `human_adjudicated`; final labels UNSAFE 16 / SAFE 12 / AMBIGUOUS 2. Adjudicated by bhargavi on all 8 rows (one of the two annotators, not a third person as the draft rules propose). An earlier build had cut three rationales at an unquoted comma; it was deleted, the importer now rejects rows with cells beyond the header, and Claude quoted those three cells at the user's request (text unchanged). Rebuilt with `--build --force` after committing the importer fix, so the manifest records a clean git state. `--build` now writes the v0.2 manifest before rewriting the tracked review report; otherwise the report's new timestamp would mark the build dirty.
  - Agreed seeds (same label and same resolved category) take the shared values (`label_status: human_agreed`). Worksheet seeds take the adjudicated values (`human_adjudicated`). `category_status: human_assigned`, and `intended_label` stays as the provisional history.
  - Each record gets `dataset_version`, `parent_dataset_version` and a `review` block: both annotations (raw + mapped), pre-review category/label, resolution, and adjudication (by, rationale, worksheet sha256).
  - v0.1 is only read. The build refuses to overwrite an existing v0.2 unless `--force` is given.
  - `seed_version` is unchanged, because the prompt text did not change and variant ids depend on the content.
- **Not done (by instruction):** no adjudications filled and no `final_label` set by Claude; no new worksheet.
- **Tests:** 16 in `tests/test_review_import.py`, run on temp copies of the real files with the worksheet's adjudication columns blanked, so they don't depend on the team's edits. Full suite: 182 pass.

## Implemented (Phase 2b + 3 — real adapters, language/script layer)

Full notes: `docs/phase2b_3_notes.md`.

- **Environment** (Windows, GTX 1650 4 GB, Python 3.13):
  - `torch 2.14.1+cu126`; `torch.cuda.is_available()` is True.
  - The wheel was downloaded with resume to `D:\wheels\`, because pip's 2.6 GB download kept getting reset.
  - `transformers 4.57.6`. 5.x cannot be used: the model's remote code imports `transformers.onnx`, removed in 5.0.
- **Hugging Face:** logged in as ergocoder; access to the gated model is confirmed.
  - Model cached in the default HF cache on C:, about 1.1 GB.
  - **C: has only about 1.3 GB free** (2026-10-08, after the model download). Set `HF_HUB_CACHE` or the `cache_dir` option to use D:.
- **Translation** (`generator/indictrans2.py`): provider `indictrans2`.
  - Model `ai4bharat/indictrans2-en-indic-dist-200M`, pinned to commit `173b9423…`. Remote code checked: imports only torch, transformers and sentencepiece.
  - Runs fp16 on CUDA; `device: auto` falls back to fp32 on CPU. On OOM the batch is halved down to 1, then the item fails as ERROR.
  - Beam 5, max_new_tokens 256, batch_size 2.
  - **KV cache off:** the remote code breaks with the transformers 4.57 cache objects. This costs speed only.
  - IndicTransToolkit `IndicProcessor` is **vendored** as a mechanical pure-Python port (`generator/vendor/`, MIT). The PyPI sdist needs MSVC, which isn't installed. A compiled install is preferred automatically.
  - Records: `provider_version` = `1.0+float16+beam5+max256`, `generator_model` = `<repo>@<sha>`. `provider_metadata` holds the revision, device, dtype, decoding parameters, token count, truncation flag, preprocessor and library/GPU versions.
- **Romanisation** (`generator/romanization.py`): provider `colloquial_roman`, now the configured default.
  - Aksharamukha `RomanColloquial` + `RemoveSchwaHindi` + final anusvara → n. Gujarati goes through a Devanagari pivot.
  - IndicXlit is **not installable** here: it depends on fairseq, which has no Windows/py3.13 build.
  - Compared against ISO 15919 / ITRANS on pilot outputs (notes §5). The strict schemes are unreadable as user text.
  - Known errors: schwa deletion in compounds (*lokasbheche*), phonetic loanwords (*deta*).
- **Language/script QC** (`generator/language_qc.py`): one `LanguageQCRecord` per variant in `language_qc.jsonl`.
  - **Script check** with the existing `qc.script` thresholds.
  - **LID** = Lingua 2.2.0 over en/hi/mr/gu, plus hi/mr function-word markers. The markers apply when there are ≥ 2 of them, because Lingua alone is near chance on hi vs mr.
  - **Statuses:** PASS ≥ 0.80; otherwise REVIEW, or FAIL on a confident mismatch.
  - Romanised / code-mixed text is `NOT_APPLICABLE` for LID.
  - Config: `qc.language` (typed `LanguageQCConfig`).
- **Pipeline:** `generator/pilot_translation.py` + `scripts/run_pilot_translation.py`.
  - Batched translate → engine → romanise → QC → `review_{hi,mr,gu}.csv` (utf-8-sig, blank reviewer columns) + `pilot_translation_summary.json`.
  - Adapters register on `import generator.providers`.
- **Pilot run** `data/pilot/translations/TRANSFORM_20261007T225125Z_7dbbfe84/` (**preliminary**, input v0.1):
  - 90/90 translations and 90/90 romanisations SUCCEEDED; no warnings or truncation; about 155 s on the GPU.
  - Language QC 179/180 PASS. One Marathi item is REVIEW (correct Marathi, too few markers).
- **Tests:** 180 pass, plus 1 GPU integration test (`-m integration`) that passes here and skips without torch / CUDA / the cached model.
  - 25 new unit tests. They use a fake model backend, a fake LID, and the real (fast) Aksharamukha and Lingua.

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
- **Language ID is heuristic for hi vs mr** (Lingua + hand-written markers, uncalibrated). Romanised text is not language-identified.
- **Romanisation is one fixed colloquial spelling per word**, with known schwa and loanword errors (see `docs/phase2b_3_notes.md` §5).
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
| 2 | Transformation Engine: interface, translation adapter (IndicTrans2 default, optional LLM), paraphrase and transliteration interfaces, `parent_prompt_id`/`seed_id` lineage, validation hooks | **Done, awaiting review** (interfaces only; real IndicTrans2 / transliteration adapters still to add) |
| 2a | Pilot review layer: double-annotation import, agreement, adjudication → v0.2 | **Done**; pilot v0.2 built (8 adjudicated) |
| 2b | Real adapters: IndicTrans2 translation + chosen romanisation method; pilot translation evaluation | **Done, awaiting review**; native-speaker review of the CSVs pending |
| 3 | Language/script layer | **Done, awaiting review** (script check + Lingua/marker LID, QC records) |
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

1. Review Phase 2b/3 code and outputs: `generator/{indictrans2,romanization,language_qc,pilot_translation,providers}.py`, `generator/vendor/`, `docs/phase2b_3_notes.md`, the run folder under `data/pilot/translations/`.
2. Commit in PowerShell: `git add .; git commit -m "phase 2b+3 adapters, language qc, preliminary pilot translations"`.
3. Team: send `review_hi.csv`, `review_mr.csv` and `review_gu.csv` to native speakers. The sheets contain the UNSAFE pilot prompts in translation; reviewers should know that. Gujarati reviewers are still unconfirmed.
4. Rerun `scripts\run_pilot_translation.py --seeds data/pilot/pilot_seeds_v0.2-pilot-seeds.jsonl`. Variant ids for unchanged seeds stay the same.
5. Use the review results to calibrate the QC thresholds and markers and to confirm IndicTrans2 as the production model.
6. Next phase (new chat, "Read SESSION_STATE.md first."): Phase 4 code-mixing engine (hi-en / mr-en / gu-en, L1/L2). It needs mr/gu code-mix references (team sources, still to inspect).

## Commands

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\import_seeds.py      # --no-pilot | --manual-csv <csv> | --force
.\.venv\Scripts\python.exe scripts\import_reviews.py    # review layer + agreement; --build [--force] -> pilot v0.2
.\.venv\Scripts\python.exe scripts\run_pilot_translation.py   # --seeds <jsonl> --languages hi mr gu --limit N
.\.venv\Scripts\python.exe -m pytest                          # -m integration: GPU test only
```

If pip times out on the 2.6 GB torch wheel, download it with `curl -C -` and `pip install` the file.
