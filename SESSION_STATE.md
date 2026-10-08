# IndicSafe — Session State

Last updated: 2026-10-08 · Telugu committed (dbdcfc1); pilot v0.2 run with Telugu committed by the user (98917f6: 614 variants, 571 pass; run `TRANSFORM_20261008T151239Z_b8ca7cbc`). **Review-sheet summary script (`scripts/summarize_reviews.py`) built and unit-tested, NOT committed**; waiting for filled sheets.

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
- **`--build`** writes `pilot_seeds_v0.2-pilot-seeds.{jsonl,csv,manifest.json}` only when every worksheet row has `adjudicated_label` (a valid label), `adjudicated_category` (an exact category_id), `adjudicated_by` and `rationale`, and the worksheet still matches the review layer. Otherwise it exits 2 and lists what is missing. **Built 2026-10-08:** `data/pilot/pilot_seeds_v0.2-pilot-seeds.{jsonl,csv,manifest.json}`, 30 seeds: 22 `human_agreed` + 8 `human_adjudicated`; final labels UNSAFE 16 / SAFE 12 / AMBIGUOUS 2. Adjudicated by bhargavi on all 8 rows (one of the two annotators, not a third person as the draft rules propose). An earlier build had cut three rationales at an unquoted comma; it was deleted, the importer now rejects rows with cells beyond the header, and Claude quoted those three cells at the user's request (text unchanged). Rebuilt with `--build --force` after committing the importer fix, so the manifest records a clean git state. `--build` now writes the v0.2 manifest before rewriting the tracked review report; otherwise the report's new timestamp would mark the build dirty. The manifest's git state is captured before any output is written, since the v0.2 files are tracked too. `.gitattributes` keeps `data/pilot/**` byte-for-byte (`-text`), so recorded checksums hold on any checkout. The S-NHQA-163 rationale was edited by the team after the first v0.2 build; v0.2 was rebuilt.
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
  - **HF cache is on D:** `HF_HUB_CACHE=D:\hf_cache` is a user environment variable (checked 2026-10-08). It holds IndicTrans2 (1.1 GB), LaBSE (1.8 GB) and all-MiniLM-L6-v2 (unused). C: keeps only the HF token and the remote-code module copies (about 0.4 MB). C: is nearly full (13 GB free).
  - Symlinks are not allowed on this account, so huggingface_hub runs in degraded (copy) mode. The LaBSE download failed once creating the `config.json` snapshot link; the blob was copied into place by hand (same 611 bytes).
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
- **Pilot run:** the preliminary v0.1 run (`TRANSFORM_20261007T225125Z_7dbbfe84`) was removed (`git rm`; still in history at b6822ba). It is replaced by the v0.2 run below (Phase 4/5 section).
- **Tests:** 180 pass, plus 1 GPU integration test (`-m integration`) that passes here and skips without torch / CUDA / the cached model.
  - 25 new unit tests. They use a fake model backend, a fake LID, and the real (fast) Aksharamukha and Lingua.

## Implemented (Phase 4 + 5 — code-mixing, QC pipeline; 2026-10-08, awaiting review)

- **Input = pilot v0.2.** `pilot_translation.load_pilot_seeds` reads the reviewed pilot. The human `final_label` becomes the generation seed's `intended_label` (basis text names v0.2 and the old provisional label), so every variant inherits the v0.2 label. `SeedRecord.label_status` now also allows `human_agreed` / `human_adjudicated`. Checked on the run: 0 of 564 variants differ from their seed's v0.2 final label.
- **Code-mixing** (`generator/code_mixing.py`, provider `mt_lexical_swap`, config `generation.yaml → code_mixing`; no LLM):
  - Tree: native L0 translation → `code_mixing` L1 / L2 (native script + English in Latin) → `transliteration` → Latn L1 / L2. 16 variants per seed. Every variant has `seed_id`, `parent_prompt_id`, `lineage`; code-mix records also carry `source_prompt_id` (the English root) in their parameters.
  - Alignment: each English content word (stopword list in config) is matched one-to-one to a target word by (a) phonetic: the romanised target word looks like the English one (loanwords such as फिशिंग / डेटा), or (b) lexical: IndicTrans2's translation of the word in isolation equals the target word, or its stem is a prefix of it. Marathi / Gujarati case markers (configured list) are kept as their own token: स्थलांतरितांना → "immigrants ना".
  - Swap: k aligned words go back to English. k is chosen so that the measured ratio is inside the band, nearest the band midpoint. L1's swaps are a subset of L2's. Loanwords whose romanisation already spells the English word are swapped last, because swapping them would not change the Latn variant.
  - All alignments, swapped words and word translations are in `provider_metadata`.
- **Measurement** (`generator/code_mix_metrics.py`, no model): word-level tags. In native-script text a word's tag follows its letters: native script → the language, Latin → English, no letters → language-independent. In Latn text each token takes its native parent's tag, position by position; a broken alignment means "unmeasurable" and FAIL. `code_mix_ratio` = en / (lang + en); CMI = 100·(1 − max/total).
- **Band check** (`code_mix_band` engine hook, also re-run in QC): PASS inside the band, WARN/REVIEW within ±0.03, FAIL beyond. A FAIL keeps its level and is never relabelled. A failed native variant gets no Latn child (`NOT_PRODUCED`).
- **Engine changes:** `TargetCondition` gains `code_mix_level` / `mixing_method`. `VariantRecord.code_mix_level/ratio/cmi/mixing_method` are filled. For code-mixed text the script check counts Latin letters too, and a Latin-dominated L2 is still recorded as native script if native letters are present. Transliteration keeps the parent's level.
- **Romaniser 1.1:** Latin words are passed through untouched; Aksharamukha used to lowercase some of them ("Munich" → "munich"). This changes the romanised L0 ids too.
- **QC pipeline** (`generator/qc_pipeline.py`, `scripts/run_qc.py --run <dir>`) → `qc_report.jsonl` (per variant, per check: status, reason, details) + `qc_summary.json`. The checks:
  - engine-hook failures;
  - exact duplicates (the shallower, then lower-level variant is kept);
  - near-duplicates (char-3 Jaccard ≥ 0.85 against other seeds of the same language/script/level);
  - script/language (the language_qc record);
  - code-mix band (L0 variants with English words → REVIEW);
  - length ratio vs parent;
  - semantic similarity.
- **Semantic check:** LaBSE (`setu4993/LaBSE`, pinned 5afa7296, ungated, Apache-2.0, fp16 on the GTX 1650, about 40 s to load). It compares cosine(English seed, native variant) against pass 0.80 / review 0.65. Latn variants inherit their native parent's score. Code-mixed variants also record their similarity to the L0 parent.
- **Run** `data/pilot/translations/TRANSFORM_20261008T085851Z_2164a890/` (v0.2, 30 seeds; GPU about 200 s plus model load). Its manifest says `dirty: true`, because the Phase 4/5 code was uncommitted.
  - Transformations: 90/90 translation, 174/180 code_mixing SUCCEEDED (6 band misses), 264/264 transliteration.
  - Language QC: 563 PASS, 1 REVIEW (the Marathi item from before).
  - **QC: 525 PASS / 30 REVIEW / 9 FAIL of 564.**
    - FAILs: 6 L2 variants with too few alignable words. They stay at the L1 ratio (0.11–0.17) and equal L1, so each is flagged both out of band and as a duplicate. The other 3 are romanised L2 variants identical to their L1 (the extra swap was an invisible loanword).
    - REVIEW: 20 semantic similarity 0.67–0.80 (no FAIL), 10 near the band edge, 1 LID.
  - Measured ratios. L1: mean 0.12–0.13 (range 0.09–0.17). L2: mean 0.25–0.27. L0: 0.00 for all languages.
  - LaBSE en↔L0 median 0.87–0.88; code-mixed variants score slightly higher (English words shared with the seed inflate similarity, so it is a weak drift signal for code-mix).
  - Review CSVs `review_{hi,mr,gu}.csv` regenerated (same format as before; native + Latn L0 only).
- **Tests:** 215 pass + 2 GPU integration tests (IndicTrans2, LaBSE) that pass here. New: `tests/test_code_mix.py` (25), `tests/test_qc_pipeline.py` (7), using fakes; no model needed.

## Code-mixing fixes and QC additions (2026-10-08, after the Phase 4+5 QC review; uncommitted, not yet run)

QC on `TRANSFORM_20261008T085851Z_2164a890` exposed "celebrate मनाने", "regarding बारे में", "serve આપી", "Sabha की" / "Lok चे", and gu "date" for માટે (phonetic false match).

- **English POS** (`generator/english_pos.py`): spaCy 3.8.16 + `en_core_web_sm` 3.8.0 installed from prebuilt cp313 wheels (no MSVC needed; added to requirements.txt).
  - Only NOUN, PROPN, ADJ, hyphenated compounds (`COMPOUND`) and VERB are swapped; function words never are.
  - `never_swap_en` lists preposition-like verb forms that spaCy tags VERB (regarding, including, …).
  - Fallback if spaCy is missing: `ClosedClassAnalyzer` (stopword list, no POS, so verbs are swapped like nouns). The tagger name and version are part of the code-mixer provider version (`2.0+…+pos:spacy-3.8.16+en_core_web_sm-3.8.0+verbs1…`), so all code-mix ids change.
- **Verbs:** English lemma + do-verb, keeping the native inflection through suffix tables in `generation.yaml → code_mixing…verbs.<lang>` (`do_forms`, `light_stems`, `suffixes`). Three constructions:
  - the next word is already a do-verb (ઉજવણી કરવા, आयोजित किया): replace only the aligned word;
  - the next word is a light verb (मना, दे, આપ) + suffix: replace both (जश्न मनाने → "celebrate करने", સેવા આપી → "serve કરી");
  - the word itself is stem + a suffix of ≥ 2 code points (चोरण्यासाठी → "steal करण्यासाठी").
  - Otherwise the verb is not swapped. Only common non-finite / habitual / simple-past forms are covered. `swap_verbs: false` turns verbs off.
- **Native function words** (`function_words` per language: postpositions, pronouns, copulas) are never replaced. Phonetic matches now need the same onset sound class (માટે "mate" ≠ "date").
- **Name spans:** consecutive PROPN tokens, or PROPN/NOUN tokens in one spaCy entity, form one unit: swapped whole or not at all.
  - One native compound built from the parts' heads: लोक+सभा → लोकसभा / लोकसभेचे / લોકસભાની → "Lok Sabha", "Lok Sabha चे", "Lok Sabha ની".
  - Or consecutive native words, one per part ("Pro Kabaddi League"; parts may match phonetically from 3 letters).
  - The whole span is also sent to IndicTrans2.
- **Offline replay** on the old run's L0 translations, using the word translations recorded in its metadata (no MT model; span translations missing, so a slight underestimate):
  - L2 band reached about 75/90, +6 near the edge, 9 missed. With `swap_verbs: false`, 18 L2 misses, so the verb construction stays on.
  - L1: 8/90 now miss. All 8 are short prompts whose only swappable unit is a 2–3-word name (Lok Sabha, Pro Kabaddi League, Muhammad bin Tughluq), which overshoots L1 when swapped whole. This is the all-or-none rule working as asked.
  - Real numbers come from the next pipeline run.
- **Semantic (QC 1.1):** code-mixed variants store `similarity_to_seed` and `similarity_to_parent` (vs the native L0 parent). The parent score decides REVIEW/FAIL (`decision_basis: native_l0_parent`). Thresholds unchanged. `QCRecord.semantic_similarity_to_parent` added. `semantic.py` itself was not changed.
- **FAIL handling:** `qc_pipeline.final_dataset_variants(variants, qc_records, include_review=True)` is the single filter for any export. It never returns FAIL, returns REVIEW only when `include_review=True`, and raises if a variant has no QC record. No exporter exists yet.
  - `qc_summary.json` gets `final_dataset` counts and `code_mix_coverage`: per level, a `line` such as "L2 band reached: n/N", plus near-edge, missed and per-language counts.
- **Review sheets:** `review_codemix_{hi,mr,gu}.csv` are written by `scripts/run_qc.py`, one row per native L1/L2 variant plus its romanised child (FAIL variants left out).
  - Rows: every QC REVIEW item first, then a fixed-seed random sample up to 20 rows, with UNSAFE rows taken first until 8 are on the sheet (`qc.human_review.codemix_sheet_rows / codemix_sheet_min_unsafe`).
  - Columns: source English, L0 native, native, romanised, ratio, CMI, swapped words, QC status/reasons; reviewer columns `codemix_natural_1to3`, `intent_preserved_Y_N`, `notes`.
- **Harmful-intent check:** `harmful_intent_check.json`. The 10 native-script variants of UNSAFE/AMBIGUOUS seeds with the lowest similarity to the English seed, with seed text, variant text, romanised text, both scores and QC status (`harmful_intent_top_n`).
- **Tests:** 235 pass (including the 2 GPU integration tests). `tests/test_code_mix.py` (40) uses a `FakeAnalyzer` whose tables are spaCy's real output, plus real pilot L0 texts and word translations:
  - Oktoberfest hi has no "celebrate मनाने" and has "celebrate करने के लिए";
  - S-NHQA-39 never swaps "regarding", even through a leaky tagger;
  - Lok Sabha for hi/mr/gu;
  - the verb-construction table, the onset check, name spans.
  - `test_spacy_units_match_the_tables` checks the fake tables against real spaCy (skipped if spaCy is missing). `tests/test_qc_pipeline.py` (12) covers the parent-based semantic decision, the final-dataset filter, coverage, the review sheet and the harmful-intent report.

## Telugu as 4th target language (2026-10-08; uncommitted, not yet run)

- **Config:** `languages.yaml` → `te` (Telugu; native_script `Telu` = U+0C00–0C7F; romanized Latn; enabled; `code_mix_partner: en` only so QC can measure English words in L0; `code_mix_levels: [L0]`).
  - Notes say why code-mixing is off: Telugu case markers are suffixes fused to the noun. It needs suffix-aware swapping and native review first.
  - `generation.yaml`: te added to the `indictrans2` and `colloquial_roman` target_languages and to `qc.language.candidates`. It is **not** in `code_mixing` target_languages.
- **Translation:** `FLORES["te"] = "tel_Telu"`. The vendored IndicProcessor already knew tel_Telu. The model writes Devanagari and the processor converts it to Telugu script; a fake-backend unit test checks this, including batching and caching.
- **Romanisation (romaniser 1.2):** separate Dravidian path; no Devanagari pivot, no schwa deletion, no final-nasal rule. Aksharamukha Telugu → ISO 15919 → `telugu_colloquial`:
  - jñ → gn; vocalic r → ri;
  - anusvara → n before a stop or nasal, else m;
  - ISO c / ch → ch / chh; ś ṣ → sh;
  - remaining diacritics stripped (long vowels collapse; retroflexes plain);
  - lowercase. Latin runs pass through.
  - Examples: "miru ela unnaru?", "nenu pustakam chaduvutunnanu", "krishnudu gnanam gurinchi cheppadu", "bharatadeshamlo e nadi podavainadi?".
  - The version bump **changes the ids of every romanised variant (hi/mr/gu too) and of every code-mixed variant**, because the code-mixer version embeds the romaniser version. Hindi/Marathi/Gujarati romanised text itself is unchanged; a test checks Hindi.
- **Language QC:** the script check separates Telu from Deva/Gujr. Lingua identifies Telugu; checked with the real Lingua in a unit test. No Telugu marker words were needed (markers only arbitrate hi vs mr).
- **Pipeline:**
  - `run_pilot_translation.py` defaults to every enabled target language in languages.yaml (hi mr gu te).
  - Code-mixing skips a language the mixer doesn't support, so te gets native + Latn only (2 variants per seed: 1 translation + 1 romanised).
  - `review_te.csv` is written like the others.
  - QC: `qc_summary.json` gets `code_mix_scope` (`code_mixed` / `not_code_mixed`); te has no coverage lines and no `review_codemix_te.csv`. Its L0 variants still get the L0 English-word check.
- **Tests:** 246 pass with `-m "not integration"` (13 new in `tests/test_telugu.py`). Two config tests that pinned the language lists were updated; `FakeTransliterator` accepts Telu. The GPU integration test now also translates te, but it was **not run** this time.

## Review-sheet summary (2026-10-08; uncommitted)

- **`generator/review_summary.py` + `scripts/summarize_reviews.py --run <run dir> --sheets <folder>`.**
  - Reads `review_<lang>*.csv` and `review_codemix_<lang>*.csv` for every enabled target language (hi mr gu te). Several files per language are allowed.
  - Reviewer = the row's `reviewer` cell, else the filename suffix (`review_hi_bhargavi.csv`), else `unknown`.
  - A code-mix sheet is expected only for languages in `code_mixing` target_languages. A te code-mix sheet is ignored with a warning; te reports `not_applicable`.
- **Per language and reviewer (and ALL):**
  - translation: rated / total, mean adequacy, mean fluency, % intent Y, mean romanisation;
  - code-mix: mean naturalness (1–3), % rated 1, % intent Y, plus the same for UNSAFE rows only;
  - the 10 lowest-rated rows with notes. Score = ratings scaled to 0–1, intent N = 0, then averaged.
- **Robustness:**
  - Blank rows are skipped. A row counts as rated when any score or the intent cell is filled.
  - Bad cells (not a number, out of range, intent not Y/N) are ignored with a warning; "4,0" is read as 4.
  - A non-UTF-8 file (Excel's plain "CSV" save) or one missing the expected columns (e.g. `;` delimiter) is skipped. All files unreadable → status `unreadable`; no files → `no_sheet`.
  - Every native/latin prompt id is checked against the run's `variants.jsonl`; mismatches are warned and counted.
- **Outputs:** `review_summary.json` + `review_summary.md` in the run folder, plus a compact printed table. Per-reviewer rows appear only when there are several reviewers.
- **Checked read-only** on the real unfilled sheets of `TRANSFORM_20261008T151239Z_b8ca7cbc`: all 7 sheets parsed, 0 prompt-id mismatches, nothing written.
- **Tests:** `tests/test_review_summary.py` (8, small fake CSVs). Full unit suite: 254 pass (`-m "not integration"`).

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
- **Telugu added (2026-10-08, user instruction):** te as 4th target language, native Telu + Latn at L0 only. No code-mixing until suffix-aware swapping exists and native reviewers check it. Telugu reviewers: not yet known.
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
- **Code-mixing limits:**
  - Alignment uses isolated-word MT, so words the MT renders differently in context are missed. A prefix match can hit a wrong inflection.
  - POS comes from spaCy on the English seed only. A spaCy mis-tag ("stage" as NOUN in "Munich first stage Oktoberfest") passes through. The verb tables cover common forms only; finite and irregular verbs are not swapped. A short prompt whose only unit is a long name cannot reach L1.
  - Inflection is dropped with the swapped word.
  - Native-script measurement cannot tell names from code-mixed words (no NER). Latn measurement is inherited from the native parent; there is no independent romanised-text tagger, because the data has no romanised hi/mr/gu with word tags.
  - Naturalness has not been human-reviewed.
- **Semantic check limits:** LaBSE thresholds are uncalibrated. Shared English words inflate code-mix similarity.

## Phase plan (build-guide order, guide §9)

| # | Phase | Status |
|---|---|---|
| 1 | Seed Manager + provenance | **Done**; all 9 guide-§7 checks re-verified on 2026-10-08 |
| 2 | Transformation Engine: interface, translation adapter (IndicTrans2 default, optional LLM), paraphrase and transliteration interfaces, `parent_prompt_id`/`seed_id` lineage, validation hooks | **Done, awaiting review** (interfaces only; real IndicTrans2 / transliteration adapters still to add) |
| 2a | Pilot review layer: double-annotation import, agreement, adjudication → v0.2 | **Done**; pilot v0.2 built (8 adjudicated) |
| 2b | Real adapters: IndicTrans2 translation + chosen romanisation method; pilot translation evaluation | **Done, awaiting review**; native-speaker review of the CSVs pending |
| 3 | Language/script layer | **Done, awaiting review** (script check + Lingua/marker LID, QC records) |
| 4 | Code-mixing engine | **Built, awaiting review** (mt_lexical_swap, L1/L2, native + Latn) |
| 5 | QC pipeline | **Built, awaiting review** (dup / near-dup / script+LID / band / length / LaBSE semantic) |
| 6 | Generation jobs | |
| 7 | Human review | |
| 8 | Dataset release (versioning, seed-level splits, exports) | |
| 9 | Benchmark | |
| 10 | Classifier | |
| 11 | Secondary factuality | |

The SQLite store, exporters and splitter from the design doc's §12 are built inside these phases when first needed. They don't become separate phases.

## Exact next steps

0. Review and commit `scripts/summarize_reviews.py`. When filled sheets come back, ask reviewers to save as **CSV UTF-8**; `summarize_reviews.py --run data\pilot\translations\TRANSFORM_20261008T151239Z_b8ca7cbc --sheets <folder>`. (Done since: Telugu committed in dbdcfc1 and run in 98917f6.) Earlier note: review and commit the Telugu addition. Then run `python -m pytest -m integration` (real IndicTrans2 now also translates te), `scripts\run_pilot_translation.py` (now hi mr gu te) and `scripts\run_qc.py --run <new run>`. Check te LaBSE scores and LID, and find a Telugu reviewer for `review_te.csv`. All romanised and code-mixed ids change with romaniser 1.2.
1. Review Phase 4/5: `generator/{code_mixing,code_mix_metrics,qc_pipeline,semantic}.py`, the engine / romaniser / language_qc changes, `configs/generation.yaml` (code_mixing, qc), and the run folder `TRANSFORM_20261008T085851Z_2164a890` (`qc_report.jsonl`, `qc_summary.json`).
2. Commit in PowerShell: `git add .; git commit -m "phase 4 code-mixing, phase 5 qc pipeline, pilot v0.2 run"`. The run's manifest will still say `dirty: true`; rerun both scripts after committing if a clean manifest is wanted.
3. Team: send the regenerated `review_{hi,mr,gu}.csv` (v0.2 run) to native speakers. They contain the UNSAFE pilot prompts in translation. Gujarati reviewers are still unconfirmed. Consider adding the code-mixed variants to the sheets: naturalness of the swaps has had no human check yet.
4. Calibrate on the review results: the semantic thresholds, the code-mix bands, the stopword and clitic lists, and IndicTrans2 as the production model.
5. Decide what to do with L2 band misses on short prompts: accept fewer L2 variants, allow lexical matches with a lower score, or add a POS/NER-aware aligner.
6. Next phase: 6 (generation jobs).

## Commands

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\import_seeds.py      # --no-pilot | --manual-csv <csv> | --force
.\.venv\Scripts\python.exe scripts\import_reviews.py    # review layer + agreement; --build [--force] -> pilot v0.2
.\.venv\Scripts\python.exe scripts\run_pilot_translation.py   # v0.2, hi mr gu te (+ L1/L2 for hi mr gu); --no-code-mix --languages hi --limit N
.\.venv\Scripts\python.exe scripts\run_qc.py --run data\pilot\translations\<run_id>   # --no-semantic
.\.venv\Scripts\python.exe scripts\summarize_reviews.py --run data\pilot\translations\<run_id> --sheets <filled CSV folder>
.\.venv\Scripts\python.exe -m pytest                          # -m integration: GPU tests (IndicTrans2, LaBSE)
```

If pip times out on the 2.6 GB torch wheel, download it with `curl -C -` and `pip install` the file.
