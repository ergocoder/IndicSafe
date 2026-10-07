# IndicSafe / Indic-SafeBench

An automated framework for generating and benchmarking LLM safety data for Indian multilingual, transliterated and code-mixed languages. It is a BE major project.

**Current phase: 2 (Transformation Engine).** Built so far: configuration, source registry, Seed Manager, provenance, a 30-seed pilot, and the transformation engine (interfaces, lineage, validation hooks; no real MT/transliteration adapter yet). Code-mixing, the full QC pipeline, the review UI and the classifier are **not built yet**. See [docs/phase0_design.md](docs/phase0_design.md) for the design and the phase plan.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# import all seed-eligible sources, write the seed pool, export the pilot
.\.venv\Scripts\python.exe scripts\import_seeds.py

# options
.\.venv\Scripts\python.exe scripts\import_seeds.py --no-pilot
.\.venv\Scripts\python.exe scripts\import_seeds.py --manual-csv data\manual\seeds.csv
.\.venv\Scripts\python.exe scripts\import_seeds.py --force    # replace an existing, different pilot selection

# tests
.\.venv\Scripts\python.exe -m pytest
```

Outputs:

| Path | Content |
|---|---|
| `data/processed/seed_pool.jsonl` | Every imported row, including REJECTED and DUPLICATE rows with their reasons (git-ignored; can be rebuilt) |
| `data/processed/import_report.json` | Run manifest: counts per source, rejection reasons, parse failures, checksums, environment |
| `data/pilot/pilot_seeds_<version>.jsonl` / `.csv` | The pilot seeds |
| `data/pilot/pilot_seeds_<version>.manifest.json` | Selection method, distributions, fingerprint, checksums |

## Layout

```
backend/config.py          YAML loading + validation (pydantic)
configs/sources.yaml       raw-file registry: role, format, field mapping, SHA-256, licence status
configs/languages.yaml     languages, scripts (Unicode ranges), code-mix level bands
configs/taxonomy.yaml      frozen safety taxonomy (v1.0) + provisional source-category mappings
configs/generation.yaml    versions, random seed, seed validation, pilot, transformations/providers/hooks, QC thresholds
generator/schemas.py       SeedRecord, VariantRecord, TransformationRecord schemas
generator/source_readers.py  read-only zip readers
generator/text_utils.py    normalisation, dedup key, script detection
generator/provenance.py    checksums, run ids, manifests
generator/seed_manager.py  import / validate / de-duplicate / select / export
generator/transformation_engine.py  transformation interface, deterministic ids, lineage, validation hooks, provider registry, run export
generator/translation.py   TranslationProvider interface + translation transformation
generator/transliteration.py  Transliterator (script conversion) interface + transformation
generator/paraphrase.py    ParaphraseProvider interface + paraphrase transformation
scripts/import_seeds.py    CLI
tests/                     pytest suite (fixture mini-project + real-file checksum checks)
```

## Ground rules

- `data/raw/` is read-only. Files are read from inside their zips, and a checksum mismatch stops the import.
- `intended_label` is a provisional expectation that comes from the source. `final_label` is set only by human annotation.
- Nothing is silently dropped: rejected and duplicate rows are kept with a reason.
- No secrets in the code: copy `.env.example` to `.env` when later phases need API keys.
- The origin and licence of the raw datasets have not been verified yet (see `configs/sources.yaml`).
