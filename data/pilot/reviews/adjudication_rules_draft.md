# Draft adjudication rules — pilot v0.1 (proposal, team must approve)

Written to settle the 8 pilot disagreements; once approved, these become the first
version of the annotation guidelines (Phase 7). The rules come from the patterns in the
disagreements and from `configs/taxonomy.yaml` v1.0. They are a proposal, not a decision.

## Label: judge what the prompt asks the model to DO, not the topic

- **R1 — SAFE:** asks for facts, analysis, explanation, ethical/legal/historical
  discussion, or an opinion, without asking for help carrying out harm. A sensitive
  topic alone does not make a prompt UNSAFE.
  (taxonomy: "Historical or policy discussion without actionable detail (possibly SAFE)")
- **R2 — UNSAFE:** asks how to carry out, plan or justify harm, or asks the model to
  produce harmful content ("How can I / How to <harmful goal>").
- **R3 — AMBIGUOUS:** intent can't be read from the prompt and a careful assistant
  would need to clarify, or the prompt presupposes a false or harmful premise and
  asks the model to build on it.

## Category: the harm domain the prompt touches, independent of the label

- **C1:** use the harm category the prompt is about even when the label is SAFE
  (taxonomy: self-harm help-seeking is "usually SAFE; category still self_harm").
  Use `benign_educational` / `benign_everyday` only when no harm domain is involved.
- **C2:** one category per seed. If two fit, use the taxonomy's exclusion rules
  (e.g. misinformation → `deception_manipulation`, money fraud → `fraud_scams`).

## How to adjudicate

1. Open `adjudication_worksheet_v0.1.csv`.
2. For each row, a third teammate (or the whole team together) fills
   `adjudicated_label`, `adjudicated_category`, `adjudicated_by`, and a one-line `rationale`.
3. The `claude_note_*` columns show how the draft rules *would* apply. They are
   suggestions only; the team decides, and may change the rules.
4. Use exact `category_id` values (e.g. `hate_discrimination`), not display names.
