# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase-6 M0: Scenario & Rule Engine design record (`docs/08_scenario_engineering/`).
  The knowledge layer that turns detections over the 23-class taxonomy into
  elderly-care scenarios (risk level → patient prompt → next best action →
  caregiver alert). **This phase is spec-drift reconciliation plus the missing
  governed knowledge dataset, not a new engine** — `src/pipeline/` already ships
  a rule evaluator, event memory, an alert queue, a TTS sink and six live rules,
  and `near(class_a, class_b, pixel_dist)` has been documented as a V1 condition
  at `docs/02_technical_architecture_specification/rule_engine.md:34` since
  Phase 2 without ever being implemented.
  - `architecture_review.md` — runtime as built vs. as documented, including
    **ten defects verified against the code**: every feature flag is inert
    (`orchestrator.py:49` reads a `feature_flags:` root key that does not exist,
    so `passport: false  # privacy` is a false claim); the only CRITICAL rule
    cannot fire cold (`event_memory.py:114` returns the 150-frame window for
    never-seen ⇒ 10.0 s, against a 30 s threshold); recall-tuned per-class
    thresholds are structurally unreachable (`detector.py:133` filters at the
    global 0.25 first, and `class_thresholds.yaml` is never loaded at runtime);
    `any_of([...])` silently discards the rest of its condition; AND/OR
    precedence is inverted; one bad severity string zeroes all alerts silently
    and the rule engine has **zero tests**; `save_csv_report` writes CRLF while
    its JSON sibling documents why LF is mandatory; and the `AnalysisPlugin`
    seam has three mutually incompatible contracts with no registration
    mechanism. These are the content of M1.
  - `requirements_specification.md` — Scenario/Rule/Predicate/Event definitions,
    the eight validity conditions, mandatory vs optional fields, and the
    forbidden assumptions (absence of `person` ≠ absence of a person;
    single-resident homes; a working system; repetition as escalation).
  - `domain_research_report.md` — CDC STEADI, WHO falls guidance, Katz ADL /
    Lawton–Brody IADL, alarm-fatigue evidence, DPDP Act 2023 posture, and the
    `evidence` field contract (`supports: hazard` never licenses
    `detectability: direct`).
  - `negative_register.md` — scenarios permanently out of scope with these 23
    static-object classes (fall detection, "stove is on", gas leaks, medication
    adherence, hydration, wandering), plus the three live rules being **deleted
    rather than migrated** (`knife_near_person`, `medicine_reminder`,
    `gas_cylinder_check`; projected day-one volume ~370 alerts, ~0 actionable).
  - `adr/ADR-P6-01…09` — per-scenario YAML master in `configs/scenarios/` ·
    CSV is an export-only view · structured AST over the string DSL ·
    `BaseRuleEngine` injection over the plugin seam · class capability map (the
    taxonomy fingerprint provably cannot catch the scheduled `wet_floor`
    demotion, since class ID 20 stays reserved) · `rule_hash` semantic
    versioning with a derived-semver gate · `near()` units and room-vs-zone ·
    rejected-scenario negative register · additive extension of the LOCKED
    `Alert` contract.
- Phase-6 M1: `src/scenario_engine/` package with `EvalContext` — the single
  input type every scenario predicate will receive (detections **with bounding
  boxes and multiplicity**, temporal memory via a `MemoryView` Protocol, the
  measured FPS, frame id, and deployment room). This is why "extend the existing
  DSL" was not an option: the legacy evaluator was handed a bare `set[str]`, so
  geometry and multiplicity were destroyed before any predicate ran, making
  `near`/`overlaps`/`count` unimplementable rather than merely awkward. A
  half-done `near(person, stove, 0.2)` would have silently degraded into
  `detected(person) AND detected(stove)` — the exact rule being deleted for
  false positives — while looking implemented.
  `tests/unit/scenario_engine/test_layering.py` statically enforces the ADR-P6-04
  dependency direction (the scenario engine is a leaf; `src/pipeline` must never
  import it), scanning the AST so lazy and conditional imports are caught too.
- Phase-6 M2: predicate registry and structured trigger trees.
  - `src/scenario_engine/predicates/` — the fourth instance of the house
    registry pattern (`completeness_policies.py`, `splitting/registry.py`,
    `annotation/registry.py`). A predicate is a pure function of `EvalContext`
    plus declaratively-typed arguments, so one generic checker produces every
    error message and each predicate gets a table-driven test with no YAML in
    the loop. Twelve built-ins across five *kinds*: `detected`/`any_of`/`all_of`
    (static) · `count` (counting) · `absent_for`/`present_for`/`ever_seen`
    (temporal) · `near`/`overlaps`/`above`/`below` (spatial) · `in_room`
    (context). An unknown predicate raises and enumerates the valid set, where
    the legacy engine returned `False` and produced a rule that never fired.
  - `src/scenario_engine/trigger.py` — triggers are authored as nested
    `all`/`any`/`not`/`op` structures, validated at parse time, and compiled to
    an immutable tree. Precedence is structural rather than implied, empty
    `all`/`any` are rejected rather than silently constant, and `not` composes
    with every predicate. Errors name the field path
    (`trigger.all[1].args.max_dist`) instead of a character offset — the whole
    reason for structured trees over condition strings. `render_trigger`
    emits the one-line string for the CSV view and logs; it is never parsed back.
  - `trigger_kinds`/`is_static_only` give the compiler the check that
    disqualifies a trigger satisfiable by object presence alone. Tests assert it
    rejects all three legacy rules being deleted (`knife_near_person`,
    `medicine_reminder`, `gas_cylinder_check`) while accepting the same
    conditions once a dwell or proximity term is added.
  - `near` is pinned by test to be **false for two distant detections in the
    same frame** — the failure that would silently reduce every spatial
    predicate to co-presence — and to be true for adjacent boxes whose IoU is 0,
    which is why `overlaps` cannot replace it.
  - `src/scenario_engine/schema.py` — the scenario row as frozen dataclasses
    with a hand-rolled `_validate`, matching the house pattern (no pydantic; it
    would be a new dependency and a convention break needing its own ADR).
    The **trigger tree is the only authored condition**: `required_objects` is
    derived from it so the CSV view cannot drift from the executable rule, and
    negation stays inside the trigger rather than being split back out into a
    flat `forbidden_objects` list.
    `caregiver_channel` is an enum (`none`/`digest`/`push`/`push_and_call`) —
    as a boolean it gets set true everywhere, fatiguing the caregiver whose
    attention actually protects the resident.
    Safety invariants enforced at construction: `status: active` requires
    `reviewed_by`/`reviewed_on` (this project already demands two annotators at
    IAA ≥ 0.75 to accept a bounding box); `detectability: inferred` requires a
    `capability_disclaimer`; `detectability: rejected` requires a
    `rejection_reason` naming the missing capability; `quiet_hours: always_speak`
    is CRITICAL-only; `max_repeats` is capped at 3; escalation steps must be
    ordered.
  - `rule_hash` (ADR-P6-06) spans the trigger and the behavioural scalars only.
    Tests pin both directions: a risk downgrade, a dwell change, a threshold
    change or a trigger edit **moves** the hash; a reworded prompt, a note, a
    renamed scenario or an evidence edit **does not**. Getting this backwards
    either trains contributors to ignore staleness warnings or lets the clip
    suite pass green against behaviour that no longer exists.
- Phase-6 M3: the scenario compiler, class capability map and CSV view.
  - `src/scenario_engine/taxonomy.py` — the **class capability map** (ADR-P6-05),
    derived from `configs/data.yaml` + `configs/feature_flags.yaml` + a
    git-tracked R24 decision artifact. A taxonomy fingerprint answers "is the
    taxonomy the same?", which is not the question a scenario needs: the
    documented `wet_floor` demotion keeps class ID 20 reserved, so `nc` and
    `names` are unchanged and the fingerprint is **byte-identical** while the
    detector emits nothing. A test pins exactly that — the fingerprint is blind
    to the demotion, the capability map is not. `passport` is the live instance:
    the compiled artifact records it as `enabled: false`, so any scenario keyed
    on it is now a build failure rather than a rule that never fires.
    `taxonomy_fingerprint` deliberately re-implements the algorithm from
    `src/dataset/completeness.py` because `src/scenario_engine` may not depend on
    `src.dataset`; a test pins the two byte-for-byte so the duplication cannot
    drift.
  - `src/scenario_engine/compile.py` — set-level validation (duplicate ids,
    unknown/demoted/disabled classes, static-only triggers, evidence gating
    risk level and push channel) and the compiled artifact. The artifact carries
    `rule_count` + `content_hash` so the loader can **fail closed**: a
    hand-edited or truncated artifact is rejected, where the legacy loader did
    `data.get("rules", [])` and logged `Loaded 0 rules` at INFO. It also carries
    an **inverted index** (class → scenario ids) so an edge device evaluates only
    the scenarios that touch a detected class, which is what keeps a few hundred
    scenarios inside the 5 ms rule budget.
  - `src/scenario_engine/csv_view.py` — CSV is a **view, not a format**
    (ADR-P6-02). `scenarios.view.csv` is a read-only projection with a fixed
    column order, `QUOTE_ALL`, LF endings and embedded newlines *rejected rather
    than escaped* (a quoted multi-line cell renders as broken rows in `git
    diff`). `scenarios.editable.csv` is a restricted subset a stakeholder can
    safely edit, with **prompt fields excluded** — which is what keeps Devanagari
    out of Excel's cp1252 reach — written with a BOM, matched by `scenario_id`,
    and unable to create or delete scenarios.
  - `scripts/scenarios/30_compile_scenarios.py` with `--check` (exit 0 current,
    1 validation failure, 2 drift), `configs/scenario_engine.yaml`, and the
    `compile_scenarios` DVC stage. All three outputs are `cache: false` so they
    stay git-readable at a tag. `.gitignore`'s generic `build/` rule silently
    ignored them — a `cache: false` out under an ignored directory is
    uncommittable, which is exactly how an artifact "that records what a thing
    is" goes missing from a release; the directory is now explicitly re-allowed.
  - First exemplar scenario `configs/scenarios/SC-BTH-001.yaml` (wet floor),
    carried over from `risk_rules.yaml` with its three defects fixed: a 3 s dwell
    so flicker cannot fire it, an explicit clear condition so a floor wet for 20
    minutes is one event rather than ten announcements, and escalation to the
    caregiver instead of repeating at the resident.
- Phase-6 M4: the validator suite and the alert-volume simulation gate.
  - `src/scenario_engine/validators.py` — report-level checks with ERROR/WARN
    severities, following the `--exit-zero-on-warnings` convention. The message
    lints are tied to real strings from `configs/risk_rules.yaml`, not
    hypotheticals: **no unobservable assertions** catches "the stove appears to
    be **on**" (`:54`) — hedging the verb does not make an unobservable state
    observable; **no unanswerable questions** catches "Have you taken your
    medication today?" (`:63`), since there is no ASR and the answer is never
    heard; **no startle language** catches "Please be careful" (`:31`), because
    startle is itself a fall mechanism. Plus locale completeness (a `hi` field
    with no Devanagari is an English string pasted into it), message length,
    regulated-claim verbs, confidence reachability against the detector floor,
    dwell against the Event Memory ceiling, safety-class coverage, exhaustive
    determinism, and ambiguous arbitration.
  - `src/scenario_engine/simulate.py` — replays a day of detections through the
    scenario set and projects per-scenario and whole-set alert volume, modelling
    dwell, cooldown, `max_repeats`, the daily budget, quiet hours, and event
    hysteresis. It deliberately excludes the global rate limit and queue
    eviction: those are runtime back-pressure, and counting them would let a
    scenario set be "in budget" only because the queue was discarding its alerts.
  - **The simulation surfaced something worth recording.** Modelling the three
    legacy rules against a representative Indian-kitchen day (cylinder and stove
    permanently visible, two cooking sessions, a medicine strip out all day)
    projects ~370 alerts under the *legacy* engine, but only 16 under the new
    schema — because `max_repeats` is capped at 3 and hysteresis makes a
    persisting condition one event rather than a metronome. The failure mode was
    designed out by the schema rather than left for the gate to catch. The gate
    still earns its place on the case per-scenario limits cannot see: ten
    individually-reasonable scenarios, each inside its own budget, summing well
    past the whole-set ceiling — which is how a taxonomy actually degrades, since
    nobody adds an obviously noisy scenario, they add the twentieth reasonable one.
  - `scripts/scenarios/31_validate_scenarios.py` + the `validate_scenarios` DVC
    stage, writing `data/qa_reports/scenario_validation_report.json` as a
    `cache: false` metric (git-readable at a tag, allowlisted in `.gitignore`).
  - The validators immediately flagged two prompts in the M3 exemplar scenario as
    over the 14-word bound; the messages were shortened rather than the bound
    raised.
- Phase-5: Production Dataset Engineering, Missing-Annotation Resolution &
  Dataset v1.0 — makes dataset quality the primary solution and demotes
  Phase-4 masking to a safety net. Core invariant: auto-generated labels never
  touch `labels/` (ADR-P5-01) — they live in a candidate artifact, cross a
  mandatory human CVAT round-trip, land in a verification ledger + delta labels,
  and only then overlay onto merged labels. Masking shrinks exactly as
  verification grows. **All milestone tooling (M0–M11) is implemented and
  validated; the remaining work is operational/human (see
  `docs/07_dataset_production/phase5_milestone_status.md`).**
  - Missing-annotation resolution L1–L5: L1 human annotation (P3 CVAT flow +
    verification batches) · L2 model-assisted (`src/dataset/annotation/` —
    `AutoAnnotator` ABC + registry, `yolo_world` primary backend, optional
    `grounding_dino`, MobileSAM refine pass; sha256-pinned weights; within-
    machine determinism, `--verify-determinism`) · L3 cross-dataset salvage
    (`src/dataset/cross_dataset_salvage.py` — exact-SHA256 label transplant with
    IoU≥0.9 suppression + near-dup `cross_dataset` candidates) · L4 coverage
    estimation (`coverage.py` — pure arithmetic over pinned candidates, no
    inference at report time) · L5 dataset quality report (`quality.py` —
    residual-risk quantification, verification-progress + batch throughput)
  - Verification loop: candidate artifact → verification batches
    (`cvat_package.py`, CVAT YOLO-1.1 pre-annotation zips) → human CVAT verify →
    import (`verified_import.py`, class-order + non-target byte-equality
    hard-fails, dual-annotator IAA gate) → append-only ledger (`ledger.py`,
    `cache: false`, git-tracked) → labels overlay `data/merged_verified`
    (`apply.py`) — empty ledger ⇒ byte-identical passthrough (golden regression)
  - Completeness expansion: new policy mode `trusted_list_with_ledger`
    (`completeness_policies.py`) composes source policies with ledger-verified
    trust; additive `PolicyContext.verification_ledger` field (back-compatible);
    preflight gate G9 (ledger ↔ verified_labels ↔ provenance consistency)
  - Releases as code: `src/dataset/release/{gates,manifest}.py` — gates RG1–RG10
    + MODE/WETFLOOR prerequisites (pure functions over loaded artifacts), per-
    track thresholds in `configs/release.yaml`, frozen `record_release` stage,
    v0.5.0→v0.7.0→v0.9.0→v1.0.0 ladder
  - CLI ladder `12`–`19` (`scripts/dataset/`): auto_annotate, build/import
    verification batches, apply verified labels, coverage/quality reports,
    make_release, quality_delta; `scripts/training/evaluate_model.py` (single-
    checkpoint eval + wet_floor AP50 checkpoint); `scripts/qa/validate_phase5.py`
    (M6 correctness gate) + `full_build_preflight.py` (FB1–FB6)
  - New config files `configs/annotation.yaml` + `configs/release.yaml` (no new
    keys in acquisition params — DVC hash hygiene); `configs/eval_data.yaml`
  - DVC stages: `auto_annotate`, `apply_verified_labels`, `coverage_report`,
    `dataset_quality_report` (auto-repro); `build_verification_batches`,
    `import_verified_annotations`, `record_release`, `evaluate_yolo11n` (frozen,
    human/GPU-committed — the human-loop stages declare no deps to break the
    declared-graph cycle; freshness enforced by sha256 cross-checks)
  - Vectorized-Hamming dedup for full-mode scale (`dedup.py`, ADR-P5-12) with a
    decision-equivalence property test; WIDER FACE `class_caps`; auto-derived
    Roboflow per-slug licenses for RG7
  - Local-drive DVC remote as default (`C:\dvc_remote`, off the OneDrive tree) —
    first `dvc push` executed, closing audit risk C-1; S3 kept as secondary
  - Docs: `docs/07_dataset_production/` (README, auto-annotation / verification /
    release runbooks, **twelve ADRs `adr/ADR-P5-01…12`**, milestone-status doc);
    risk register R30–R38
  - ~440 new tests (suite → 1003 passing) across unit/integration/system/
    performance, including the M6 full-loop system smoke test; coverage floor
    raised 40 → 50

- Phase-4: Missing Annotation Mitigation — masked-loss training framework
  (public datasets label only part of the 23-class taxonomy; stock BCE turns
  every unlabeled class into false background supervision — mean smoke image
  trusts just 6.25/23 classes). Strictly opt-in; disabled ⇒ byte-for-byte
  stock pipeline (golden train-kwargs regression pins this).
  - M1 completeness metadata: top-level `completeness.policies` section in
    `configs/dataset_sources.yaml` (explicit per-source semantics — never
    name-inferred; `negatives` = verified absence of ALL classes → all-ones
    mask; `custom_captures` = per finalized session manifest), pluggable
    policy-provider registry (`src/dataset/completeness_policies.py`),
    hard-fail generator + validator (`src/dataset/completeness.py`, orphan
    refs / duplicate keys / drift / unknown images all fatal), CLI
    `scripts/dataset/11_generate_completeness.py`, new DVC stage
    `generate_completeness` (split → completeness → frozen train dep),
    report triplet `data/qa_reports/completeness_report.*`
  - M2 preflight gates G1–G8 (`src/training/preflight.py` +
    `scripts/training/preflight_check.py`, exit 0/1/2): artifact
    exists/valid, taxonomy fingerprint vs live data.yaml, train/val
    coverage, self-consistency, environment + loss-surface source canary
    (`assert_ultralytics_compat`), config validity, input-hash freshness,
    strict mixing-augmentation gate (mosaic/mixup/copy_paste forbidden
    under mitigation — ADR-P4-04)
  - M3 masked BCE loss + trainer injection: `MaskedDetectionLoss`
    (v8DetectionLoss subclass; `_MaskingBCE` wrapper multiplies the
    elementwise BCE map by a per-image {0,1}^23 mask — no upstream math
    copied; box/DFL untouched), criterion attached at `on_train_start` to
    train + EMA models (model class stays stock → portable checkpoints,
    ADR-P4-02), `build_masked_trainer` factory for
    `model.train(trainer=...)`, `--mitigation on|off` CLI,
    `missing_annotation_mitigation` config section (yolo11n + yolo11s)
  - M3.5 masking-correctness gate (committed evidence:
    `data/qa_reports/phase4_mitigation/masking_validation_report.*`):
    bit-identity vs stock loss under all-ones masks, exact-zero gradients
    for masked classes, real-artifact spot-checks (coco 10/23,
    openimages 3/23, wider_face 1/23, negatives 23/23), 1-epoch mitigated +
    disabled runs — all PASS; re-runnable via
    `scripts/training/validate_masking.py` + env-gated
    `tests/system/test_training_smoke.py`
  - M4 evaluation framework (`src/training/evaluation.py`,
    `scripts/training/evaluate_mitigation.py`): per-class P/R/F1/mAP,
    confusion-matrix export, mitigated−baseline delta reports with the
    partial-annotation caveat documented
  - M5 benchmark framework (`src/training/benchmark.py`,
    `scripts/training/benchmark_mitigation.py`): baseline vs mitigated,
    repeats, process-tree peak RSS (psutil), loss-forward + mask-build
    microbenchmarks, explicit performance budgets each marked PASS/FAIL
    (verdict FAIL on any breach); executed smoke benchmark committed
    (`data/qa_reports/phase4_mitigation/benchmark_report.*`) — all budgets
    PASS (end-to-end wall-time overhead ≈0 %, loss-forward ≈0.6 ms/call
    ≈0.2 % of a training step, mask build ≈0.06 ms/batch); microbenchmark
    uses interleaved stock/masked rounds with median reduction after
    single-series timing proved unreliable on desktop hardware
  - M6 documentation: `docs/06_training_engineering/` (engineering report,
    masked-loss architecture with identity proof + compat contract,
    operational runbook keyed by gate IDs, ADR-P4-01…05), risk register
    R25–R29, README Phase-4 section
  - 153 new CI-scope tests (unit incl. torch-dependent drift canaries +
    integration against the real shipped configs; suite 412 → 565, coverage
    73.35 % → 74.99 %) plus an env-gated system smoke test; CI mypy scope
    now includes `src/training`; `psutil` promoted to an explicit
    dependency

- Pre-Phase-4 production readiness audit
  (`docs/05_audit/pre_phase4_production_readiness_audit.md`) — phase
  verification (1/2/WP3.0/3 all PASS), full findings register, CI/DVC/git
  review, Phase-4 readiness assessment, prioritized action plan; verdict:
  ✅ ready for Phase 4 (first `dvc push` remains the gate before real
  capture collection)
- Phase-3: Custom Dataset Collection & Annotation tooling
  - `src/dataset/capture/` — collection/annotation library: typed capture
    config (`configs/capture_config.yaml`), PII-free consent verification
    against a local-only registry, EXIF/GPS metadata stripping, inbox→session
    ingest (corruption/size/duplicate gates, session manifests, aggregate
    manifest rebuild), CVAT-compatible YOLO-export import with class-order
    verification (the CVAT footgun: a subset/reordered label list silently
    shifts every class ID) and session-scoped label validation, staging +
    finalize, inter-annotator agreement (greedy IoU matching, per-class
    gates incl. a `wet_floor` R24 override), and governance-target progress
    tracking (per-class counts, houses/rooms/lighting coverage, withdrawn-
    consent flags)
  - `scripts/dataset/08_ingest_capture_session.py`,
    `09_import_annotations.py`, `10_capture_progress.py` — CLIs (ingest,
    stage/compare/finalize, progress), consistent exit 0/1/2 contract
  - `src/dataset/splitting/leave_one_house_out.py` — house-level split
    strategy (all sessions of one house share a split; `holdout_houses`
    forces named houses into test — the eval-set leakage-prevention
    mechanism); public-source images without a house match degrade to
    `group_aware` behavior
  - `scripts/qa/run_full_qa.py`: eval-set overlap guard (exact SHA-256 +
    flip-robust perceptual near-duplicate against train-facing data,
    CRITICAL) and house-exclusivity check (train/eval house overlap,
    WARNING); both opportunistic (`{"available": false}` pre-Phase-3)
  - `dvc.yaml`: `ingest_custom_captures` / `ingest_eval_set` frozen stages
    — human-in-the-loop data enters `dvc.lock` via `dvc commit -f`, never
    via `dvc repro` (which would delete-then-regenerate real photos as
    empty on any machine without the capture inbox); `merge_datasets`
    gains a dependency on `data/raw/custom_captures`
  - `docs/04_dataset_engineering/capture_annotation_runbook.md` — full
    operational SOP (consent → capture → ingest → CVAT annotation → IAA →
    finalize → DVC recording → eval-set locking → wet_floor R24 pilot gate
    → Roboflow slug checklist → dataset-v1.0.0 release checklist);
    `docs/03_engineering_appendix/consent_form_template.md`;
    `data/consent/README.md`
  - Risk register: R24 (`wet_floor` taxonomy risk) added with a measurable
    gate (docs/01 `risk_register.md`)
  - 129 new unit tests + 1 end-to-end integration test simulating the full
    human workflow on synthetic data (inbox → ingest → dual-annotator CVAT
    export → IAA → finalize → merge → split → QA → eval-set overlap →
    lock)

- WP3.0 platform remediation (Phase-2 closure review follow-up)
  - `tests/unit/test_downloaders.py` + `tests/unit/test_downloaders_parsers.py` —
    40 offline unit tests for the acquisition framework (fetch_url retry/resume/
    atomicity, download() template + manifests, COCO/Open Images/WIDER FACE
    parsers, negatives selection, Roboflow skip contract, CLI exit codes);
    downloader package coverage 0% → ~93%, overall 43% → 65%
  - `.env.example` documenting `ROBOFLOW_API_KEY` (graceful-skip semantics)
- H-B: `sources.roboflow.datasets` populated with one verified Roboflow
  Universe dataset per specialty class — `obj-dect/gas-cylinder-detection`
  (gas_cylinder, 108 img, India-specific), `project-ko6pf/medicine-bottle`
  (medicine_bottle, 308), `test-agunz/wire_v3` (wire, 3,377) and
  `muhammads-workspace-5acq6/charger-lbdun` (charger, 730). All CC BY 4.0,
  each verified from its Universe page's schema.org JSON-LD. Ingestion still
  needs `ROBOFLOW_API_KEY` + a download run; the `charger-lbdun` alias
  (`"My Tugas"`) is flagged in-config as needing a visual spot-check.
  `TestRepoRoboflowDatasets` validates the entries statically (well-formed
  slug/version/license/classes, aliases resolve against `configs/data.yaml`,
  and every declared `trusted_classes` entry has a backing dataset).
- H-B acquisition executed: **1,846 images / 1,939 boxes** ingested
  (wire 1,332 · medicine_bottle 343 · gas_cylinder 158 · charger 106),
  remapped 1,939 kept / 0 dropped, DVC-committed. Corrected the config
  against the real exports first — every alias had been taken from page
  prose rather than the exported class string, and the originally-chosen
  charger project had zero downloadable versions (see Fixed). Data is in
  `data/raw/roboflow_imports` + `data/interim`; the merge into
  `data/merged` is a separate deliberate rebuild.
- `roboflow` declared as an optional extra in `pyproject.toml`
  (`pip install -e ".[roboflow]"`) — `roboflow_dl.py` imported the SDK but
  the only reference in the repo was a mypy override.

### Fixed
- Phase-6 M1: the nine runtime defects catalogued in
  `docs/08_scenario_engineering/architecture_review.md` §3. Each fix ships with
  the regression test that pins it; `src/pipeline/rule_engine.py`,
  `event_memory.py` and `detector.py` had **no tests at all** beforehand.
  - **Every feature flag was inert.** `orchestrator.py` read a `feature_flags:`
    root key that does not exist in `configs/feature_flags.yaml`, so the flag
    dict was permanently `{}` while the correct loader (`SystemConfig`) was
    reachable only from tests. The orchestrator now loads `SystemConfig` and
    routes every component through it, per that class's own documented
    contract. This makes `passport: false  # privacy`, the per-rule toggles,
    `memory_window_frames`, and `smolvlm_analysis: false` real for the first
    time — the VLM previously loaded unconditionally because the orchestrator
    consulted a `vlm_enabled` key that exists in no config file.
  - **Class gating is now enforced in the detector**, so a disabled class never
    becomes a `Detection` and therefore never reaches a rule, a log, or an
    alert — a comment in a config file is not a privacy guarantee.
  - **Recall-tuned thresholds were unreachable.** Ultralytics filters by `conf`
    before the per-class pass runs, so passing the global 0.25 silently defeated
    every lower safety threshold in `configs/class_thresholds.yaml` — which was
    itself never loaded at runtime. Prediction now runs at the loosest threshold
    any class asks for, with the per-class cut applied after.
  - **`stove_unattended`, the only CRITICAL rule, could not fire cold.**
    `frames_since_seen*` returned the 150-frame window for never-seen classes,
    saturating at exactly 10.0 s at 15 FPS, so any `absent_for` threshold above
    10 s was unreachable until the class had been seen once. Never-seen now
    means "absent for the whole session".
  - **Temporal rules used the nominal FPS.** The orchestrator now measures the
    real loop rate, so a throttled device no longer silently rescales a 30 s
    threshold to 225 s.
  - **The condition DSL was replaced with a real parser**
    (`src/pipeline/condition_parser.py`): tokenizer, recursive-descent parse to
    an immutable AST, evaluated per frame. Fixes `any_of([...])` silently
    discarding the rest of its condition, AND/OR precedence being inverted,
    parentheses being unsupported, and `NOT` composing only with `detected`.
    Conditions are parsed **once at load time**, so a malformed condition, an
    unknown predicate, an invalid severity, a duplicate id, a negative cooldown,
    or an empty rule set is now a loud failure instead of a rule that silently
    never fires. A bad hot-reload leaves the previous rule set active, and
    cooldown entries for removed rules are pruned rather than leaking forever.
    Alerts are returned severity-ordered, as `rule_engine.md:37-43` has always
    specified. Optionally validates class references against `configs/data.yaml`,
    so `detected(knive)` fails at load.
  - **`save_csv_report` wrote CRLF**, which permanently dirties the working tree
    (failing release gate RG5) and gives one report two DVC hashes across the CI
    matrix. Now writes LF, matching its JSON sibling's documented contract.
  - **`log_alert` wrote `explanation` unredacted** while `log_frame` redacted the
    same `person`/`face` geometry. Explanations are now redacted at any nesting
    depth, on word boundaries, before they reach `logs/events.jsonl`.
- Phase-6 M1: **alert arbitration now exists.** `AlertQueue` — severity-ordered,
  bounded, evicting the lowest-priority pending item on overflow — was fully
  implemented and unit-tested but never imported by anything. The orchestrator
  instead spoke `max(alerts)` each frame and discarded the rest, so a backlog
  could not survive a frame. It is now wired in, and `max_alerts_per_minute`
  (commented "Hard cap — prevents alert fatigue" in `configs/feature_flags.yaml`,
  referenced by no code) is enforced. CRITICAL alerts bypass the cap: a limit
  that can silence an emergency is a worse failure than the fatigue it prevents,
  and the alarm-fatigue evidence concerns routine chatter. Suppression affects
  speech only — every alert is still queued, logged, and counted.
- Phase-6 M1: `PiperTTS.speak` **evicts the lowest-priority queued message on
  overflow instead of dropping the incoming one.** With five routine prompts
  backed up behind a multi-second synthesis, an arriving CRITICAL was discarded
  with a log warning while the chatter still played — inverting the very
  priority contract the class documents. A message is dropped now only when
  nothing queued outranks it.
- Phase-6 M1: CI now runs `mypy src/` (whole tree, matching `Makefile:78`). The
  `src/pipeline` exclusion rested on a "17 errors" note from 2026-07-14 that had
  gone stale — it measures clean — and it was hiding the dataclasses that
  `src/pipeline/__init__.py` declares LOCKED. Local dev had been stricter than CI.
- CVAT label paste failed with `unknown label type "undefined"` on the
  deployed CVAT: `build_cvat_labels_spec` omitted `type`, assuming CVAT
  defaults it to `"any"`. It now emits `{"name": ..., "type": "rectangle",
  "attributes": []}` per class — all verification boxes are axis-aligned.
  (Follow-on to the `attributes: []` fix below; both are required.)
- `14_import_verified_batch.py` could not import a batch built before a
  merge rebuild: every non-target byte-equality check and provenance lookup
  compared against `data/merged/labels/`, where none of the batch's images
  exist any more. Added `--allow-missing-base`, which skips the non-target
  check for images with no base label (there is nothing to diff against) and
  infers provenance from the filename prefix. Scoped to genuinely missing
  images — present-base images still get the full check.
- `generate_completeness` hard-failed on any verification-ledger entry whose
  image predates the current merge snapshot, blocking `coverage_report` and
  `dataset_quality_report` behind it. Such entries are historical records,
  not inconsistencies: they are now skipped with a warning naming the count,
  the affected batch ids and example filenames. The ledger is unchanged.
- Non-target label comparison in `verified_import.py` used ambiguous `l`
  lambdas over 100 columns and a non-`strict` `zip()`, failing `ruff`/`black`
  in CI. Extracted a named `_sort_key` and made the pairing `strict=True`
  (redundant behind the existing length guard, but keeps it honest if that
  guard ever moves).
- `exports/` — where `verification_runbook.md` §5 tells reviewers to put CVAT
  exports — was not git-ignored, so RG5's clean-tree gate
  (`rg5_working_tree_tagged`, which fails on *any* porcelain output including
  untracked paths) could never pass once a batch had been reviewed.
- CVAT task creation failed with `Could not create the task` / `labels:
  [object Object]` and `POST /api/tasks 400` on CVAT 2.5.14:
  `build_cvat_labels_spec` (`cvat_package.py`) emitted bare `{"name": ...}`
  labels, but CVAT's Raw label editor (`validateParsedLabel` in `cvat-ui`
  `labels-editor/common.ts`) requires each label to carry an `attributes`
  array (`"Attributes must be an array"`). The spec now emits
  `{"name": ..., "attributes": []}` per class (taxonomy id order preserved).
  Documented the schema, the pinned CVAT version, and the
  images-as-data-vs-`preannotations.zip`-as-annotations distinction (the
  "1 image instead of 80" symptom) in `verification_runbook.md`.
- Stale DVC pipeline state (audit H-2): `dvc repro` re-run with Phase-3 code
  refreshed `dvc.lock` (merge/split/QA stages) and locked the new
  `generate_completeness` stage; regenerated QA metric verified sane
  (188 images, 0 critical, 18 pre-existing warnings)
- Local mypy gate aborted on venvs with numpy 2.x installed (PEP 695 `type`
  statements in numpy stubs vs the hard `python_version = "3.10"` pin);
  pin removed — mypy now checks under the running interpreter while CI's
  Python-3.10 quality job keeps enforcing the 3.10 floor. Supersedes the
  ineffective `numpy.*` ignore-missing-imports override attempt.
- `AlertQueue` heap comparison `TypeError` on coarse-resolution clocks
  (equal-severity alerts with identical `time.monotonic()` timestamps fell
  through to non-orderable `Alert` objects, failing windows-latest CI);
  strictly increasing sequence-number tiebreaker added
  (`src/pipeline/alert_queue.py`)
- `requests` added to runtime dependencies (`requirements.txt`,
  `pyproject.toml`) — downloader tests import it transitively and fresh
  environments failed collection
- Roboflow cross-dataset image budget decremented by *distinct class count*
  instead of images copied (`_consolidate_export` now returns the copied
  count; regression-tested)
- QA reports no longer embed absolute machine paths: `data_dir` and issue
  file paths are written cwd-relative with posix separators
  (`portable_path` in `scripts/qa/check_annotations.py`)
- Machine-specific Windows cache path removed from the tracked
  `.dvc/config`; per-machine relocation now documented via
  `dvc cache dir --local` (docs/04 §6)
- `generate_splits.py` docstring falsely claimed to be the DVC stage entry
  point (the stage runs `split_dataset.py`); clarified as a convenience
  wrapper
- 4 mypy errors in `src/logging/structured_logger.py` /
  `src/config/config_loader.py`; `psutil` added to stub overrides

### Changed
- Acquisition budgets: `class_caps` now bounds accumulation, not just selection,
  and Open Images gained cap support it never had. Both downloaders share one
  rule (`base.is_capped_out`): fetch an image only when **every** class in it is
  under budget, checked before the network call.
  Found while launching the `mode: full` build. The old rule fetched on *any*
  class being under cap and then credited the whole label file, so a saturated
  class rode along on other classes' recruits — an 800-box `person` cap yielded
  **36,469** person boxes on COCO train2017, against a config comment reading
  "prevent imbalance". Capping only some classes had the same effect, since
  uncapped ones never saturate and so never gate.
  Measured on the real annotation indexes, offline, before fetching anything:
  COCO 26,409 imgs / 135,098 boxes / ~11 h → **5,300 / 11,737 / ~2.2 h** at 1200
  per class, every class at cap; Open Images 14,562 / 21,135 / ~6 h →
  **2,021 / 2,953 / ~0.8 h**. Open Images is 92% `Door`, so only `Door` is
  budgeted; `Gas stove` (taxonomy `stove`, the scarcest safety class at 267
  boxes) is left effectively uncapped and contributes 520.
  WIDER FACE's existing single-class cap was verified correct and unchanged.
- Release ladder: new `dataset-v0.6.0` track in `configs/release.yaml`
  (`mode: full`, gates RG1–RG8) describing the local-capture build — ADR-P5-13.
  It adds RG8 (zero split leakage) over `v0.5.0` and claims **neither** RG9
  (custom-capture targets) nor RG10 (A/B + locked-eval evidence), because
  `data/raw/custom_captures/manifests` is empty and no training run exists.
  **`dataset-v1.0.0`'s gate list is unchanged** — RG9/RG10 stay mandatory there,
  now pinned by a test rather than a comment. Exists so Phase-F release-candidate
  validation is exercised on a real release instead of being debugged during the
  v1.0 cut. Also: `record_release.cmd` in `dvc.yaml` was still pinned to
  `dataset-v0.5.0`, and `configs/release.yaml`'s header claimed to be a DVC
  params file when `dvc.yaml` never referenced it — both corrected.
- DVC: two remotes in `.dvc/config` — `localstore` (`C:\dvc_remote`) is the
  **default** so a bare `dvc push` cannot incur S3 transfer by accident, and
  `storage` is the off-site S3 copy at
  `s3://elderly-assistant-mlops-329117470647-ap-south-1-an/datasets/yolo_v1`
  (`ap-south-1`), used explicitly via `dvc push -r storage`. No `profile` key is
  set — credentials resolve through the standard AWS chain, so pinning a profile
  name cannot break other machines or CI. Runbook in docs/04 §6; `dvc` dependency
  installs the S3 extra (`dvc[s3]`).
  **Activated 2026-07-27, closing risk C-1 (single-copy dataset):** 46,337 objects
  / 6.21 GB uploaded, `dvc status -c -r storage` in sync, bucket has SSE-AES256
  and versioning enabled.
- CI: test matrix expanded to ubuntu+windows × py3.10/3.12; coverage gate
  enabled (`fail_under = 40`, ratchet-only); mypy widened to
  `src/dataset src/utils src/config src/logging` (`src/pipeline` joins in
  Phase-6); dev tooling installs use requirements.txt-matching bounds
- `run_workflow.sh` now wraps `dvc repro` — the DVC DAG is the single
  orchestration path (previously drove a divergent script chain plus
  webcam inference)

- Stage 2: Dataset Collection & Dataset Engineering (Phase-2)
  - `src/dataset/` — dataset engineering library: provenance manifests
    (source / capture-session / merged), acquisition config loader with
    smoke/full mode + license gate, class remapping (copy & in-place modes),
    indoor/quality filters, flip-robust perceptual dedup, multi-source merge
    with lineage, negative selection, split-strategy registry
    (`group_aware`, `stratified_group`; `kfold`/`leave_one_house_out` reserved)
  - `src/dataset/downloaders/` — bespoke annotations-first downloaders for
    COCO 2017, Open Images V7, WIDER FACE (license-gated), negatives, plus a
    Roboflow Universe SDK integration (graceful skip without API key)
  - `scripts/dataset/01–07` acquisition/processing CLIs matching `dvc.yaml`
  - `scripts/qa/run_full_qa.py` — QA orchestrator: structural checks + stats
    + license gate + label-completeness + blur/low-light checks (risk R01),
    all merged into the DVC metric `data/qa_reports/annotation_qa_report.json`
  - `configs/dataset_sources.yaml` — acquisition config, doubles as DVC params;
    `configs/dataset_split_config.yaml` now actually read by the split CLIs
  - DVC initialized (cache outside OneDrive), truthful `dvc.yaml` DAG
    (download → remap → merge → split → QA; training stage frozen for Phase-5)
  - `docs/04_dataset_engineering/` — license register, label-completeness
    policy, DPDP/PII notes, split governance, Phase-2 descope statement
  - `tests/integration/test_dataset_pipeline.py` — first offline end-to-end
    pipeline test; ~70 new unit tests (296 total assertions across 241+ tests)

### Changed
- Smoke dataset validated end-to-end: 188 images / 4 sources through
  `dvc repro` with QA zero critical issues (tag `dataset-v0.1.0-smoke`)
- CI: unit/integration tests now blocking; mypy gates `src/dataset`
- Repo-wide lint cleanup (60+ pre-existing ruff violations fixed);
  Windows cp1252 console crashes fixed; `.gitkeep` no longer counted as
  split leakage; `PipelineMetrics`-unrelated runtime defects logged for
  Phase-6 (see docs/04 §7 and the Phase-2 plan)

- Stage 1: Repository foundation and project skeleton
  - Production-ready folder structure
  - `pyproject.toml` with Black, Ruff, MyPy, Pytest configuration
  - `requirements.txt` with all V1 dependencies
  - `Makefile` with development workflow targets
  - `configs/` — YAML configuration stubs for data, training, deployment, rules, feature flags
  - `src/pipeline/__init__.py` — Locked data contracts (Detection, Alert, SceneContext, etc.)
  - `src/pipeline/` — Module stubs for all pipeline components
  - `src/config/config_loader.py` — Configuration system stub
  - `src/logging/structured_logger.py` — Logging system stub
  - `tests/` — Complete test directory structure (unit / integration / system / performance)
  - `dvc.yaml` — DVC pipeline definition stub
  - `docs/` — Full technical documentation (3-document structure)

---

## [0.1.0] - Superseded

This entry originally described a Stage-1-only "skeleton release, no application
logic implemented." That is no longer accurate — Stage 1 (repository foundation)
and Stage 2 / Phase-2 (dataset engineering platform) are both complete and
documented in the `[Unreleased]` section above. No `0.1.0` tag has actually been
cut; the dataset-specific milestone is tracked instead via the `dataset-v0.1.0-smoke`
git tag. This stub is kept only for changelog continuity and should not be read
as a current status statement.

---

*Future versions will be documented here as each stage is completed.*
