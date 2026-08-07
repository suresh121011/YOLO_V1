# Phase-6 Architecture Decision Records (ADR-P6)

Records the load-bearing design decisions for the Scenario & Rule Engine knowledge layer. Each ADR
follows the Phase-4/Phase-5 format (Status / Deciders / Context / Decision / Alternatives considered /
Consequences) and is referenced from the code, configs, and `dvc.yaml` entries that implement it.

ADR-P6-01 through ADR-P6-09 were ratified together on 2026-08-07 by a three-lens cold-read council
(architecture & integration · knowledge representation & data governance · safety, clinical validity &
human factors). `llm-council-skill` remains uninstalled in this environment; the multi-agent cold read is
the established substitute, as in Phases 4 and 5. Every repo claim the council made was independently
verified against the code before being adopted — see `../architecture_review.md` §3.

| ADR | Decision |
|:---|:---|
| [ADR-P6-01](ADR-P6-01-per-scenario-yaml-master.md) | One YAML file per scenario in `configs/scenarios/`; knowledge is source, not DVC data |
| [ADR-P6-02](ADR-P6-02-csv-is-a-view.md) | CSV is an export-only view; lossless round-trip is not attempted |
| [ADR-P6-03](ADR-P6-03-structured-ast-over-string-dsl.md) | Triggers are structured predicate trees compiled to an AST; the string DSL survives only as a rendering |
| [ADR-P6-04](ADR-P6-04-rule-engine-injection.md) | Integrate by `BaseRuleEngine` constructor injection, not the `AnalysisPlugin` seam |
| [ADR-P6-05](ADR-P6-05-class-capability-map.md) | The compiled artifact carries a class capability map; the taxonomy fingerprint alone cannot catch a class demotion |
| [ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md) | `rule_hash` over semantic fields only; derived-semver validator with a failing gate |
| [ADR-P6-07](ADR-P6-07-near-units-and-room-vs-zone.md) | `near()` is frame-normalized screen space; `room` is config, `zone` is deferred |
| [ADR-P6-08](ADR-P6-08-rejected-scenario-negative-register.md) | Rejected scenarios stay in-repo as versioned rows with a `rejection_reason` |
| [ADR-P6-09](ADR-P6-09-alert-contract-extension.md) | Extend the LOCKED `Alert` additively rather than returning a parallel outcome type |
