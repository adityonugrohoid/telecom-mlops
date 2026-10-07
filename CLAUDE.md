# CLAUDE.md - telecom-mlops

Six telecom ML use cases on one shared MLOps pipeline with a simulated
daily drift loop. `README.md` is the human overview.

## Rules for this repo (owner rulings, 2026-09-26)

1. Every use case owns its synthetic data generator, schema, features,
   model and baseline. No use case imports another's code. Tuning one use
   case must never move another's results.
2. `packages/core` is the pipeline and tooling only: validate, drift,
   evaluate, registry, report, CLI. It never generates or holds data.
3. Generators follow `docs/generators.md`; a generator change needs a PR
   that states which rule it touches.
4. Run state lives in `state/` and is never committed. Results are
   committed only as a chosen summary in `results/`.
5. Personal repo: no employer or client name, branding or detail. Data,
   scenarios and reports use invented names only.
6. Every published result says the data and environments are simulated.
7. Ported code names its source repo and commit in the PR body. No commit
   subject says "house template" or "standardize".
