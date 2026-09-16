# Parallel Scene Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make independent scene generation concurrent while moving cross-scene continuity into versioned technical handoffs.

**Architecture:** Plan reviews use an immutable all-scene snapshot and independent reviewer workers. Technical design propagates a structured `TechnicalHandoff` from predecessor to successor; generated source code is no longer required to start the successor's technical design. Once each scene has a current TechnicalSpec, its Code→Code Review loop runs in its own worker, while ordered manifest/ledger publication remains serialized. Runs that lack the new handoff contract retain the existing safe serial path.

**Tech Stack:** Python 3.10+, Pydantic, explicit FSM, `ThreadPoolExecutor`, existing LLM semaphore, deterministic lifecycle/export validators.

## Global Constraints

- Do not introduce LangChain, AutoGen, LangGraph, or another agent framework.
- Preserve deterministic AST, API, continuity, lifecycle, and capability validation.
- Do not submit real Slurm jobs or call real LLMs in ordinary tests.
- Keep old manifests and test doubles usable; fall back to the serial code barrier when a predecessor has no structured technical handoff.
- Keep `evaluation-guidelines.md` untracked and unmodified.
- A scene may publish its manifest/state-ledger entry only in Scene ID order.

---

### Task 1: Add structured technical handoffs

**Files:**
- Modify: `src/kd1_anime/agents/technical_planner.py`
- Modify: `src/kd1_anime/orchestrator.py`
- Modify: `src/kd1_anime/agents/coder.py`
- Test: `tests/test_technical_planner.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Add `TechnicalHandoff` with `source_scene_id` and exported `TechnicalObject` snapshots.
- Add optional `handoff_in` and `handoff_out` fields to `TechnicalSpec`.
- Add `build_technical_handoff(spec) -> TechnicalHandoff`.
- Extend `TechnicalPlannerAgent.plan(..., previous_technical_handoff=...)`.
- Keep `TechnicalSpec.contract_version == 2` and all old constructor call sites valid.

- [ ] Write tests proving a handoff contains only `export_element_ids`, is serializable, and old `TechnicalSpec(...)` construction still works.
- [ ] Add the models and deterministic handoff builder.
- [ ] Pass the predecessor handoff into the Technical Planner prompt and attach normalized `handoff_in/out` after generation.
- [ ] Add an explicit Coder prompt section explaining that `handoff_in` is the continuity source; source code from another Scene is optional legacy context, not a dependency.
- [ ] Run the focused technical/lifecycle/orchestrator tests and commit.

### Task 2: Parallelize initial Plan Review

**Files:**
- Modify: `src/kd1_anime/orchestrator.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Add a private `_run_plan_review_parallel(ctx, active_states) -> dict[int, PlanReviewResult]`.
- Preserve `_run_plan_review_batch` as a compatibility hook; use independent workers for batches with multiple scenes.

- [ ] Add a test with two blocking-independent review doubles and a barrier that proves both reviews overlap.
- [ ] Snapshot all plans/deterministic findings before submitting workers; instantiate one `PlanReviewerAgent` per worker and use `_llm_sem`.
- [ ] Return only successful scene results so existing per-scene fallback handles worker exceptions.
- [ ] Verify plan re-planning and mechanical handoff repair remain coordinator-owned and serialized.
- [ ] Run plan-review tests and commit.

### Task 3: Prepare technical designs from handoffs

**Files:**
- Modify: `src/kd1_anime/orchestrator.py`
- Modify: `src/kd1_anime/run_store.py` only if a persisted field is required
- Test: `tests/test_scheduler.py`
- Test: `tests/test_run_store.py`

**Interfaces:**
- Add a helper that obtains the predecessor's filtered `TechnicalHandoff` for the current plan.
- Make `_technical_input_hash` include the handoff digest and use the old code/manifest inputs only for legacy serial fallback.

- [ ] Test that Scene N TechnicalSpec receives Scene N-1's handoff without requiring Scene N-1 `code` or `exported_elements_code`.
- [ ] Make `_ensure_technical_spec` accept the optional predecessor handoff, invalidate cached specs when the handoff changes, and persist the resulting spec artifact/hash.
- [ ] Keep the old `_prepare_inherited_context` path when no structured handoff is available.
- [ ] Run scheduler/run-store tests and commit.

### Task 4: Parallelize Code→Code Review with ordered publication

**Files:**
- Modify: `src/kd1_anime/orchestrator.py`
- Modify: `src/kd1_anime/agents/coder.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Add a dependency-aware parallel path inside `_run_code_review_barrier`.
- Add a `defer_continuity_commit` option to the internal review application path.
- Add an ordered finalizer that refreshes exports, `ElementManifest`, `StateLedger`, and incremental reuse in Scene ID order.

- [ ] Add a test showing two scenes with technical handoffs enter Code concurrently and both complete Code Review.
- [ ] Prepare all technical specs first; use the structured handoff path for new contracts and fall back to the current serial barrier for legacy/test contracts.
- [ ] Run each scene's code/review loop in an independent worker with the existing LLM semaphore.
- [ ] Defer shared manifest/ledger publication until workers finish, then publish in order and checkpoint atomically.
- [ ] Ensure a worker failure marks only its Scene and prevents rendering until the barrier returns.
- [ ] Run the full scheduler/orchestrator tests and commit.

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Test: `tests/`

- [ ] Document the dependency-aware pipeline, technical handoffs, parallel Plan Review, and ordered publication.
- [ ] Add regression coverage for legacy serial fallback and handoff invalidation.
- [ ] Run Ruff, format check, compileall, `bash -n install.sh`, full pytest, and wheel build.
- [ ] Commit documentation and final test changes.

