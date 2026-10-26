# AGENTS.md

Repository guidance for coding agents working on `kd1-anime`.

## Project summary

`kd1-anime` is a Python CLI/TUI that turns a natural-language request into Manim Community Edition scenes, reviews generated code, renders scenes as independent Slurm jobs, repairs render failures, and merges successful scene videos with FFmpeg.

Do not introduce LangChain, AutoGen, LangGraph, or another agent framework. The project intentionally uses the OpenAI-compatible SDK, Pydantic models, and an explicit finite-state machine.

## Important commands

```bash
# Install editable package and development tools
python -m pip install -e '.[dev]'

# Quality gates
ruff check .
ruff format --check .
python -m compileall -q .
bash -n install.sh
pytest -q
python -m build --sdist --wheel

# CLI
python main.py
python main.py generate "..." --dry-run
python main.py plan "..."
python main.py render scene.py --class MyScene --wait
python main.py version
```

Do not run a real LLM request, submit Slurm jobs, or execute generated code during ordinary unit tests.

## Configuration

Settings are defined in `src/kd1_anime/config.py` and loaded in this order:

```text
when config.toml exists: process environment > ~/.kd1-anime/config.toml > defaults;
otherwise: process environment > ./.env > ~/.kd1-anime/.env > defaults
```

Never commit `config.toml`/`.env` or print API keys. The API is provider-neutral:
`LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` must work with any OpenAI-compatible
endpoint. Visual evaluation uses the separate `VISUAL_LLM_*` profile and must never
silently inherit the main endpoint.

## Pipeline

```text
INIT → (主 LLM/RAG 可用性检查) → PLANNING → DETAILING → PLAN_REVIEWING → CODING → REVIEWING
     → DISPATCHING → MONITORING → (FIXING → REVIEWING)
     → VISUAL_EVALUATING → MERGING → EVALUATING → DONE
```

Key invariants:

1. Every run uses a unique `~/.kd1-anime/workspace/runs/<run-id>/` directory by default.
2. Every generated file contains exactly one supported Scene class.
3. All generated or auto-fixed code passes `validate_manim_code()` before submission.
4. Any code change must be reviewed again.
5. Review attempts, validation attempts, and render-fix attempts are bounded.
6. Cairo jobs never request a GPU; OpenGL jobs require a configured GPU type.
7. Scheduler timeout/unknown failures must cancel the remote Slurm job.
8. Video merge inputs must come from the exact current `RenderJob`, never a shared directory scan.
9. Partial output is rejected unless `ALLOW_PARTIAL_OUTPUT=true`.
10. Parallel workers must not stream to or read from shared stdin.
11. Slurm directive values must pass validated single-line schemas; never concatenate unchecked CLI input.
12. A failed `scancel` must never trigger automatic resubmission of the same scene.
13. External files passed to `render` must be copied into the private run directory before submission.
14. Run directories are private (`0700`), and prompt/generated source files are `0600`.
15. FFmpeg writes to a temporary file and atomically replaces the final output only after success.
16. Every FSM transition and successful Slurm submission must be checkpointed atomically in `manifest.json`.
17. Resume must verify generated-code hashes and hold the per-run lock before reusing any Slurm Job ID.
18. Visual reports and repair candidates must be bound to exact code, inherited-context, frame, and video hashes.
19. A visual endpoint failure is `unknown`, never a fabricated low score or a reason to delete a valid render.
20. Main/visual requests, probes, retries and compatibility fallbacks reserve and checkpoint their budget before dispatch; unknown outcomes still count.
21. Resume/retry preserve request limits, reservations and retry counters. Incremental runs have independent budgets.
22. Manifest v10 is executable; v4-v9 are read-only and must be rejected before probes, job access, run locking or file changes.
23. Commit worker results only when their scene revision and complete input digest match; invalidate consumers by teaching, handoff or source dependencies.
24. Persist a unique submission attempt before external submission and the returned Job ID immediately afterward. Uncertain submission or failed cancellation prohibits automatic resubmission.
25. Slurm, local rendering and smoke use the same environment whitelist and container policy. Never adopt a local process by historical PID.
26. All source/report objects are immutable and content addressed; display aliases cannot authorize execution or recovery.
27. Stage drafts merge only through `CommitCommand`; preserve transport reservations during concurrent commits.
28. Pin non-sensitive run settings and execution identity; missing or mismatched actual environment forbids reuse.
29. `SCENE_CODEGEN_MODE=llm` remains the default; v2 compilation requires complete inputs and falls back explicitly when unsupported.

## Module map

- `src/kd1_anime/cli.py`: Typer commands and installed package entry point.
- `main.py`: thin source-tree compatibility launcher.
- `src/kd1_anime/tui.py`: Rich + prompt_toolkit clarification UI and callback rendering.
- `src/kd1_anime/config.py`: validated pydantic-settings configuration.
- `src/kd1_anime/orchestrator.py`: FSM, per-run paths, scene state, parallel LLM stages.
- `src/kd1_anime/run_store.py`: v10 manifests, read-only legacy formats, atomic checkpoints, code hashes and run locks.
- `src/kd1_anime/domain/`: planning, technical, review, execution, dependency, boundary and program contracts; old Agent paths re-export them.
- `src/kd1_anime/commits.py`, `stage_transactions.py`, `objects.py`: isolated stage proposals, durable state commit and immutable objects.
- `src/kd1_anime/control.py`, `scheduling.py`, `clients.py`: cancellation/deadlines, bounded workers and run-owned clients.
- `src/kd1_anime/request_budget.py`: durable request reservations and request-scoped usage diagnostics.
- `src/kd1_anime/services/`: explicit planning/generation/review/recovery/visual inputs and results, dependency calculations and v2 compiler.
- `src/kd1_anime/events.py`, `telemetry.py`: TUI callback adapter and correlated run/scene/stage spans.
- `src/kd1_anime/agents/base.py`: OpenAI-compatible calls, retries, streaming, JSON/code parsing.
- `src/kd1_anime/agents/planner.py`: outline and detailed scene planning prompts/calls, with compatible model exports.
- `src/kd1_anime/agents/plan_reviewer.py`: pre-code mathematical and feasibility review of ScenePlan.
- `src/kd1_anime/agents/coder.py`: ManimCE code generation and rewrite prompt.
- `src/kd1_anime/agents/reviewer.py`: closed structured review contract and checklist.
- `src/kd1_anime/agents/validator.py`: deterministic AST and Scene-structure checks.
- `src/kd1_anime/agents/auto_fixer.py`: render-log-driven repair.
- `src/kd1_anime/eval/`: deterministic metrics, keyframe sampling, and independent multimodal visual evaluation.
- `src/kd1_anime/cluster/slurm.py`: sbatch script generation, submission, batch polling, cancellation.
- `src/kd1_anime/cluster/jobs.py`, `monitoring.py`, `render_backend.py`: neutral render jobs, shared monitoring and independent local execution.
- `src/kd1_anime/cluster/execution.py`: shared environment whitelist, resource execution policy and container isolation.
- `src/kd1_anime/api_catalog.py`, `manim_catalog_0201.py`: exact-version public API metadata, independent of semantic RAG.
- `src/kd1_anime/eval/benchmark.py`: 24 fixed tasks, offline compiler checks and evidence-aware runtime comparisons.
- `src/kd1_anime/media/merger.py`: exact inputs and FFmpeg xfade/acrossfade atomic merge.
- `install.sh`: no-sudo Ubuntu/HPC environment installer.
- `tests/`: deterministic tests with LLM/Slurm/FFmpeg behavior mocked or isolated.

## Coding conventions

- Support Python 3.10+.
- Use `pathlib.Path` for paths and `subprocess.run([...], shell=False)` for commands.
- Quote values embedded into sbatch shell scripts with `shlex.quote` or `shlex.join`.
- Keep Pydantic output schemas closed with `extra="forbid"` and constrained literals/ranges.
- Keep `Settings` assignment validation enabled so CLI overrides cannot bypass field validators.
- Use `default_factory` for mutable model/dataclass fields.
- Do not weaken AST validation merely to accept a model hallucination; feed the deterministic error back to the Coder instead.
- Keep TUI display logic behind orchestrator callback events.
- In concurrent stages, instantiate independent Agent objects per worker and set `stream=False`.
- Add a regression test for every state-machine, parser, path-selection, or timeout bug.

## Installation constraints

`install.sh` must remain non-interactive and sudo-free by default (the model configuration wizard is opt-in in non-TTY environments). It should:

- try the required Python/Miniconda modules;
- create/reuse `manim_env`;
- install Manim, FFmpeg, CJK fonts, and only the TeX packages needed by Manim;
- prefer a complete existing XeLaTeX from PATH or common system/user TeX Live roots without mutating it;
- install TeX Live under `~/texlive/<release>` using the USTC CTAN mirror only as a no-sudo fallback;
- install only the XeLaTeX/Manim/CJK packages required for `.xdv`, `ctex`, `xeCJK`, and `fontspec`;
- use a release-matched historic repository only as fallback;
- work when downloaded alone;
- install remote source through a temporary GitHub ZIP, never clone the repository into cwd or `$HOME`;
- preserve an existing user config instead of overwriting it.
- support x86_64 and aarch64 TeX Live platforms, and protect user `config.toml`/`.env` with mode `0600`.

The wheel contains only Python runtime code. Keep host/environment provisioning in `install.sh` and documentation.

## Security

Generated Python is untrusted. AST validation is defense in depth, not a sandbox. Preserve the Apptainer path (`--containall --cleanenv --no-home`, current-run bind only, optional `--nv`) and the `SLURM_REQUIRE_CONTAINER` fail-closed option.
