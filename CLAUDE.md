# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`kd1-anime` is a CLI tool that uses multiple LLM agents to automatically generate Manim (Community Edition) math animations and render them on a Slurm HPC cluster. Users describe an animation in natural language; the pipeline plans scenes, generates Python Manim code, reviews it, submits render jobs to Slurm, auto-fixes failures, and merges the resulting MP4 fragments into a final video.

**Core principle**: No heavy agent frameworks (no LangChain, AutoGen, etc.). Uses only raw `openai` library + Pydantic for structured output + a hand-written FSM for orchestration.

## Commands

### Setup

```bash
# Install system dependencies (Manim, FFmpeg, TeX Live) on a HPC node
bash install.sh

# Python dependencies (in the manim_env conda environment)
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env   # then edit .env with your LLM API key and Slurm settings
```

All configuration lives in `config.py` (pydantic-settings, reads from `.env` / env vars).

### Running

```bash
# Interactive TUI mode with requirement clarification (default)
python main.py
python main.py chat

# Direct generation (no clarification phase)
python main.py generate "Animate the derivation of the quadratic formula"

# Dry-run: generate scene plans and code only, skip Slurm rendering
python main.py generate "..." --dry-run

# Scene planning only
python main.py plan "Explain eigenvalues visually"

# CLI options for generate: --api-key / -k, --model / -m, --output / -o,
#   --partition / -p, --max-fix, --dry-run
```

There are no tests yet in this project.

## Architecture

### State machine pipeline (`orchestrator.py`)

The `Orchestrator` class drives a finite state machine through these phases:

```
INIT → PLANNING → CODING → REVIEWING → DISPATCHING → MONITORING → (FIXING → REVIEWING → ...) → MERGING → DONE
```

- **Planner** (`agents/planner.py`): Decomposes a natural-language prompt into a `list[ScenePlan]` (JSON structured output via Pydantic). The system prompt includes 6 narrative modes, 3b1b visual principles, and a Manim element cheatsheet.
- **Coder** (`agents/coder.py`): Generates Manim Python code for each `ScenePlan`. The system prompt embeds an extensive Manim CE API knowledge base (classes, animations, transforms, updaters, 3D, common pitfalls) sourced from `adithya-s-k/manim_skill`.
- **Reviewer** (`agents/reviewer.py`): Reviews generated code against a 28-item checklist (version/imports, class structure, deprecated APIs, LaTeX, animation logic, layout). If invalid, the feedback is fed back to the Coder for a rewrite (max `MAX_REVIEW_ROUNDS` loops).
- **Slurm Dispatcher** (`cluster/slurm.py`): Generates sbatch scripts via string concatenation (no Jinja2 dependency), submits with `sbatch`, polls with `squeue`/`sacct`. Handles GPU config, conda env activation, and transient scheduler errors with retry.
- **Monitor + Auto-Fix** (in orchestrator + `agents/auto_fixer.py`): Polls job status; on failure, reads the last `LOG_TAIL_LINES` of the stderr log, classifies the error type, and sends both the failing code and log to the AutoFixer agent. Fixed code re-enters the REVIEWING → DISPATCHING → MONITORING loop (max `MAX_FIX_ATTEMPTS` per scene).
- **Video Merger** (`media/merger.py`): Collects rendered MP4s from the nested Manim output directories (`videos/<ClassName>/1080p60/`), generates a `filelist.txt`, runs `ffmpeg concat` (stream copy preferred, falls back to re-encode).

### LLM interaction (`agents/base.py`)

`BaseAgent` wraps the `openai` library's `chat.completions.create`:

- **Exponential backoff retry** on `RateLimitError`, `APITimeoutError`, and general `APIError`.
- **`BadRequestError` handling**: If caused by `response_format`, falls back to prompt-only JSON mode instead of giving up.
- **Structured output**: `call_llm_json()` and `call_llm_json_list()` parse LLM responses into Pydantic models with robust JSON extraction (handles markdown fences, prose-wrapped JSON, truncated output).
- **Streaming**: `call_llm(stream=True)` prints tokens as they arrive (used by the TUI clarifier to avoid UI freezes on long calls).
- **Code extraction**: `_extract_code_block()` handles missing closing fences (model truncation at `max_tokens`).
- The OpenAI client is lazily constructed so that missing API keys produce a friendly error before any LLM call, not at import time.

### Interactive TUI (`tui.py`)

The default `python main.py` mode:

1. Shows a banner, prompts for initial description.
2. A `Clarifier` agent conducts a multi-turn Q&A (uses streaming via `BaseAgent.call_llm`) to refine requirements. Ends when the LLM outputs `{"READY": true, "prompt": "..."}` or `MAX_CLARIFY_ROUNDS` is reached.
3. User confirms the refined prompt, then the `Orchestrator.run()` is called with a callback that renders progress with Rich tables and status lines.

### Configuration (`config.py`)

`Settings` (pydantic-settings) has four groups:

- **LLM API**: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_SEND_MAX_TOKENS`
- **Slurm**: `SLURM_PARTITION`, `SLURM_QOS`, `SLURM_CONDA_ENV`, `SLURM_TIME_LIMIT`, `SLURM_CPUS_PER_TASK`, `SLURM_MEM_GB`, `SLURM_GPU_TYPE`, `SLURM_GPU_COUNT`
- **Paths**: `WORKSPACE_DIR`, `SCENES_DIR`, `LOGS_DIR`, `VIDEOS_DIR`, `OUTPUT_FILE`
- **Agent tuning**: `MAX_REVIEW_ROUNDS`, `MAX_FIX_ATTEMPTS`, `MAX_CLARIFY_ROUNDS`, `MONITOR_POLL_INTERVAL`, `MONITOR_TIMEOUT`, `MONITOR_MAX_UNKNOWN`, `LOG_TAIL_LINES`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_DELAY`

The config is a global singleton (`settings = Settings()`) — import with `from config import settings`.

## Key design decisions

- **No Jinja2 in production**: The Slurm script builder uses string concatenation to avoid an extra dependency. The ARCHITECTURE.md originally specified Jinja2 templates; the implementation chose not to.
- **Rich console output everywhere**: All agents and modules use `rich.console.Console()` for styled logging. The TUI uses `rich.panel.Panel`, `rich.table.Table`, and `rich.prompt.Prompt`.
- **Lazy OpenAI client**: `BaseAgent.client` is a `@property` that constructs the client on first access, so API key errors surface at call time rather than import time.
- **Robust output parsing**: `_extract_json()` uses bracket-pairing with string-awareness to handle prose-wrapped or fence-wrapped JSON. `_extract_code_block()` handles truncated code (missing closing fence).
- **Single-scene failure isolation**: If one scene fails code generation, review, or rendering, the orchestrator marks it as failed/given-up and continues with the remaining scenes. Only scenes that actually rendered are included in the final merge.
- **Callback pattern for TUI**: `Orchestrator.run()` accepts an optional `callback(event, data)` that the TUI uses to render progress without coupling the core logic to the UI.
