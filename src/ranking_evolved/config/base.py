"""Top-level Config dataclasses.

We use plain dataclasses (no Pydantic). Validation happens at
construction time via type defaults; the loader catches typos in keys
before instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..prompts.sampler import PromptConfig
from ..search.map_elites_islands import MapElitesIslandsConfig
from .objective import ObjectiveConfig


@dataclass
class TaskConfig:
    seed: str
    evaluator: str


@dataclass
class EvolutionConfig:
    max_iterations: int = 200
    random_seed: int = 42
    capture_replay: bool = False
    # Sample replay/step_NNNN.json snapshots every N iterations to bound disk
    # usage on long runs. Always snapshots: step 1, any new global best, and
    # the final step (regardless of N). Default 1 = every step (legacy
    # behavior; ~10 MB per 50 iterations).
    capture_replay_every: int = 1


@dataclass
class ProposerConfig:
    kind: str = "fake"  # openai_responses | anthropic | claude_code | codex | fake | ensemble
    models: list[dict] = field(default_factory=list)
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: int = 180
    retries: int = 3
    # Reasoning-effort knob for openai_responses (GPT-5 / o-series).
    # Default `None` means "omit from the request body". Set to one of
    # {"none","minimal","low","medium","high","xhigh"} to opt in.
    reasoning_effort: str | None = None
    # Per-iteration retry policy for unusable candidates. Diff failures and
    # evaluator crashes both burn LLM tokens but don't admit a program; this
    # cap lets the controller request up to N fresh proposals per iteration
    # (with the prior attempts' failure modes fed back into the prompt) so a
    # single noisy LLM sample doesn't waste a slot. Retries do NOT increment
    # the iteration counter — the iteration is "owed" one usable candidate.
    # Set to 1 to disable retries (legacy behavior).
    candidate_retries: int = 3
    # Fake-proposer specific:
    transcript_path: str | None = None  # JSONL file of {"raw_response": str, "model": str}
    # Scripted-proposer specific:
    proposals_jsonl: str | None = (
        None  # JSONL file of {"score": float, "complexity": float, "diversity": float}
    )


@dataclass
class EvaluationConfig:
    timeout: float = 1800.0
    isolation: str = "inline"  # inline | subprocess
    cascade: bool = False
    parallelism: dict = field(default_factory=dict)
    # Per-run env vars passed through to the evaluator. The runner exports
    # these to `os.environ` for the duration of each `evaluate(...)` call
    # (and restores after). Use this for evaluator-side knobs the user wants
    # checked in alongside the YAML — EVAL_DEVICE, EVAL_DATASETS,
    # EVAL_SAMPLE_QUERIES, EVAL_WARMUP_QUERIES, EVAL_TIMED_REPEATS, etc.
    # Values are coerced to str at use time (YAML often parses "10" as int).
    env: dict = field(default_factory=dict)


@dataclass
class TraceConfig:
    enabled: bool = True
    include_prompts: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    debug_log: bool = False
    run_log_max_mb: int = 10
    run_log_backups: int = 5
    proposer_log_max_mb: int = 50
    proposer_log_backups: int = 3
    proposer_log_gzip: bool = True
    evaluator_log_max_mb: int = 25
    evaluator_log_backups: int = 3


@dataclass
class RunStoreConfig:
    backend: str = "sqlite"
    vacuum_on_close: bool = True


@dataclass
class Config:
    task: TaskConfig
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    search: MapElitesIslandsConfig = field(default_factory=MapElitesIslandsConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    run_store: RunStoreConfig = field(default_factory=RunStoreConfig)
