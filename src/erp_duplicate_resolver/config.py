"""Typed loading of config/settings.yaml and config/abbreviations.yaml (spec section 7)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_ABBREVIATIONS = PROJECT_ROOT / "config" / "abbreviations.yaml"

BlockingStrategy = Literal["exact_key", "first_token", "ann"]
SurvivorshipRule = Literal[
    "status_active", "has_transactions", "completeness", "created_on", "material_number"
]

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class BlockingConfig(BaseModel):
    strategies: list[BlockingStrategy] = Field(min_length=1)
    ann_top_k: int = Field(gt=0)


class ScoringConfig(BaseModel):
    w_string: float = Field(ge=0)
    w_embed: float = Field(ge=0)
    w_attr: float = Field(ge=0)
    dimension_veto_cap: float = Field(ge=0, le=1)
    weight_tolerance_pct: float = Field(ge=0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoringConfig:
        total = self.w_string + self.w_embed + self.w_attr
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scoring weights must sum to 1, got {total}")
        return self


class ThresholdConfig(BaseModel):
    high: float = Field(ge=0, le=1)
    low: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _low_below_high(self) -> ThresholdConfig:
        if self.low > self.high:
            raise ValueError(f"thresholds.low ({self.low}) must not exceed high ({self.high})")
        return self


class SurvivorshipConfig(BaseModel):
    rules: list[SurvivorshipRule] = Field(min_length=1)


class EmbeddingConfig(BaseModel):
    model: str
    revision: str | None = None
    cache: Path


class LLMConfig(BaseModel):
    enabled: bool = True
    model: str | None = None
    temperature: float = 0.0

    @property
    def usable(self) -> bool:
        """The LLM is only called when enabled and a model string is configured."""
        return self.enabled and bool(self.model)


class Settings(BaseModel):
    blocking: BlockingConfig
    scoring: ScoringConfig
    thresholds: ThresholdConfig
    survivorship: SurvivorshipConfig
    embeddings: EmbeddingConfig
    llm: LLMConfig


def _expand_env(value: Any) -> Any:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in string values.

    An unset variable without a default becomes ``None`` when it is the whole
    value (so ``model: ${LLM_MODEL}`` cleanly means "no model"), and an empty
    string when it is embedded in a longer string.
    """
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    full = _ENV_PATTERN.fullmatch(value)
    if full:
        resolved = os.environ.get(full["name"]) or full["default"]
        return resolved or None
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m["name"]) or m["default"] or "", value)


def load_settings(path: Path | str = DEFAULT_SETTINGS) -> Settings:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(_expand_env(raw))


def load_abbreviations(path: Path | str = DEFAULT_ABBREVIATIONS) -> dict[str, str]:
    """Raw abbreviation mapping; folding/lowercasing is done by the normaliser."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of abbreviation -> expansion")
    return {str(k): str(v) for k, v in raw.items()}
