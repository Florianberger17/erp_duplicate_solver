"""Data models shared by all pipeline nodes (spec section 5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

MATERIAL_TYPES = ("ROH", "HALB", "FERT", "HIBE")
BASE_UNITS = ("ST", "KG", "M", "L")
WEIGHT_UNITS = ("KG", "G")
STATUS_ACTIVE = "01"
STATUS_BLOCKED = "02"
STATUS_TO_BE_BLOCKED = "to_be_blocked"

# Column order of data/material_master.csv (spec section 3).
MASTER_COLUMNS: tuple[str, ...] = (
    "material_number",
    "material_type",
    "description",
    "base_unit",
    "plant",
    "valuation_class",
    "net_weight",
    "weight_unit",
    "supplier_id",
    "status",
    "has_transactions",
    "created_on",
)

Decision = Literal[
    "auto_match",
    "llm_match",
    "llm_reject",
    "pending_human",
    "human_match",
    "human_reject",
    "discarded",
]
MATCH_DECISIONS: frozenset[str] = frozenset({"auto_match", "llm_match", "human_match"})


def make_pair_id(a: str, b: str) -> str:
    """Canonical, order-independent pair id: ``"<smaller>|<larger>"``."""
    if a == b:
        raise ValueError(f"a pair needs two distinct records, got {a!r} twice")
    lo, hi = sorted((a, b))
    return f"{lo}|{hi}"


class NormalisedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    material_number: str
    normalised_description: str
    dimension_tokens: frozenset[str]
    block_keys: list[str] = Field(default_factory=list)

    @field_serializer("dimension_tokens")
    def _sorted_tokens(self, tokens: frozenset[str]) -> list[str]:
        # Sets have no stable order; serialise sorted so reports are reproducible.
        return sorted(tokens)


class CandidatePair(BaseModel):
    pair_id: str
    a: str
    b: str
    s_string: float = Field(ge=0.0, le=1.0)
    s_embed: float = Field(ge=-1.0, le=1.0)
    s_attr: float = Field(ge=0.0, le=1.0)
    dimension_conflict: bool
    score: float
    decision: Decision
    llm_rationale: str | None = None
    llm_certainty: Literal["high", "low"] | None = None
    distinguishing_features: list[str] | None = None

    @model_validator(mode="after")
    def _check_pair_id(self) -> CandidatePair:
        if self.a >= self.b:
            raise ValueError(f"pair members must be sorted, got a={self.a!r} b={self.b!r}")
        if self.pair_id != make_pair_id(self.a, self.b):
            raise ValueError(f"pair_id {self.pair_id!r} does not match members")
        return self

    @property
    def is_match(self) -> bool:
        return self.decision in MATCH_DECISIONS


class DuplicateGroup(BaseModel):
    group_id: str
    members: list[str]
    survivor: str
    survivor_reason: str
    needs_review: bool = False

    @model_validator(mode="after")
    def _check_survivor(self) -> DuplicateGroup:
        if len(self.members) < 2:
            raise ValueError("a duplicate group needs at least two members")
        if self.survivor not in self.members:
            raise ValueError(f"survivor {self.survivor!r} is not a member of the group")
        return self


class MergeProposal(BaseModel):
    group_id: str
    survivor: str
    loser: str
    score: float
    decided_by: Literal["rule", "llm", "human"]
    inherited_fields: dict[str, Any] = Field(default_factory=dict)
