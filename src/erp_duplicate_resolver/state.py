"""LangGraph state of the resolver (spec section 5)."""

from __future__ import annotations

from typing import Any, TypedDict

from erp_duplicate_resolver.models import (
    CandidatePair,
    DuplicateGroup,
    MergeProposal,
    NormalisedRecord,
)


class ResolverState(TypedDict, total=False):
    records: list[dict[str, Any]]
    normalised: dict[str, NormalisedRecord]  # material_number -> normalised record
    candidate_pairs: list[CandidatePair]
    matches: list[CandidatePair]  # confirmed
    pending_human: list[str]  # pair_id
    groups: list[DuplicateGroup]
    merges: list[MergeProposal]
    stats: dict[str, Any]  # blocking and run metrics
    run_id: str
