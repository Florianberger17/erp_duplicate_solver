import pytest
from pydantic import ValidationError

from erp_duplicate_resolver.models import (
    CandidatePair,
    DuplicateGroup,
    NormalisedRecord,
    make_pair_id,
)


def _pair(**overrides):
    values = dict(
        pair_id="R-1|R-2",
        a="R-1",
        b="R-2",
        s_string=0.9,
        s_embed=0.8,
        s_attr=1.0,
        dimension_conflict=False,
        score=0.88,
        decision="llm_match",
    )
    values.update(overrides)
    return CandidatePair(**values)


def test_pair_id_is_order_independent():
    assert make_pair_id("R-2", "R-1") == make_pair_id("R-1", "R-2") == "R-1|R-2"


def test_pair_id_rejects_self_pair():
    with pytest.raises(ValueError):
        make_pair_id("R-1", "R-1")


def test_candidate_pair_requires_sorted_members():
    with pytest.raises(ValidationError):
        _pair(pair_id="R-1|R-2", a="R-2", b="R-1")


def test_candidate_pair_rejects_mismatched_id():
    with pytest.raises(ValidationError):
        _pair(pair_id="R-1|R-3")


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("auto_match", True),
        ("llm_match", True),
        ("human_match", True),
        ("llm_reject", False),
        ("pending_human", False),
        ("discarded", False),
    ],
)
def test_is_match(decision, expected):
    assert _pair(decision=decision).is_match is expected


def test_duplicate_group_survivor_must_be_member():
    with pytest.raises(ValidationError):
        DuplicateGroup(group_id="G1", members=["R-1", "R-2"], survivor="R-3", survivor_reason="x")


def test_duplicate_group_needs_two_members():
    with pytest.raises(ValidationError):
        DuplicateGroup(group_id="G1", members=["R-1"], survivor="R-1", survivor_reason="x")


def test_dimension_tokens_serialise_sorted():
    rec = NormalisedRecord(
        material_number="R-1",
        normalised_description="schraube",
        dimension_tokens=frozenset({"m8x40", "40mm", "dn50"}),
    )
    assert rec.model_dump(mode="json")["dimension_tokens"] == ["40mm", "dn50", "m8x40"]
