import csv
import json
import random
from collections import Counter

import pytest

from erp_duplicate_resolver.models import MASTER_COLUMNS
from erp_duplicate_resolver.synthetic import (
    FAMILIES,
    N_ATTRIBUTE_PAIRS,
    N_DIMENSION_NEGATIVES,
    N_MATERIAL_NEGATIVES,
    N_TRIPLES,
    N_TYPE_NEGATIVES,
    N_VARIANT_PAIRS,
    Entity,
    generate,
    render_variant,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return generate()


@pytest.fixture(scope="module")
def by_number(dataset):
    return {r["material_number"]: r for r in dataset.records}


def test_record_count_and_unique_numbers(dataset):
    numbers = [r["material_number"] for r in dataset.records]
    assert len(numbers) == 500
    assert len(set(numbers)) == 500


def test_group_sizes_match_spec(dataset):
    sizes = Counter(len(g["members"]) for g in dataset.groups)
    assert sizes == {2: N_VARIANT_PAIRS + N_ATTRIBUTE_PAIRS, 3: N_TRIPLES}


def test_hard_negative_counts_match_spec(dataset):
    kinds = Counter(n["kind"] for n in dataset.hard_negatives)
    assert kinds == {
        "dimension": N_DIMENSION_NEGATIVES,
        "material_surface": N_MATERIAL_NEGATIVES,
        "material_type": N_TYPE_NEGATIVES,
    }


def test_each_record_in_at_most_one_group(dataset, by_number):
    members = [m for g in dataset.groups for m in g["members"]]
    assert len(members) == len(set(members))
    assert set(members) <= set(by_number)


def test_hard_negatives_are_never_ground_truth_pairs(dataset):
    group_of = {m: g["group_id"] for g in dataset.groups for m in g["members"]}
    for neg in dataset.hard_negatives:
        a, b = neg["members"]
        assert a not in group_of or group_of.get(a) != group_of.get(b)


def test_duplicate_descriptions_differ_except_attribute_pairs(dataset, by_number):
    identical = 0
    for g in dataset.groups:
        descriptions = {by_number[m]["description"] for m in g["members"]}
        if len(descriptions) == 1:
            identical += 1
        else:
            assert len(descriptions) == len(g["members"])
    assert identical == N_ATTRIBUTE_PAIRS


def test_attribute_pairs_diverge_in_plant_or_valuation_class(dataset, by_number):
    for g in dataset.groups:
        recs = [by_number[m] for m in g["members"]]
        if len({r["description"] for r in recs}) == 1:
            a, b = recs
            assert a["plant"] != b["plant"] or a["valuation_class"] != b["valuation_class"]


def test_duplicates_share_material_type(dataset, by_number):
    for g in dataset.groups:
        assert len({by_number[m]["material_type"] for m in g["members"]}) == 1


def test_material_type_negatives_differ_in_type(dataset, by_number):
    for neg in dataset.hard_negatives:
        a, b = (by_number[m] for m in neg["members"])
        if neg["kind"] == "material_type":
            assert a["material_type"] != b["material_type"]
        else:
            assert a["material_type"] == b["material_type"]


def test_hard_negatives_differ_in_exactly_one_word_group(dataset, by_number):
    # Canonical descriptions of a hard negative pair differ, but only slightly.
    for neg in dataset.hard_negatives:
        a, b = (set(by_number[m]["description"].split()) for m in neg["members"])
        assert a != b
        assert len(a ^ b) <= 4, (neg, a, b)


def test_generation_is_deterministic():
    first, second = generate(seed=7), generate(seed=7)
    assert first.records == second.records
    assert first.groups == second.groups
    assert first.hard_negatives == second.hard_negatives


def test_different_seed_changes_data():
    assert generate(seed=1).records != generate(seed=2).records


def test_too_few_records_rejected():
    with pytest.raises(ValueError):
        generate(n_records=50)


def test_render_variant_never_returns_excluded_text():
    rng = random.Random(0)
    entity = Entity("hex_bolt", (("d", "8"), ("l", "40")), "verzinkt", "ROH")
    seen = {entity.description()}
    for _ in range(20):
        text = render_variant(entity, rng, exclude=seen)
        assert text not in seen
        seen.add(text)


def test_every_family_renders_without_placeholders():
    for fam in FAMILIES:
        entity = Entity(fam.key, tuple(fam.dims[0].items()), fam.mats[0], fam.types[0])
        text = entity.description()
        assert "{" not in text and "}" not in text
        assert entity.weight_kg() > 0


def test_written_files_round_trip(dataset, tmp_path):
    paths = write_dataset(dataset, tmp_path)
    with paths["master"].open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == MASTER_COLUMNS
    assert len(rows) == 500
    assert {r["status"] for r in rows} <= {"01", "02"}
    assert {r["has_transactions"] for r in rows} == {"true", "false"}
    assert any(r["net_weight"] == "" for r in rows)
    groups = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    assert groups[0]["group_id"] == "G001"
    assert groups == dataset.groups
