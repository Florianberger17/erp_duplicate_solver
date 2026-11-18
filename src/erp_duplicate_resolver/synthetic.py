"""Reproducible synthetic material master with ground truth (spec section 10).

Every record is a rendering of an *entity* (a distinct real-world part).
Duplicates are several records of the same entity whose descriptions differ
by spelling only. Hard negatives are two *different* entities whose canonical
descriptions differ in exactly one decisive feature (dimension, material /
surface, material type). All randomness goes through one seeded ``Random``.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from erp_duplicate_resolver.models import MASTER_COLUMNS, STATUS_ACTIVE, STATUS_BLOCKED

DEFAULT_SEED = 42
DEFAULT_N_RECORDS = 500

N_VARIANT_PAIRS = 40
N_TRIPLES = 5
N_ATTRIBUTE_PAIRS = 8
N_DIMENSION_NEGATIVES = 15
N_MATERIAL_NEGATIVES = 10
N_TYPE_NEGATIVES = 8

PLANTS = ("1000", "1100", "2000")
PLANT_WEIGHTS = (0.5, 0.3, 0.2)
SUPPLIERS = tuple(f"V-{10001 + i}" for i in range(30))
VALUATION_CLASS = {"ROH": "3000", "HALB": "7900", "FERT": "7920", "HIBE": "3030"}
ALT_VALUATION_CLASS = {"ROH": "3001", "HALB": "7910", "FERT": "7925", "HIBE": "3031"}
TYPE_SUFFIX = {"ROH": "Rohteil", "HALB": "bearbeitet", "FERT": "komplett"}

HardNegativeKind = Literal["dimension", "material_surface", "material_type"]


# --------------------------------------------------------------------------- families


@dataclass(frozen=True, eq=False)
class Family:
    key: str
    template: tuple[str, ...]
    dims: tuple[dict[str, str], ...]
    mats: tuple[str, ...]
    types: tuple[str, ...]
    base_unit: str
    weight_kg: Callable[[dict[str, str]], float]
    popularity: int
    dimensional: bool = True  # dims are physical dimensions -> usable for dimension negatives
    material_variants: bool = True  # mats are material/surface -> usable for material negatives


def _grid(**axes: Sequence[str]) -> tuple[dict[str, str], ...]:
    keys = list(axes)
    return tuple(dict(zip(keys, combo, strict=True)) for combo in itertools.product(*axes.values()))


def _f(p: dict[str, str], key: str) -> float:
    return float(p[key].replace(",", "."))


_M = ("5", "6", "8", "10", "12", "16")
_LEN = ("16", "20", "25", "30", "35", "40", "45", "50", "60", "70", "80", "100")
_DN = ("15", "20", "25", "32", "40", "50", "65", "80", "100")
_INCH_KG = {"1/4": 0.03, "3/8": 0.045, "1/2": 0.07, "3/4": 0.11, "1": 0.18}
_PARTS = ("Gehäuse", "Welle", "Deckel", "Lagerbock", "Halter", "Flanschplatte", "Zahnrad")


def _part_weight(p: dict[str, str]) -> float:
    return 0.5 + (int(p["num"]) % 200) / 10


def _bearing_weight(p: dict[str, str]) -> float:
    series = {"60": 1.0, "62": 1.6, "63": 2.5}[p["code"][:2]]
    return 0.02 * (int(p["code"][2:]) + 3) * series


FAMILIES: tuple[Family, ...] = (
    Family(
        "hex_bolt",
        ("Sechskantschraube", "DIN 933", "M{d}x{l}", "{mat}"),
        _grid(d=_M, l=_LEN),
        ("verzinkt", "Edelstahl A2", "Edelstahl A4", "feuerverzinkt", "blank"),
        ("ROH",),
        "ST",
        lambda p: 7.7e-6 * _f(p, "d") ** 2 * _f(p, "l"),
        14,
    ),
    Family(
        "socket_screw",
        ("Zylinderschraube", "ISO 4762", "M{d}x{l}", "{mat}"),
        _grid(d=_M, l=_LEN),
        ("verzinkt", "schwarz", "Edelstahl A2"),
        ("ROH",),
        "ST",
        lambda p: 7.0e-6 * _f(p, "d") ** 2 * _f(p, "l"),
        10,
    ),
    Family(
        "hex_nut",
        ("Sechskantmutter", "DIN 934", "M{d}", "{mat}"),
        _grid(d=_M),
        ("verzinkt", "Edelstahl A2", "Edelstahl A4"),
        ("ROH",),
        "ST",
        lambda p: 8.0e-5 * _f(p, "d") ** 2,
        8,
    ),
    Family(
        "washer",
        ("Unterlegscheibe", "DIN 125", "M{d}", "{mat}"),
        _grid(d=_M),
        ("verzinkt", "Edelstahl A2"),
        ("ROH",),
        "ST",
        lambda p: 1.8e-5 * _f(p, "d") ** 2,
        6,
    ),
    Family(
        "ball_valve",
        ("Kugelhahn", "DN{dn}", "PN{pn}", "{mat}"),
        _grid(dn=_DN, pn=("16", "40")),
        ("Messing", "Edelstahl"),
        ("ROH",),
        "ST",
        lambda p: 7.0e-4 * _f(p, "dn") ** 2,
        5,
    ),
    Family(
        "flange",
        ("Vorschweißflansch", "DN{dn}", "PN16", "{mat}"),
        _grid(dn=_DN),
        ("Stahl", "Edelstahl"),
        ("ROH",),
        "ST",
        lambda p: 0.3 + 1.0e-3 * _f(p, "dn") ** 2,
        4,
    ),
    Family(
        "tube",
        ("Stahlrohr", "{d}x{w}", "{mat}"),
        _grid(d=("12", "16", "20", "25", "30", "40"), w=("1,5", "2", "3")),
        ("E235 blank", "1.4301"),
        ("ROH",),
        "M",
        lambda p: 3.14 * (_f(p, "d") - _f(p, "w")) * _f(p, "w") * 7.85e-3,
        4,
    ),
    Family(
        "round_bar",
        ("Rundstahl", "Ø{d}", "{mat}"),
        _grid(d=("10", "12", "16", "20", "25", "30", "40", "50")),
        ("S235JR", "C45", "42CrMo4"),
        ("ROH",),
        "M",
        lambda p: 6.17e-3 * _f(p, "d") ** 2,
        4,
    ),
    Family(
        "flat_bar",
        ("Flachstahl", "{w}x{t}", "{mat}"),
        _grid(w=("20", "30", "40", "50", "60"), t=("5", "6", "8", "10")),
        ("S235JR", "S355J2"),
        ("ROH",),
        "M",
        lambda p: 7.85e-3 * _f(p, "w") * _f(p, "t"),
        3,
    ),
    Family(
        "fitting",
        ("Gewindemuffe", '{inch}"', "{mat}"),
        _grid(inch=tuple(_INCH_KG)),
        ("Messing", "Edelstahl", "Temperguss verzinkt"),
        ("ROH",),
        "ST",
        lambda p: _INCH_KG[p["inch"]],
        5,
    ),
    Family(
        "o_ring",
        ("O-Ring", "{d}x{t}", "{mat}"),
        _grid(d=("10", "12", "15", "20", "25", "30", "40", "50"), t=("2", "3", "4")),
        ("NBR 70", "FKM 80", "EPDM 70"),
        ("ROH",),
        "ST",
        lambda p: 1.3e-6 * 3.14 * _f(p, "d") * 0.785 * _f(p, "t") ** 2,
        6,
    ),
    Family(
        "cable",
        ("Mantelleitung", "NYM-J", "{n}x{a}", "grau"),
        _grid(n=("3", "5"), a=("1,5", "2,5", "4")),
        ("",),
        ("ROH",),
        "M",
        lambda p: 0.016 * _f(p, "n") * _f(p, "a") + 0.05,
        3,
        material_variants=False,
    ),
    Family(
        "cable_tie",
        ("Kabelbinder", "{l}x{w}", "{mat}"),
        _grid(l=("100", "150", "200", "300", "370"), w=("2,5", "3,6", "4,8", "7,6")),
        ("schwarz", "natur"),
        ("HIBE",),
        "ST",
        lambda p: 1.2e-6 * _f(p, "l") * _f(p, "w") * 1.5,
        3,
    ),
    Family(
        "hose",
        ("PVC-Schlauch", "Ø{d}", "{mat}"),
        _grid(d=("6", "8", "10", "12", "16", "19", "25")),
        ("transparent", "gewebeverstärkt"),
        ("HIBE",),
        "M",
        lambda p: 1.0e-3 * _f(p, "d") ** 2,
        3,
    ),
    Family(
        "bearing",
        ("Rillenkugellager", "{code}", "{mat}"),
        _grid(code=tuple(f"{s}{i:02d}" for s in ("60", "62", "63") for i in range(0, 9))),
        ("2RS", "ZZ", "offen"),
        ("ROH",),
        "ST",
        _bearing_weight,
        6,
        dimensional=False,
        material_variants=False,
    ),
    Family(
        "hydraulic_oil",
        ("Hydrauliköl", "HLP {vg}", "Kanister 20 L"),
        _grid(vg=("22", "32", "46", "68")),
        ("",),
        ("HIBE",),
        "L",
        lambda p: 17.5,
        2,
        dimensional=False,
        material_variants=False,
    ),
    Family(
        "grease",
        ("Schmierfett", "NLGI {nlgi}", "{mat}", "Kartusche 400 g"),
        _grid(nlgi=("1", "2", "3")),
        ("Lithium", "Calcium"),
        ("HIBE",),
        "ST",
        lambda p: 0.4,
        2,
        dimensional=False,
        material_variants=False,
    ),
    Family(
        "paint",
        ("Lack", "RAL {ral}", "{mat}"),
        _grid(ral=("1023", "3000", "5010", "6011", "7035", "9005", "9010")),
        ("seidenmatt", "glänzend"),
        ("HIBE",),
        "L",
        lambda p: 1.3,
        3,
        dimensional=False,
    ),
    Family(
        "gasket",
        ("Flachdichtung", "DN{dn}", "PN16", "{mat}"),
        _grid(dn=_DN),
        ("Klingersil C-4400", "PTFE", "EPDM"),
        ("ROH",),
        "ST",
        lambda p: 1.0e-4 * _f(p, "dn") + 0.002,
        3,
    ),
    Family(
        "part",
        ("{part}", "{prefix}-{num}", "{suffix}"),
        _grid(
            part=_PARTS,
            prefix=("KP", "GW", "ZH", "LB", "HT"),
            num=tuple(str(n) for n in range(100, 1000, 7)),
        ),
        ("",),
        ("ROH", "HALB", "FERT"),
        "ST",
        _part_weight,
        10,
        dimensional=False,
        material_variants=False,
    ),
)
FAMILY_BY_KEY = {f.key: f for f in FAMILIES}


@dataclass(frozen=True)
class Entity:
    family: str
    dims: tuple[tuple[str, str], ...]
    mat: str
    material_type: str

    @property
    def fam(self) -> Family:
        return FAMILY_BY_KEY[self.family]

    def parts(self) -> list[str]:
        values = dict(self.dims, mat=self.mat, suffix=TYPE_SUFFIX.get(self.material_type, ""))
        rendered = (part.format(**values).strip() for part in self.fam.template)
        return [p for p in rendered if p]

    def description(self) -> str:
        return " ".join(self.parts())

    def weight_kg(self) -> float:
        return self.fam.weight_kg(dict(self.dims))


def _canonical_key(description: str) -> str:
    return " ".join(sorted(description.lower().split()))


# --------------------------------------------------------------------------- spelling variants

# canonical word -> abbreviated spellings. Entries marked (*) are NOT in
# config/abbreviations.yaml on purpose: real dictionaries are incomplete.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "verzinkt": ("verz.", "vz", "vzk"),  # vzk (*)
    "feuerverzinkt": ("feuerverz.", "fvz"),
    "Edelstahl": ("Edelst.", "VA"),  # VA (*)
    "Sechskantschraube": ("Sechskantschr.", "6KT-Schraube"),  # 6KT-Schraube (*)
    "Zylinderschraube": ("Zyl.-Schr.",),
    "Sechskantmutter": ("6kt-Mutter",),
    "Unterlegscheibe": ("U-Scheibe",),
    "Kugelhahn": ("Kug.-Hahn",),
    "Messing": ("Ms",),
    "Stahlrohr": ("St.-Rohr",),
    "Rundstahl": ("Rd.-Stahl",),
    "Flachstahl": ("Fl.-Stahl",),
    "schwarz": ("schw.", "sw"),  # sw (*)
    "komplett": ("kpl.",),
    "Rillenkugellager": ("RKL",),
    "Hydrauliköl": ("Hydr.-Öl",),
    "Flachdichtung": ("Flachdicht.",),  # (*) - "dicht." only matches as a whole word
    "transparent": ("transp.",),
    "bearbeitet": ("bearb.",),
    "Gehäuse": ("Geh.",),
    "Kabelbinder": ("Kab.-Binder",),  # (*)
    "Schmierfett": ("Schm.-Fett",),  # (*)
    "Mantelleitung": ("Mantelltg.",),  # (*)
}
_ABBREV_LOOKUP = {k.lower(): v for k, v in ABBREVIATIONS.items()}

COMPOUND_SPLITS = {
    "Sechskantschraube": "Sechskant Schraube",
    "Zylinderschraube": "Zylinder Schraube",
    "Sechskantmutter": "Sechskant Mutter",
    "Unterlegscheibe": "Unterleg Scheibe",
    "Kugelhahn": "Kugel Hahn",
    "Stahlrohr": "Stahl Rohr",
    "Rundstahl": "Rund Stahl",
    "Kabelbinder": "Kabel Binder",
    "Rillenkugellager": "Rillen Kugellager",
    "Flachdichtung": "Flach Dichtung",
}
_SPACING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^M(\d+)x(\d+)$"), r"M\1 x \2"),
    (re.compile(r"^(\d+)x([\d,]+)$"), r"\1 x \2"),
    (re.compile(r"^DN(\d+)$"), r"DN \1"),
    (re.compile(r"^Ø(\d+)$"), r"Ø \1"),
    (re.compile(r"^(DIN|ISO) (\d+)$"), r"\1\2"),
)


def _abbreviate(parts: list[str], rng: random.Random) -> list[str] | None:
    slots = [
        (i, j)
        for i, part in enumerate(parts)
        for j, word in enumerate(part.split(" "))
        if word.lower() in _ABBREV_LOOKUP
    ]
    if not slots:
        return None
    out = [p.split(" ") for p in parts]
    for i, j in rng.sample(slots, k=min(len(slots), rng.choice((1, 2)))):
        out[i][j] = rng.choice(_ABBREV_LOOKUP[out[i][j].lower()])
    return [" ".join(words) for words in out]


def _respace(parts: list[str], rng: random.Random) -> list[str] | None:
    options: list[tuple[int, str]] = []
    for i, part in enumerate(parts):
        if part in COMPOUND_SPLITS:
            options.append((i, COMPOUND_SPLITS[part]))
        for pattern, repl in _SPACING_RULES:
            if pattern.match(part):
                options.append((i, pattern.sub(repl, part)))
    if not options:
        return None
    out = list(parts)
    for i, new in rng.sample(options, k=min(len(options), rng.choice((1, 2)))):
        out[i] = new
    return out


def _reorder(parts: list[str], rng: random.Random) -> list[str] | None:
    if len(parts) < 2:
        return None
    out = list(parts)
    while out == parts:
        rng.shuffle(out)
    return out


def render_variant(entity: Entity, rng: random.Random, exclude: set[str]) -> str:
    """A spelling variant of the entity's description that is not in ``exclude``.

    Applies 1-3 of: reorder, abbreviate, respace (on parts), then joins with a
    punctuated separator and/or changes case.
    """
    for _ in range(200):
        parts = entity.parts()
        ops = rng.sample(
            ["reorder", "abbreviate", "respace", "punctuate", "case"],
            k=rng.choices((1, 2, 3), weights=(0.45, 0.4, 0.15))[0],
        )
        # Part-level ops first, string-level ops last, in a fixed order.
        for op, fn in (("reorder", _reorder), ("abbreviate", _abbreviate), ("respace", _respace)):
            if op in ops:
                parts = fn(parts, rng) or parts
        sep = rng.choice((", ", " - ", "; ", ",")) if "punctuate" in ops else " "
        text = sep.join(parts)
        if "punctuate" in ops and rng.random() < 0.3:
            text += "."
        if "case" in ops:
            text = text.upper()
        if "respace" in ops and rng.random() < 0.3:
            text = text.replace(" ", "  ", 1)
        if text not in exclude:
            return text
    raise RuntimeError(f"could not produce a new spelling variant for {entity}")


# --------------------------------------------------------------------------- generator


@dataclass
class Dataset:
    records: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    hard_negatives: list[dict[str, Any]]


@dataclass
class _Generator:
    rng: random.Random
    used_keys: set[str] = field(default_factory=set)

    # -- entities -----------------------------------------------------------------

    def _claim(self, entity: Entity) -> bool:
        key = _canonical_key(entity.description())
        if key in self.used_keys:
            return False
        self.used_keys.add(key)
        return True

    def new_entity(self, families: Sequence[Family] = FAMILIES) -> Entity:
        weights = [f.popularity for f in families]
        for _ in range(10_000):
            fam = self.rng.choices(families, weights=weights)[0]
            entity = Entity(
                family=fam.key,
                dims=tuple(self.rng.choice(fam.dims).items()),
                mat=self.rng.choice(fam.mats),
                material_type=self.rng.choice(fam.types),
            )
            if self._claim(entity):
                return entity
        raise RuntimeError("entity space exhausted")

    def _sibling(self, kind: HardNegativeKind) -> tuple[Entity, Entity]:
        """Two distinct entities that differ only in one decisive feature."""
        pools: dict[HardNegativeKind, list[Family]] = {
            "dimension": [f for f in FAMILIES if f.dimensional],
            "material_surface": [f for f in FAMILIES if f.material_variants and len(f.mats) > 1],
            "material_type": [f for f in FAMILIES if len(f.types) > 1],
        }
        for _ in range(10_000):
            a = self.new_entity(pools[kind])
            dims = dict(a.dims)
            if kind == "dimension":
                # Change exactly one dimension axis, e.g. M8x40 -> M8x50.
                axis = self.rng.choice(sorted(dims))
                options = sorted({d[axis] for d in a.fam.dims} - {dims[axis]})
                b = Entity(
                    a.family,
                    tuple({**dims, axis: self.rng.choice(options)}.items()),
                    a.mat,
                    a.material_type,
                )
            elif kind == "material_surface":
                mat = self.rng.choice([m for m in a.fam.mats if m != a.mat])
                b = Entity(a.family, a.dims, mat, a.material_type)
            else:
                mtype = self.rng.choice([t for t in a.fam.types if t != a.material_type])
                b = Entity(a.family, a.dims, a.mat, mtype)
            if self._claim(b):
                return a, b
            self.used_keys.discard(_canonical_key(a.description()))
        raise RuntimeError(f"could not build a {kind} hard negative")

    # -- records ------------------------------------------------------------------

    def _date(self, start: date = date(2005, 1, 1), days: int = 7300) -> date:
        return start + timedelta(days=self.rng.randrange(days))

    def base_record(self, entity: Entity, description: str | None = None) -> dict[str, Any]:
        rng = self.rng
        mtype = entity.material_type
        weight = entity.weight_kg()
        net_weight: float | None
        weight_unit: str | None
        if rng.random() < 0.15:
            net_weight, weight_unit = None, None
        elif weight < 1 and rng.random() < 0.15:
            net_weight, weight_unit = round(weight * 1000, 2), "G"
        else:
            net_weight, weight_unit = round(weight, 4), "KG"
        has_supplier = mtype in ("ROH", "HIBE") and rng.random() < 0.9
        return {
            "material_number": None,  # assigned once all records exist
            "material_type": mtype,
            "description": description or entity.description(),
            "base_unit": entity.fam.base_unit,
            "plant": rng.choices(PLANTS, weights=PLANT_WEIGHTS)[0],
            "valuation_class": VALUATION_CLASS[mtype],
            "net_weight": net_weight,
            "weight_unit": weight_unit,
            "supplier_id": rng.choice(SUPPLIERS) if has_supplier else None,
            "status": STATUS_BLOCKED if rng.random() < 0.08 else STATUS_ACTIVE,
            "has_transactions": rng.random() < 0.7,
            "created_on": self._date(),
        }

    def duplicate_of(self, original: dict[str, Any], description: str) -> dict[str, Any]:
        """A second record of the same entity, as typically created by someone else later."""
        rng = self.rng
        dup = dict(original, description=description)
        if dup["supplier_id"] and rng.random() < 0.25:
            dup["supplier_id"] = None
        if dup["net_weight"] is not None:
            roll = rng.random()
            if roll < 0.25:
                dup["net_weight"], dup["weight_unit"] = None, None
            elif roll < 0.40:
                # Same weight, other unit: scoring must compare in a common unit.
                if dup["weight_unit"] == "KG":
                    dup["net_weight"], dup["weight_unit"] = round(dup["net_weight"] * 1000, 2), "G"
                else:
                    dup["net_weight"], dup["weight_unit"] = round(dup["net_weight"] / 1000, 4), "KG"
            else:
                dup["net_weight"] = round(dup["net_weight"] * rng.uniform(0.985, 1.015), 4)
        dup["status"] = STATUS_BLOCKED if rng.random() < 0.15 else STATUS_ACTIVE
        dup["has_transactions"] = rng.random() < 0.5
        dup["created_on"] = self._date()
        return dup

    def sibling_record(self, entity: Entity, partner: dict[str, Any]) -> dict[str, Any]:
        """Record of a hard-negative entity, sharing context with its partner record."""
        rec = self.base_record(entity)
        rec["plant"] = partner["plant"]
        if rec["supplier_id"] and partner["supplier_id"] and self.rng.random() < 0.7:
            rec["supplier_id"] = partner["supplier_id"]
        return rec


def generate(seed: int = DEFAULT_SEED, n_records: int = DEFAULT_N_RECORDS) -> Dataset:
    rng = random.Random(seed)
    gen = _Generator(rng)
    groups: list[list[dict[str, Any]]] = []
    negatives: list[tuple[HardNegativeKind, dict[str, Any], dict[str, Any]]] = []
    singles: list[dict[str, Any]] = []

    # Positives: spelling variants (pairs and triples).
    for size, count in ((2, N_VARIANT_PAIRS), (3, N_TRIPLES)):
        for _ in range(count):
            entity = gen.new_entity()
            original = gen.base_record(entity)
            seen = {original["description"]}
            members = [original]
            for _ in range(size - 1):
                text = render_variant(entity, rng, exclude=seen)
                seen.add(text)
                members.append(gen.duplicate_of(original, text))
            groups.append(members)

    # Positives: identical description, diverging structured attributes.
    for i in range(N_ATTRIBUTE_PAIRS):
        entity = gen.new_entity()
        original = gen.base_record(entity)
        dup = gen.duplicate_of(original, original["description"])
        if i % 2 == 0 or i % 3 == 0:
            dup["plant"] = rng.choice([p for p in PLANTS if p != original["plant"]])
        if i % 2 == 1 or i % 3 == 0:
            dup["valuation_class"] = ALT_VALUATION_CLASS[original["material_type"]]
        groups.append([original, dup])

    # Hard negatives.
    for kind, count in (
        ("dimension", N_DIMENSION_NEGATIVES),
        ("material_surface", N_MATERIAL_NEGATIVES),
        ("material_type", N_TYPE_NEGATIVES),
    ):
        for _ in range(count):
            a, b = gen._sibling(kind)  # type: ignore[arg-type]
            rec_a = gen.base_record(a)
            negatives.append((kind, rec_a, gen.sibling_record(b, rec_a)))  # type: ignore[arg-type]

    n_fixed = sum(len(g) for g in groups) + 2 * len(negatives)
    if n_records < n_fixed:
        raise ValueError(f"n_records must be at least {n_fixed}, got {n_records}")
    for _ in range(n_records - n_fixed):
        singles.append(gen.base_record(gen.new_entity()))

    # Material numbers are drawn at random so duplicates are not adjacent.
    records = (
        [r for g in groups for r in g] + [r for _, a, b in negatives for r in (a, b)] + singles
    )
    numbers = rng.sample(range(100000, 100000 + 2 * n_records), n_records)
    for rec, num in zip(records, numbers, strict=True):
        rec["material_number"] = f"R-{num}"

    ordered_groups = sorted(sorted(r["material_number"] for r in g) for g in groups)
    ground_truth = [
        {"group_id": f"G{i:03d}", "members": members}
        for i, members in enumerate(ordered_groups, start=1)
    ]
    hard_negatives = sorted(
        (
            {"kind": kind, "members": sorted([a["material_number"], b["material_number"]])}
            for kind, a, b in negatives
        ),
        key=lambda n: (n["kind"], n["members"]),
    )
    records.sort(key=lambda r: r["material_number"])
    return Dataset(records=records, groups=ground_truth, hard_negatives=hard_negatives)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def write_dataset(dataset: Dataset, out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "master": out / "material_master.csv",
        "ground_truth": out / "ground_truth_duplicates.json",
        "hard_negatives": out / "hard_negatives.json",
    }
    with paths["master"].open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MASTER_COLUMNS)
        for rec in dataset.records:
            writer.writerow([_csv_value(rec[col]) for col in MASTER_COLUMNS])
    for key, payload in (
        ("ground_truth", dataset.groups),
        ("hard_negatives", dataset.hard_negatives),
    ):
        paths[key].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return paths
