"""Generate the synthetic material master and its ground truth (spec section 10).

Usage:
    python scripts/generate_data.py [--seed 42] [--n-records 500] [--out data]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from erp_duplicate_resolver.synthetic import (
    DEFAULT_N_RECORDS,
    DEFAULT_SEED,
    generate,
    write_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-records", type=int, default=DEFAULT_N_RECORDS)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    dataset = generate(seed=args.seed, n_records=args.n_records)
    paths = write_dataset(dataset, args.out)

    sizes = Counter(len(g["members"]) for g in dataset.groups)
    kinds = Counter(n["kind"] for n in dataset.hard_negatives)
    print(f"records:         {len(dataset.records)}")
    print(f"duplicate groups: {len(dataset.groups)} (by size: {dict(sorted(sizes.items()))})")
    print(f"hard negatives:  {len(dataset.hard_negatives)} ({dict(sorted(kinds.items()))})")
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
