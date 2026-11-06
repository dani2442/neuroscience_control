#!/usr/bin/env python3
"""List every run behind the paper's tables and figures, with exclusions.

Writes one row per (table/figure, model, seed): the result file, whether the
run entered the reported numbers, and why not if it did not. A missing result
file whose SLURM log ends in a ``RuntimeError`` is recorded as a numerical
failure with that log, so exclusions are traceable without reading job logs.

Usage::

    python examples/build_run_manifest.py --out results/run_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

RUNS = Path("results/runs")
PERS = Path("results/personalization")
LOG_DIRS = [Path("logs/revision"), Path("logs/fig4d")]

ALL5 = ["hopf", "nsde", "hybrid_hopf", "gnn_hopf", "hybrid_neural"]
MAIN3 = ["hopf", "nsde", "hybrid_hopf"]
SEEDS = list(range(42, 52))

# (artefact, generator, run families, seeds, balanced across families)
ARTEFACTS = [
    ("Table 1 (tab:model_comparison_hcp)", "aggregate_seeds.py --main", [f"{m}_n94" for m in MAIN3], SEEDS, True),
    ("Supp. table tab:full_model_comparison", "aggregate_seeds.py", [f"{m}_n94" for m in ALL5], SEEDS, True),
    ("Supp. table tab:model_comparison_grid_vs_gradient", "aggregate_grid_vs_gradient.py",
     ["hopf_grid", "hopf_n94"], SEEDS, True),
    ("Supp. table tab:ablations", "aggregate_ablations.py",
     [f"hybrid_hopf_{a}" for a in
      ("baseline", "scshuffle", "scrandom", "fixedcoupling", "nofdm", "nodyn")]
     + ["nsde_pmatch"], [42, 43, 44], False),
    ("Figure 4d (10 training subjects)", "plot_fig4_panels.py --panel d",
     [f"{m}_fixtest_tr10" for m in MAIN3], SEEDS, True),
    ("Figure 4d (65 training subjects)", "plot_fig4_panels.py --panel d",
     [f"{m}_fixtest_tr65" for m in MAIN3], SEEDS, True),
]


def _failure_log(family: str, seed: int) -> str:
    """Return the log of a failed run for *family*/*seed*, if one ends in an error."""
    stem = family.replace("_n94", "") + (f"_n94_seed{seed}" if family.endswith("_n94")
                                         else f"_seed{seed}")
    stem = re.sub(r"_fixtest_tr(\d+)_seed", r"_fixtest_tr\1_seed", stem)
    for d in LOG_DIRS:
        for log in sorted(d.glob(f"{stem}_*.log")):
            m = re.findall(r"^(\w*Error: .*)$", log.read_text(errors="replace"), re.M)
            if m:
                return f"{log} ({m[-1][:80]})"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--out", default="results/run_manifest.csv")
    args = ap.parse_args()

    rows = []
    for name, generator, families, seeds, balanced in ARTEFACTS:
        present = {f: {s for s in seeds if (RUNS / f"{args.ds}_{f}_seed{s}.json").exists()}
                   for f in families}
        common = set.intersection(*present.values()) if balanced else None
        for fam in families:
            for s in seeds:
                path = RUNS / f"{args.ds}_{fam}_seed{s}.json"
                if s in present[fam]:
                    used = common is None or s in common
                    reason = "" if used else "seed missing for another model (complete-case design)"
                else:
                    used = False
                    log = _failure_log(fam, s)
                    reason = f"run failed: {log}" if log else "no result file"
                rows.append([name, generator, fam, s, path if s in present[fam] else "",
                             "yes" if used else "no", reason])

    for model in MAIN3:
        for subj in (1, 2, 14, 20, 21, 23, 29, 32, 37, 41):  # logs/revision/launch_personalization.sh
            path = PERS / f"{args.ds}_{model}_sub{subj}_seed42.json"
            rows.append(["Supp. table tab:personalization_n1 / Figure 4e", "aggregate_personalization.py",
                         f"{model}_sub{subj}", 42, path if path.exists() else "",
                         "yes" if path.exists() else "no", "" if path.exists() else "no result file"])

    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["artefact", "generator", "run_family", "seed", "result_file", "used", "note"])
        w.writerows(rows)
    n_no = sum(r[5] == "no" for r in rows)
    print(f"Wrote {out}: {len(rows)} rows, {n_no} not used")
    for r in rows:
        if r[5] == "no":
            print(f"  {r[0][:40]:40s} {r[2]:24s} seed {r[3]}: {r[6]}")


if __name__ == "__main__":
    main()
