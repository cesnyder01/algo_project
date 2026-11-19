"""Derived numbers and a speedup plot from a benchmark results CSV.

    python analyze_results.py jeopardy      # reads results_jeopardy.csv

Prints, for each training experiment, the quantities your predictions talk
about: speedup (naive / efficient) and naive time divided by N * M. It does
not interpret them. Comparing them to your predictions is your job.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else "jeopardy"
path = Path(__file__).parent / f"results_{tag}.csv"
rows = list(csv.DictReader(open(path)))


def get(exp):
    return [
        dict(chars=int(r["chars"]), n=int(r["symbols_or_tokens"]), m=int(r["merges"]),
             naive=float(r["naive_s"]), eff=float(r["efficient_s"]))
        for r in rows if r["experiment"] == exp
    ]


for exp, label in [("vary_N", "Vary N, fixed M"), ("vary_M", "Vary M, fixed N")]:
    data = get(exp)
    print(f"\n{label}")
    print(f"{'N (symbols)':>12} {'M':>5} {'naive s':>9} {'eff s':>8} {'speedup':>8} {'naive/(N*M) (ns)':>17}")
    for d in data:
        print(f"{d['n']:>12,} {d['m']:>5} {d['naive']:>9.2f} {d['eff']:>8.2f} "
              f"{d['naive'] / d['eff']:>8.1f} {d['naive'] / (d['n'] * d['m']) * 1e9:>17.1f}")

data = get("vary_M")
plt.figure(figsize=(6, 4))
plt.plot([d["m"] for d in data], [d["naive"] / d["eff"] for d in data], marker="o")
plt.xlabel("number of merges M")
plt.ylabel("speedup (naive time / efficient time)")
plt.title("Speedup vs. merges")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / f"plot_speedup_{tag}.png", dpi=150)
print(f"\nwrote plot_speedup_{tag}.png")
