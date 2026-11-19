"""Benchmarks for naive vs efficient BPE training and for encoding.

    python benchmark.py --quick     # small sizes, runs in about a minute
    python benchmark.py             # full run (naive is slow on large inputs)

Writes results.csv and three plots next to this file.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bpe import encode, encode_naive, to_sequence, train_efficient, train_naive
from corpus import load_corpus

OUT = Path(__file__).parent
TRAIN_BOOKS = [1342, 2701, 84]   # Pride and Prejudice, Moby Dick, Frankenstein
TEST_BOOKS = [98]                # A Tale of Two Cities (held out)


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def slope(xs, ys):
    """Log-log slope: the empirical exponent of growth."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def plot(xs, series, xlabel, title, fname):
    plt.figure(figsize=(6, 4))
    for label, ys in series.items():
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel("seconds")
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    train_text = load_corpus(TRAIN_BOOKS)
    test_text = load_corpus(TEST_BOOKS)
    print(f"train corpus: {len(train_text):,} chars, test corpus: {len(test_text):,} chars")

    if args.quick:
        sizes, fixed_m = [10_000, 20_000, 40_000, 80_000], 100
        merge_counts, fixed_n = [25, 50, 100, 200], 40_000
        enc_sizes, enc_m = [5_000, 10_000, 20_000, 40_000], 200
    else:
        sizes, fixed_m = [25_000, 50_000, 100_000, 200_000, 400_000], 200
        merge_counts, fixed_n = [50, 100, 200, 400, 800], 100_000
        enc_sizes, enc_m = [25_000, 50_000, 100_000, 200_000], 500

    rows = []

    # Experiment 1: vary corpus size N, fixed M
    t_naive, t_eff = [], []
    for n in sizes:
        seq = to_sequence(train_text[:n])
        (m1, s1), a = timed(train_naive, seq, fixed_m)
        (m2, s2), b = timed(train_efficient, seq, fixed_m)
        assert m1 == m2 and s1 == s2, "trainers disagree"
        t_naive.append(a); t_eff.append(b)
        rows.append(["vary_N", n, len(seq), fixed_m, a, b])
        print(f"N={n:>8,} M={fixed_m}: naive {a:7.2f}s  efficient {b:6.2f}s")
    plot(sizes, {"naive": t_naive, "efficient": t_eff}, "corpus size (chars)",
         f"Training time vs corpus size (M={fixed_m})", "plot_vary_N.png")
    print(f"slope vs N: naive {slope(sizes, t_naive):.2f}, efficient {slope(sizes, t_eff):.2f}")

    # Experiment 2: vary number of merges M, fixed N
    seq = to_sequence(train_text[:fixed_n])
    t_naive, t_eff = [], []
    for m in merge_counts:
        (m1, s1), a = timed(train_naive, seq, m)
        (m2, s2), b = timed(train_efficient, seq, m)
        assert m1 == m2 and s1 == s2, "trainers disagree"
        t_naive.append(a); t_eff.append(b)
        rows.append(["vary_M", fixed_n, len(seq), m, a, b])
        print(f"N={fixed_n:>8,} M={m:>4}: naive {a:7.2f}s  efficient {b:6.2f}s")
    plot(merge_counts, {"naive": t_naive, "efficient": t_eff}, "number of merges M",
         f"Training time vs merges (N={fixed_n:,})", "plot_vary_M.png")
    print(f"slope vs M: naive {slope(merge_counts, t_naive):.2f}, efficient {slope(merge_counts, t_eff):.2f}")

    # Experiment 3: encoding unseen text with a learned merge table
    merges, _ = train_efficient(to_sequence(train_text[:fixed_n]), enc_m)
    t_enc_naive, t_enc = [], []
    for n in enc_sizes:
        chunk = test_text[:n]
        ids1, a = timed(encode_naive, chunk, merges)
        ids2, b = timed(encode, chunk, merges)
        assert ids1 == ids2, "encoders disagree"
        t_enc_naive.append(a); t_enc.append(b)
        rows.append(["encode", n, len(ids2), enc_m, a, b])
        print(f"encode {n:>8,} chars, M={enc_m}: apply-all-merges {a:6.2f}s  rank-based {b:6.2f}s"
              f"  ({n / len(ids2):.2f} chars/token)")
    plot(enc_sizes, {"apply all merges": t_enc_naive, "rank-based": t_enc},
         "text length (chars)", f"Encoding time (M={enc_m})", "plot_encode.png")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "chars", "symbols_or_tokens", "merges", "naive_s", "efficient_s"])
        w.writerows(rows)
    print("wrote results.csv and plots")


if __name__ == "__main__":
    main()
