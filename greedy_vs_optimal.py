"""Brute-force check that greedy BPE is not optimal on a tiny input.

Greedy BPE picks the most frequent pair each step. Exhaustive search tries
every possible merge at every step and keeps the shortest final sequence.
"""

from bpe import merge_pair, train_naive


def optimal_length(seq, num_merges):
    best = len(seq)

    def rec(s, k, new_id):
        nonlocal best
        best = min(best, len(s))
        if k == 0:
            return
        for pair in set(zip(s, s[1:])):
            rec(merge_pair(s, pair, new_id), k - 1, new_id + 1)

    rec(seq, num_merges, 256)
    return best


if __name__ == "__main__":
    word, m = "caacaaaccabc", 2
    seq = [ord(c) for c in word]
    greedy = len(train_naive(seq, m)[1])
    print(f"{word!r}, {m} merges: greedy -> {greedy} tokens, optimal -> {optimal_length(seq, m)} tokens")
