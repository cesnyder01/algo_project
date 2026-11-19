"""Byte-Pair Encoding: naive and efficient trainers, plus encoders.

Corpus representation
---------------------
The corpus is one flat list of ints. Values 0-255 are UTF-8 bytes, merged
tokens get ids 256, 257, ... in merge order. SEP marks a word boundary and
is never merged, so merges cannot cross words (GPT-2 style pre-tokenization).

Tie-breaking
------------
Both trainers pick the pair with the highest count, ties broken by the
smallest pair (tuple order). This makes them produce identical merge lists,
which is the main correctness test.

Pair counts include overlapping occurrences ("aaa" has two (a, a) pairs),
but a merge pass replaces occurrences left to right, so "aaa" -> [aa, a].
"""

import heapq
import re
from collections import Counter, defaultdict

SEP = -1   # word boundary marker
DEAD = -2  # marks a node removed from the linked list (efficient trainer)

# Words keep their leading space, e.g. " the", "ing", ",", "\n\n".
PAT = re.compile(r" ?\w+| ?[^\s\w]+|\s+")


def to_sequence(text):
    """Flatten text into a list of byte values with SEP after every word."""
    seq = []
    for word in PAT.findall(text):
        seq.extend(word.encode("utf-8"))
        seq.append(SEP)
    return seq


# ---------------------------------------------------------------------------
# Naive trainer: O(N) per merge, O(N * M) total
# ---------------------------------------------------------------------------

def merge_pair(seq, pair, new_id):
    """Replace every non-overlapping occurrence of pair, left to right."""
    a, b = pair
    out = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_naive(seq, num_merges):
    """Recount every pair and rescan the whole corpus on every merge."""
    seq = list(seq)
    merges = []
    for k in range(num_merges):
        counts = Counter(
            (a, b) for a, b in zip(seq, seq[1:]) if a >= 0 and b >= 0
        )
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        merges.append(pair)
        seq = merge_pair(seq, pair, 256 + k)
    return merges, seq


# ---------------------------------------------------------------------------
# Efficient trainer: hash map of counts + max-heap with lazy deletion
# + doubly linked list over positions. O((N + M) log N) by our analysis.
# ---------------------------------------------------------------------------

def train_efficient(seq, num_merges):
    n = len(seq)
    sym = list(seq)
    if n < 2:
        return [], sym
    prv = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[-1] = -1

    counts = defaultdict(int)   # pair -> number of adjacent occurrences
    where = defaultdict(set)    # pair -> candidate left positions (may be stale)
    for i in range(n - 1):
        a, b = sym[i], sym[i + 1]
        if a >= 0 and b >= 0:
            counts[(a, b)] += 1
            where[(a, b)].add(i)

    # Heap entries are (-count, pair). Entries go stale when counts change;
    # we detect that when popping (lazy deletion).
    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    merges = []
    for k in range(num_merges):
        pair = None
        while heap:
            neg_c, p = heapq.heappop(heap)
            c = counts.get(p, 0)
            if c <= 0:
                continue                      # pair no longer exists
            if c != -neg_c:
                heapq.heappush(heap, (-c, p))  # stale count: refresh, retry
                continue
            pair = p
            break
        if pair is None:
            break

        new_id = 256 + k
        merges.append(pair)
        a, b = pair
        touched = set()

        for i in sorted(where.pop(pair)):
            j = nxt[i]
            # Skip stale positions: node dead, or pair no longer here.
            if sym[i] != a or j == -1 or sym[j] != b:
                continue
            left, right = prv[i], nxt[j]

            counts[pair] -= 1
            if left != -1 and sym[left] >= 0:
                counts[(sym[left], a)] -= 1
                new_left = (sym[left], new_id)
                counts[new_left] += 1
                where[new_left].add(left)
                touched.add(new_left)
            if right != -1 and sym[right] >= 0:
                counts[(b, sym[right])] -= 1
                new_right = (new_id, sym[right])
                counts[new_right] += 1
                where[new_right].add(i)
                touched.add(new_right)

            # Splice: node i becomes the merged token, node j is unlinked.
            sym[i] = new_id
            sym[j] = DEAD
            nxt[i] = right
            if right != -1:
                prv[right] = i

        for p in touched:
            if counts[p] > 0:
                heapq.heappush(heap, (-counts[p], p))

    out = []
    i = 0
    while i != -1:
        out.append(sym[i])
        i = nxt[i]
    return merges, out


# ---------------------------------------------------------------------------
# Vocabulary, encoding, decoding
# ---------------------------------------------------------------------------

def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return vocab


def encode_naive(text, merges):
    """Apply every merge, in order, to every word: O(M * L) per word."""
    out = []
    for word in PAT.findall(text):
        toks = list(word.encode("utf-8"))
        for k, pair in enumerate(merges):
            if len(toks) < 2:
                break
            toks = merge_pair(toks, pair, 256 + k)
        out.extend(toks)
    return out


def encode(text, merges):
    """Rank-based encoding: repeatedly merge the lowest-rank pair present.

    A merge of rank r only creates tokens used by merges of rank > r, so this
    gives the same result as applying the merges in order. Words are cached.
    """
    ranks = {pair: k for k, pair in enumerate(merges)}
    cache = {}
    out = []
    for word in PAT.findall(text):
        if word not in cache:
            toks = list(word.encode("utf-8"))
            while len(toks) > 1:
                best = min(
                    (ranks[p] for p in zip(toks, toks[1:]) if p in ranks),
                    default=None,
                )
                if best is None:
                    break
                toks = merge_pair(toks, merges[best], 256 + best)
            cache[word] = toks
        out.extend(cache[word])
    return out


def decode(ids, merges):
    vocab = build_vocab(merges)
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
