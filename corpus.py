"""Download and clean Project Gutenberg texts for the BPE benchmarks.

Usage:
    from corpus import load_corpus
    train_text = load_corpus([1342, 2701, 84])   # concatenated, cached on disk
    test_text = load_corpus([98])                # held-out book for encoding tests

If the download fails (network or rate limit), open the book's page at
gutenberg.org/ebooks/<id>, save the "Plain Text UTF-8" file as
cache/<id>.txt, and rerun. load_corpus reads the cache first.
"""

import html
import json
import random
import re
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
CACHE = Path(__file__).parent / "cache"

# Suggested books: 1342 Pride and Prejudice, 2701 Moby Dick, 84 Frankenstein,
# 98 A Tale of Two Cities, 1661 Adventures of Sherlock Holmes.


def strip_gutenberg(raw):
    """Remove the Project Gutenberg license header and footer."""
    lines = raw.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if line.startswith("*** START OF"):
            start = i + 1
        elif line.startswith("*** END OF"):
            end = i
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def unwrap(text):
    """Join hard-wrapped lines. Keeps blank lines (paragraph breaks).

    Gutenberg and many other plain-text books wrap lines at about 70
    characters. Without this, the same word appears as " the" mid-line and
    "the" at the start of a line, which splits its statistics across two
    tokens and distorts the merges.
    """
    text = text.replace("\r\n", "\n")
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def load_book(book_id):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{book_id}.txt"
    if not path.exists():
        req = urllib.request.Request(
            URL.format(id=book_id), headers={"User-Agent": "Mozilla/5.0 (class project)"}
        )
        raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
        path.write_text(strip_gutenberg(raw), encoding="utf-8")
    return unwrap(path.read_text(encoding="utf-8"))


def load_corpus(book_ids):
    return "\n".join(load_book(b) for b in book_ids)


def load_files(paths, strip_boilerplate=False):
    """Load your own plain-text files (for example a folder of Austen novels)."""
    texts = []
    for p in paths:
        raw = Path(p).read_text(encoding="utf-8", errors="replace")
        texts.append(unwrap(strip_gutenberg(raw) if strip_boilerplate else raw))
    return "\n".join(texts)


TAG = re.compile(r"<[^>]+>")


def load_jeopardy(path, seed=0, test_fraction=0.1):
    """Load Jeopardy questions from a JSON file (list of records with a
    "question" field). Returns (train_text, test_text), one question per line.

    Print a few raw records first: dataset versions differ in quoting and
    HTML. Questions are shuffled with a fixed seed so that any prefix is a
    representative sample, not one era of the show.
    """
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    questions = []
    for r in rows:
        q = TAG.sub("", html.unescape(r["question"])).replace("\\'", "'").strip()
        if len(q) >= 2 and q[0] == q[-1] == "'":
            q = q[1:-1]
        questions.append(" ".join(q.split()))
    random.Random(seed).shuffle(questions)
    cut = int(len(questions) * (1 - test_fraction))
    return "\n".join(questions[:cut]) + "\n", "\n".join(questions[cut:]) + "\n"


if __name__ == "__main__":
    text = load_corpus([1342])
    print(f"{len(text):,} characters")
    print(text[:300])
