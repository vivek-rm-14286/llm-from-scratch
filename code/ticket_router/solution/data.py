"""Synthetic support inbox for Nimbus Monitor.

    python -m solution.data --out data

writes train.jsonl, cal.jsonl, test.jsonl and ood.jsonl. Every row is
{"id", "text", "label", "clean_label", "kind", "secondary", "noisy"}.

kind is one of
    plain       a phrasing the training split may contain
    paraphrase  a held-out phrasing (calibration and test only)
    two_queue   the message also mentions a second queue; the label is the primary one
    short       a few words, compatible with several queues
    ood         not about Nimbus Monitor (ood.jsonl only, label "ood")
"""
import argparse
import json
import random
from pathlib import Path

try:
    from .bank import (CONFUSABLE, GREETINGS, INTENTS, MENTIONS, N_TRAIN_PHRASINGS, OOD,
                       QUEUE_DESCRIPTIONS, QUEUES, SHORTS, SIGNOFFS, SLOTS)
except ImportError:  # imported as a top-level module
    from bank import (CONFUSABLE, GREETINGS, INTENTS, MENTIONS, N_TRAIN_PHRASINGS, OOD,
                      QUEUE_DESCRIPTIONS, QUEUES, SHORTS, SIGNOFFS, SLOTS)

__all__ = ["QUEUES", "QUEUE_DESCRIPTIONS", "generate", "load_split", "write_splits"]

QUEUE_WEIGHTS = [0.14, 0.18, 0.16, 0.18, 0.12, 0.12, 0.10]   # same order as QUEUES
KIND_WEIGHTS = {"base": 0.63, "two_queue": 0.22, "short": 0.15}


def fill_slots(template, rng):
    """Replace every {slot} in the template with a random value from SLOTS."""
    out = template
    for name, values in SLOTS.items():
        token = "{" + name + "}"
        while token in out:
            out = out.replace(token, rng.choice(values), 1)
    return out


def add_typos(text, rng):
    """Swap two adjacent letters in one or two words longer than four characters."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    for i in rng.sample(candidates, k=min(len(candidates), rng.choice([1, 2]))):
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def restyle(text, rng):
    """Surface variation: greeting, sign-off, lower case, typos."""
    if rng.random() < 0.30:
        text = rng.choice(GREETINGS) + " " + text
    if rng.random() < 0.20:
        text = text + " " + rng.choice(SIGNOFFS)
    if rng.random() < 0.20:
        text = text.lower()
    if rng.random() < 0.15:
        text = add_typos(text, rng)
    return text


def sample_issue(queue, rng, allow_heldout):
    """One sentence for the queue. Returns (text, used_heldout_phrasing)."""
    phrasings = rng.choice(INTENTS[queue])
    idx = rng.randrange(len(phrasings) if allow_heldout else N_TRAIN_PHRASINGS)
    return fill_slots(phrasings[idx], rng), idx >= N_TRAIN_PHRASINGS


def make_two_queue(primary, rng, allow_heldout):
    """A message whose request belongs to `primary` but that also talks about another queue."""
    secondary = rng.choice([q for q in QUEUES if q != primary])
    main, _ = sample_issue(primary, rng, allow_heldout)
    form = rng.randrange(4)
    if form == 0:
        text = f"{main} For context, {fill_slots(rng.choice(MENTIONS[secondary]), rng)}."
    elif form == 1:
        mention = fill_slots(rng.choice(MENTIONS[secondary]), rng)
        text = f"{mention[0].upper()}{mention[1:]}, but that is not the issue. {main}"
    elif form == 2:
        side, _ = sample_issue(secondary, rng, allow_heldout)
        text = f"{main} Separately, and much less urgent: {side}"
    else:
        side, _ = sample_issue(secondary, rng, allow_heldout)
        text = f"Two things. The main one: {main} A minor one for later: {side}"
    return text, secondary


def make_example(rng, allow_heldout):
    """One labeled message, before label noise."""
    kind = rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
    secondary = None
    if kind == "short":
        text, candidates = rng.choice(SHORTS)
        label = rng.choice(candidates)                 # irreducible ambiguity
        if rng.random() < 0.3:
            text = text + rng.choice(["!", "??", " pls", " - urgent"])
    elif kind == "two_queue":
        label = rng.choices(QUEUES, weights=QUEUE_WEIGHTS)[0]
        text, secondary = make_two_queue(label, rng, allow_heldout)
        text = restyle(text, rng)
    else:
        label = rng.choices(QUEUES, weights=QUEUE_WEIGHTS)[0]
        text, heldout = sample_issue(label, rng, allow_heldout)
        text = restyle(text, rng)
        kind = "paraphrase" if heldout else "plain"
    return {"text": text, "label": label, "clean_label": label, "kind": kind,
            "secondary": secondary, "noisy": False}


def make_split(name, n, rng, allow_heldout, exclude):
    """n examples. Long messages never repeat a text in `exclude` or within the split."""
    rows, seen = [], set(exclude)
    while len(rows) < n:
        row = make_example(rng, allow_heldout)
        if row["kind"] != "short":                     # short messages legitimately repeat
            if row["text"] in seen:
                continue
            seen.add(row["text"])
        row["id"] = f"{name}-{len(rows):05d}"
        rows.append(row)
    return rows


def apply_label_noise(rows, rate, rng):
    """Flip `rate` of the labels to a confusable queue. clean_label keeps the truth."""
    for row in rows:
        if rng.random() < rate:
            row["label"] = rng.choice(CONFUSABLE[row["clean_label"]])
            row["noisy"] = True
    return rows


def make_ood(n, rng):
    rows = []
    for i in range(n):
        text = fill_slots(rng.choice(OOD), rng)
        if rng.random() < 0.3:
            text = rng.choice(GREETINGS) + " " + text
        if rng.random() < 0.2:
            text = text.lower()
        rows.append({"id": f"ood-{i:05d}", "text": text, "label": "ood", "clean_label": "ood",
                     "kind": "ood", "secondary": None, "noisy": False})
    return rows


def generate(seed=0, n_train=1600, n_cal=500, n_test=1000, n_ood=200, noise=0.05):
    """Build all four splits. The same arguments always give the same rows.

    Only the training labels are noisy: the calibration and test splits stand for
    a sample that a human reviewed twice.
    """
    rng = random.Random(seed)
    train = make_split("train", n_train, rng, allow_heldout=False, exclude=())
    train = apply_label_noise(train, noise, rng)
    used = {r["text"] for r in train}
    cal = make_split("cal", n_cal, rng, allow_heldout=True, exclude=used)
    used |= {r["text"] for r in cal}
    test = make_split("test", n_test, rng, allow_heldout=True, exclude=used)
    return {"train": train, "cal": cal, "test": test, "ood": make_ood(n_ood, rng)}


def write_splits(splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(data_dir, name):
    with open(Path(data_dir) / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-cal", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-ood", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.05)
    args = ap.parse_args()
    splits = generate(args.seed, args.n_train, args.n_cal, args.n_test, args.n_ood, args.noise)
    write_splits(splits, args.out)
    for name, rows in splits.items():
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        noisy = sum(r["noisy"] for r in rows)
        print(f"{name:5s} {len(rows):5d} rows  kinds={kinds}  noisy labels={noisy}")
    for r in splits["test"][:6]:
        print(f"  [{r['kind']:10s}] {r['label']:14s} {r['text']}")


if __name__ == "__main__":
    main()
