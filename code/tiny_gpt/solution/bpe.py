"""Byte-level BPE tokenizer: train, encode, decode, save, load.

Same interface as the `ByteBPE` class in the Tokenization lesson, plus two
things a real pipeline needs: incremental pair counts (training takes seconds,
not minutes) and a per-chunk cache in `encode`.
"""
import json
import re
from collections import Counter, defaultdict

# GPT-2-style pre-tokenization with the stdlib `re` module. Merges never cross
# a chunk boundary, so a token never spans two words.
PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}                    # (a, b) -> new id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}                               # text -> id
        self._cache: dict[str, list[int]] = {}                          # chunk -> ids

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Learn `vocab_size - 256` merges from `text`."""
        assert vocab_size >= 256
        self.__init__()
        chunk_freq = Counter(PAT.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_freq]           # one entry per unique chunk
        freqs = list(chunk_freq.values())

        # Pair counts, and for each pair the set of words that contain it.
        counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w, (ids, f) in enumerate(zip(words, freqs)):
            for p in zip(ids, ids[1:]):
                counts[p] += f
                where[p].add(w)

        for new_id in range(256, vocab_size):
            if not counts:
                break
            # Most frequent pair; ties go to the smaller pair so training is deterministic.
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[pair] <= 0:
                break
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {new_id}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} (count {counts[pair]})")
            # Only the words that contain the pair change. Remove their old pairs
            # from the counts, merge, add the new pairs back.
            for w in list(where[pair]):
                ids, f = words[w], freqs[w]
                for p in zip(ids, ids[1:]):
                    counts[p] -= f
                    where[p].discard(w)
                ids = words[w] = merge(ids, pair, new_id)
                for p in zip(ids, ids[1:]):
                    counts[p] += f
                    where[p].add(w)
            del counts[pair], where[pair]

    def add_special_tokens(self, tokens: list[str]):
        for t in tokens:
            if t not in self.special:
                new_id = len(self.vocab)
                self.special[t] = new_id
                self.vocab[new_id] = t.encode("utf-8")

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge that is present (lowest new id).
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        if self.special:
            # Split special tokens out first so they are never broken into bytes.
            parts = re.split("(" + "|".join(re.escape(s) for s in self.special) + ")", text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special:
                out.append(self.special[part])
            else:
                for chunk in PAT.findall(part):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")   # a token can hold half a UTF-8 character

    # ------------------------------------------------------------ persistence
    def save(self, path: str):
        """Merges in learned order are enough to rebuild everything else."""
        obj = {"merges": [[a, b] for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
               "special": sorted(self.special, key=self.special.get)}
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "ByteBPE":
        with open(path) as f:
            obj = json.load(f)
        tok = cls()
        for i, (a, b) in enumerate(obj["merges"]):
            tok.merges[(a, b)] = 256 + i
            tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
        tok.add_special_tokens(obj["special"])
        return tok


if __name__ == "__main__":
    tok = ByteBPE()
    tok.train("the cat sat on the mat. the hat sat on the cat. " * 20, vocab_size=280, verbose=True)
    s = "the cat sat. naïve 東京 12345"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    print(len(s.encode()), "bytes ->", len(ids), "tokens")
