"""Step 1. Byte-level BPE tokenizer: train, encode, decode, save, load.

Grade with:  pytest tests/test_tokenizer.py        (from code/tiny_gpt)

Reference: the `ByteBPE` class in the Tokenization lesson. This file keeps the
same interface. You may start from the lesson's simple version and make it
faster afterwards; the tests only check behavior.
"""
import json
import re
from collections import Counter, defaultdict

# GPT-2-style pre-tokenization with the stdlib `re` module. Merges never cross
# a chunk boundary, so a token never spans two words. (Given: do not change.)
PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

    merge([1, 2, 3, 1, 2], (1, 2), 9) -> [9, 3, 9]
    merge([5, 5, 5], (5, 5), 9)       -> [9, 5]        (scan left to right)
    """
    # TODO: walk through `ids`; when ids[i], ids[i+1] == pair, emit new_id and skip two.
    raise NotImplementedError("bpe.merge")


class ByteBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}                    # (a, b) -> new id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}                               # text -> id
        self._cache: dict[str, list[int]] = {}                          # chunk -> ids (for encode)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Learn `vocab_size - 256` merges from `text`.

        After training: len(self.vocab) == vocab_size (unless the text runs out of pairs),
        self.merges maps each merged pair to its new id (256, 257, ... in learned order),
        and self.vocab[new_id] == self.vocab[a] + self.vocab[b].
        """
        assert vocab_size >= 256
        self.__init__()
        # TODO 1: split `text` with PAT.findall and count identical chunks ONCE (a Counter).
        #         Represent each unique chunk as a list of byte values plus its frequency.
        # TODO 2: repeat for new_id in range(256, vocab_size):
        #           - count adjacent pairs over all chunks, weighted by chunk frequency
        #           - pick the most frequent pair (break ties deterministically)
        #           - record it in self.merges and self.vocab
        #           - apply `merge` to the chunks
        #         Stop early if no pairs are left.
        # TODO 3 (speed, optional but worth it): do not recount everything after each merge.
        #         Keep `counts` and an index pair -> set of chunk numbers, and update only
        #         the chunks that contain the merged pair.
        raise NotImplementedError("ByteBPE.train")

    def add_special_tokens(self, tokens: list[str]):
        """Give each new special token the next free id. They are matched as whole strings in encode."""
        # TODO: for each token not already in self.special: id = len(self.vocab);
        #       store it in self.special and its UTF-8 bytes in self.vocab.
        raise NotImplementedError("ByteBPE.add_special_tokens")

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk. Cache the result in self._cache."""
        # TODO: start from the chunk's UTF-8 bytes. While there are at least two ids:
        #       find the adjacent pair with the LOWEST merge id (earliest learned);
        #       if no pair is in self.merges, stop; otherwise apply that merge.
        raise NotImplementedError("ByteBPE._encode_chunk")

    def encode(self, text: str) -> list[int]:
        """Text -> list of int ids. Special tokens become single ids."""
        # TODO: if there are special tokens, split `text` on them first with re.split and a
        #       capturing group (use re.escape). Then for each part: a special token maps to
        #       its id; anything else goes through PAT.findall and _encode_chunk.
        raise NotImplementedError("ByteBPE.encode")

    def decode(self, ids) -> str:
        """Ids -> text. Join the tokens' bytes, then decode UTF-8 with errors="replace"."""
        # TODO
        raise NotImplementedError("ByteBPE.decode")

    def save(self, path: str):
        """Write a JSON file from which `load` can rebuild the tokenizer exactly.
        The merges in learned order plus the special tokens are enough."""
        # TODO
        raise NotImplementedError("ByteBPE.save")

    @classmethod
    def load(cls, path: str) -> "ByteBPE":
        """Rebuild merges, vocab and special tokens from a file written by `save`."""
        # TODO
        raise NotImplementedError("ByteBPE.load")


if __name__ == "__main__":
    tok = ByteBPE()
    tok.train("the cat sat on the mat. the hat sat on the cat. " * 20, vocab_size=280, verbose=True)
    s = "the cat sat. naïve 東京 12345"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    print(len(s.encode()), "bytes ->", len(ids), "tokens")
