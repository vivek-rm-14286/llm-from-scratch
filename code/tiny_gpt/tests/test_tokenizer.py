"""Step 1: the byte-level BPE tokenizer."""
import bpe

CORPUS = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
          "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n"
          "the cat sat on the mat. the hat sat on the cat. ") * 30

SAMPLES = [
    "",
    "hello world",
    "First Citizen:\nSpeak, speak.\n",
    "  leading and trailing spaces   ",
    "tabs\tand\n\nnewlines\r\n",
    "naïve café 東京 🙂 12345 don't it's",
    "zzzz qqqq unseen xylophone",
]


def trained(vocab_size=320):
    tok = bpe.ByteBPE()
    tok.train(CORPUS, vocab_size)
    return tok


def test_first_merge_is_most_frequent_pair():
    tok = bpe.ByteBPE()
    tok.train("aaabdaaabac", 257)            # one chunk; (a, a) occurs 4 times, more than any other pair
    assert tok.merges == {(97, 97): 256}
    assert tok.vocab[256] == b"aa"


def test_vocab_size_and_merge_order():
    tok = trained(320)
    assert len(tok.vocab) == 320
    assert sorted(tok.merges.values()) == list(range(256, 320))
    for (a, b), new_id in tok.merges.items():
        assert a < new_id and b < new_id      # a merge only uses ids that already existed
        assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]


def test_round_trip():
    tok = trained()
    for s in SAMPLES + [CORPUS[:500]]:
        ids = tok.encode(s)
        assert all(isinstance(i, int) and 0 <= i < len(tok.vocab) for i in ids)
        assert tok.decode(ids) == s


def test_untrained_tokenizer_is_bytes():
    tok = bpe.ByteBPE()
    s = "naïve 東京"
    assert tok.encode(s) == list(s.encode("utf-8"))


def test_compresses_training_text():
    tok = trained()
    n_bytes = len(CORPUS.encode("utf-8"))
    assert len(tok.encode(CORPUS)) < 0.5 * n_bytes


def test_merges_do_not_cross_chunk_boundaries():
    tok = trained()
    for token_bytes in tok.vocab.values():
        text = token_bytes.decode("utf-8", errors="replace")
        assert " " not in text[1:], f"token {token_bytes!r} spans a word boundary"


def test_special_tokens():
    tok = trained()
    tok.add_special_tokens(["<|endoftext|>"])
    eot = tok.special["<|endoftext|>"]
    assert eot == 320 and len(tok.vocab) == 321
    s = "the cat<|endoftext|>the hat"
    ids = tok.encode(s)
    assert ids.count(eot) == 1
    assert tok.decode(ids) == s


def test_save_load(tmp_path):
    tok = trained()
    tok.add_special_tokens(["<|endoftext|>"])
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = bpe.ByteBPE.load(path)
    assert tok2.merges == tok.merges
    assert tok2.vocab == tok.vocab
    for s in SAMPLES + ["a<|endoftext|>b"]:
        assert tok2.encode(s) == tok.encode(s)
        assert tok2.decode(tok2.encode(s)) == s
