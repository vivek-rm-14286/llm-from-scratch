"""Train a byte-level BPE tokenizer with Hugging Face `tokenizers`.

    # ~2 GB of FineWeb-Edu text is plenty for a 49k vocab
    python train_tokenizer.py --num_docs 1000000 --vocab_size 49152 --out tokenizer
    # or from local .txt files (one document per file or per blank-line block)
    python train_tokenizer.py --text_files my_corpus.txt --vocab_size 8192 --out tokenizer

Writes <out>/tokenizer.json. Special tokens get the lowest ids:
    0 <|endoftext|>   document separator (and pretraining EOS)
    1 <|im_start|>    chat turn start (used in SFT)
    2 <|im_end|>      chat turn end   (SFT EOS)
    3 <|pad|>
"""
import argparse
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def fineweb_iterator(num_docs, dataset, name):
    from datasets import load_dataset  # imported lazily: only needed for this path
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= num_docs:
            break
        yield ex["text"]


def file_iterator(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            buf = []
            for line in f:
                if line.strip() == "" and buf:
                    yield "".join(buf)
                    buf = []
                else:
                    buf.append(line)
            if buf:
                yield "".join(buf)


def train_tokenizer(texts, vocab_size, out_dir, length=None):
    tok = Tokenizer(models.BPE())
    # ByteLevel: every byte maps to one of 256 printable symbols, so there is
    # no unknown token ever. use_regex=True applies the GPT-2 split pattern
    # (letters / digits / punctuation / whitespace stay in separate pieces).
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    return tok, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=49152)
    ap.add_argument("--num_docs", type=int, default=1_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--text_files", nargs="*", help="train on local text instead")
    ap.add_argument("--out", default="tokenizer")
    args = ap.parse_args()

    if args.text_files:
        texts, length = file_iterator(args.text_files), None
    else:
        texts, length = fineweb_iterator(args.num_docs, args.dataset, args.name), args.num_docs
    tok, path = train_tokenizer(texts, args.vocab_size, args.out, length)
    print(f"saved {path}  (vocab {tok.get_vocab_size()})")
    sample = "The mitochondria is the powerhouse of the cell. 3.14159 def f(x): return x**2"
    enc = tok.encode(sample)
    print(f"{len(sample)} chars -> {len(enc.ids)} tokens: {enc.tokens[:20]}")
    assert tok.decode(enc.ids) == sample, "byte-level BPE must round-trip exactly"


if __name__ == "__main__":
    main()
