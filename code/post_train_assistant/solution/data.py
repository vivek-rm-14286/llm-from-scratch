"""ChatML rendering, loss masks, padding and length-bucketed batching.

Shapes follow the SFT lesson: a conversation becomes (input_ids, labels) where labels
are -100 everywhere except the assistant's answer tokens and its <|im_end|>.
"""
import random

import torch

IGNORE = -100
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def ensure_chatml_tokens(tokenizer, model):
    """Make <|im_start|> and <|im_end|> single tokens. Returns (im_start_id, im_end_id).

    If a marker is missing from the vocabulary, register it as a special token, resize
    the model's embedding matrix, and initialize each new row to the mean of the old
    rows (a far better start than random noise). A second call changes nothing.
    """
    vocab = tokenizer.get_vocab()
    missing = [t for t in (IM_START, IM_END) if t not in vocab]
    if missing:
        old_n = model.get_input_embeddings().weight.size(0)
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[old_n:] = emb[:old_n].mean(dim=0, keepdim=True)
            out = model.get_output_embeddings()
            if out is not None and out.weight is not emb:      # untied output head
                out.weight[old_n:] = out.weight[:old_n].mean(dim=0, keepdim=True)
    return tokenizer.convert_tokens_to_ids(IM_START), tokenizer.convert_tokens_to_ids(IM_END)


def render_chatml(messages, encode, im_start_id, im_end_id, nl_ids):
    """Tokenize a conversation turn by turn. Returns (input_ids, labels).

    encode: str -> list[int] for ordinary text (no special tokens added)
    nl_ids: token ids of "\n"
    Each turn is  <|im_start|> role \n content <|im_end|> \n . Only the content and
    <|im_end|> of assistant turns get real labels; everything else is IGNORE.
    """
    ids, labels = [], []
    for msg in messages:
        header = [im_start_id] + encode(msg["role"]) + nl_ids
        body = encode(msg["content"]) + [im_end_id]
        ids += header
        labels += [IGNORE] * len(header)
        ids += body
        labels += body if msg["role"] == "assistant" else [IGNORE] * len(body)
        ids += nl_ids
        labels += [IGNORE] * len(nl_ids)
    return ids, labels


def render_prompt(messages, encode, im_start_id, im_end_id, nl_ids):
    """Token ids of the conversation followed by the generation prompt
    <|im_start|> assistant \n , ready for the model to continue."""
    ids, _ = render_chatml(messages, encode, im_start_id, im_end_id, nl_ids)
    return ids + [im_start_id] + encode("assistant") + nl_ids


def collate(batch, pad_id, max_len):
    """Right-pad a list of (ids, labels) to one length, truncating to max_len.
    Returns LongTensors x, y of shape (B, T); padded label positions are IGNORE."""
    batch = [(i[:max_len], l[:max_len]) for i, l in batch]
    T = max(len(i) for i, _ in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for b, (i, l) in enumerate(batch):
        x[b, : len(i)] = torch.tensor(i, dtype=torch.long)
        y[b, : len(l)] = torch.tensor(l, dtype=torch.long)
    return x, y


def length_bucketed_batches(lengths, batch_size, *, bucket_mult=50, seed=0):
    """Group example indices into batches of similar length.

    Shuffle all indices, cut them into chunks of batch_size * bucket_mult, sort each
    chunk by length, slice it into batches, then shuffle the order of the batches.
    Every index appears exactly once. Returns a list of lists of indices.
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = batch_size * bucket_mult
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: lengths[i])
        batches += [part[b : b + batch_size] for b in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches
