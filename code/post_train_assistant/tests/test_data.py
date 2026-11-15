import torch

from conftest import make_char_tokenizer, make_tiny_model


def test_ensure_chatml_tokens_adds_and_resizes(impl):
    tok = make_char_tokenizer(with_chatml=False)
    model = make_tiny_model(len(tok))
    old = model.get_input_embeddings().weight.detach().clone()
    n = len(tok)
    im_start, im_end = impl.data.ensure_chatml_tokens(tok, model)
    assert len(tok) == n + 2 and {im_start, im_end} == {n, n + 1}
    assert tok.encode("<|im_start|>user<|im_end|>", add_special_tokens=False)[0] == im_start
    assert len(tok.encode("<|im_end|>", add_special_tokens=False)) == 1
    emb = model.get_input_embeddings().weight
    assert emb.shape[0] == n + 2
    assert torch.equal(emb[:n], old)                               # old rows untouched
    assert torch.allclose(emb[n], old.mean(0), atol=1e-6)          # new rows = mean of old rows
    assert torch.allclose(emb[n + 1], old.mean(0), atol=1e-6)
    assert model(input_ids=torch.tensor([[im_start, 5, im_end]])).logits.shape == (1, 3, n + 2)


def test_ensure_chatml_tokens_is_noop_when_present(impl, tok, tiny_model):
    before = tiny_model.get_input_embeddings().weight.detach().clone()
    n = len(tok)
    ids = impl.data.ensure_chatml_tokens(tok, tiny_model)
    assert ids == (tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>"))
    assert len(tok) == n and torch.equal(tiny_model.get_input_embeddings().weight, before)


def test_render_chatml_known_example(impl, tok, fns, conv):
    encode, im_start, im_end, nl, _ = fns
    ids, labels = impl.data.render_chatml(conv, encode, im_start, im_end, nl)
    assert len(ids) == len(labels) == 56
    assert tok.decode(ids) == ("<|im_start|>user\n2+2=?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n"
                               "<|im_start|>user\n10+7=?<|im_end|>\n<|im_start|>assistant\n17<|im_end|>\n")
    target_pos = [i for i, l in enumerate(labels) if l != -100]
    assert target_pos == [24, 25, 52, 53, 54]                      # "4", <|im_end|>, "1", "7", <|im_end|>
    assert [labels[i] for i in target_pos] == encode("4") + [im_end] + encode("17") + [im_end]
    assert all(labels[i] == ids[i] for i in target_pos)


def test_render_chatml_masks_user_and_system(impl, fns):
    encode, im_start, im_end, nl, _ = fns
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    ids, labels = impl.data.render_chatml(msgs, encode, im_start, im_end, nl)
    assert ids.count(im_start) == 2 and ids.count(im_end) == 2
    assert all(l == -100 for l in labels)


def test_render_prompt_appends_generation_prompt(impl, tok, fns, conv):
    encode, im_start, im_end, nl, _ = fns
    ids, _ = impl.data.render_chatml(conv[:1], encode, im_start, im_end, nl)
    prompt = impl.data.render_prompt(conv[:1], encode, im_start, im_end, nl)
    assert prompt[: len(ids)] == ids
    assert tok.decode(prompt[len(ids):]) == "<|im_start|>assistant\n"


def test_collate_pads_and_truncates(impl):
    batch = [([5, 6, 7], [-100, 6, 7]), ([8, 9, 10, 11, 12, 13], [-100, -100, 10, 11, 12, 13])]
    x, y = impl.data.collate(batch, pad_id=0, max_len=5)
    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.tolist() == [[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]]
    assert y.tolist() == [[-100, 6, 7, -100, -100], [-100, -100, 10, 11, 12]]


def test_length_bucketed_batches(impl):
    import random
    rng = random.Random(1)
    lengths = [rng.randint(5, 200) for _ in range(203)]
    batches = impl.data.length_bucketed_batches(lengths, 8, bucket_mult=10, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(203))      # every index once
    assert all(1 <= len(b) <= 8 for b in batches) and len(batches) <= 28
    padded = lambda bs: sum(max(lengths[i] for i in b) * len(b) for b in bs)
    naive = [list(range(s, min(s + 8, 203))) for s in range(0, 203, 8)]
    assert padded(batches) < 0.8 * padded(naive)                          # much less padding
    assert batches == impl.data.length_bucketed_batches(lengths, 8, bucket_mult=10, seed=0)
    assert batches != impl.data.length_bucketed_batches(lengths, 8, bucket_mult=10, seed=1)
    firsts = [lengths[b[0]] for b in batches]
    assert firsts != sorted(firsts)                                       # batch order is shuffled
