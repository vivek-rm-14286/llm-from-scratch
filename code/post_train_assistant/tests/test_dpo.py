import math

import pytest
import torch

PAIRS = [
    {"messages": [{"role": "user", "content": "Name a color."}], "chosen": "Blue.", "rejected": "Blue blue blue blue"},
    {"messages": [{"role": "user", "content": "2+2=?"}], "chosen": "4", "rejected": "It is 5, I think."},
    {"messages": [{"role": "user", "content": "Say hi."}], "chosen": "Hi!", "rejected": "No no no no no"},
    {"messages": [{"role": "user", "content": "Capital of France?"}], "chosen": "Paris.", "rejected": "France France France"},
]


def hand_sequence_logprob(model, ids, labels):
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids).logits.double(), dim=-1)
    out = []
    for b in range(ids.size(0)):
        out.append(sum(logp[b, t - 1, labels[b, t]].item()
                       for t in range(1, ids.size(1)) if labels[b, t] != -100))
    return out


def make_examples(impl, fns):
    encode, im_start, im_end, nl, _ = fns
    return [impl.dpo.build_dpo_example(p, encode, im_start, im_end, nl) for p in PAIRS]


def test_build_dpo_example_masks_prompt(impl, tok, fns):
    encode, im_start, im_end, nl, _ = fns
    (c_ids, c_lab), (r_ids, r_lab) = impl.dpo.build_dpo_example(PAIRS[0], encode, im_start, im_end, nl)
    assert tok.decode(c_ids) == "<|im_start|>user\nName a color.<|im_end|>\n<|im_start|>assistant\nBlue.<|im_end|>\n"
    n_prompt = len(impl.data.render_prompt(PAIRS[0]["messages"], encode, im_start, im_end, nl))
    assert c_ids[:n_prompt] == r_ids[:n_prompt]                      # same prompt tokens
    assert all(l == -100 for l in c_lab[:n_prompt] + r_lab[:n_prompt])
    assert [l for l in c_lab if l != -100] == encode("Blue.") + [im_end]
    assert [l for l in r_lab if l != -100] == encode("Blue blue blue blue") + [im_end]


def test_dpo_batch_layout(impl, fns):
    examples = make_examples(impl, fns)
    ids, labels = impl.dpo.dpo_batch(examples, pad_id=0, max_len=128)
    assert ids.shape == labels.shape and ids.size(0) == 8
    for b, ex in enumerate(examples):
        assert ids[b, : len(ex[0][0])].tolist() == ex[0][0]          # chosen rows first
        assert ids[4 + b, : len(ex[1][0])].tolist() == ex[1][0]      # then rejected rows


def test_sequence_logprobs_equals_hand_computation(impl, tiny_model, fns):
    examples = make_examples(impl, fns)
    ids, labels = impl.dpo.dpo_batch(examples, pad_id=0, max_len=128)
    with torch.no_grad():
        got = impl.dpo.sequence_logprobs(tiny_model, ids, labels)
    assert got.shape == (8,)
    assert got.tolist() == pytest.approx(hand_sequence_logprob(tiny_model, ids, labels), abs=1e-3)
    # padding must not change a row's log-prob
    with torch.no_grad():
        alone = impl.dpo.sequence_logprobs(tiny_model, *impl.data.collate([examples[1][0]], 0, 128))
    assert alone.item() == pytest.approx(got[1].item(), abs=1e-3)


def test_dpo_loss_closed_form(impl):
    pi_w, pi_l = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -15.0])
    ref_w, ref_l = torch.tensor([-11.0, -18.0]), torch.tensor([-11.0, -16.0])
    loss, stats = impl.dpo.dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5)
    # margins: 0.5*((1) - (-1)) = 1.0 and 0.5*((-2) - (1)) = -1.5
    expected = (math.log(1 + math.exp(-1.0)) + math.log(1 + math.exp(1.5))) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert stats["reward_acc"] == pytest.approx(0.5)
    assert stats["margin"] == pytest.approx(-0.25)
    assert stats["chosen_reward"] == pytest.approx(-0.25)
    assert stats["rejected_reward"] == pytest.approx(0.0)
    assert all(isinstance(v, float) for v in stats.values())


def test_dpo_loss_is_ln2_when_policy_equals_reference(impl, tiny_model, fns):
    examples = make_examples(impl, fns)
    ref = impl.dpo.freeze_copy(tiny_model)
    ids, labels = impl.dpo.dpo_batch(examples, pad_id=0, max_len=128)
    loss, stats = impl.dpo.dpo_forward(tiny_model, ref, ids, labels, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)
    assert stats["margin"] == pytest.approx(0.0, abs=1e-5)
    assert loss.requires_grad


def test_dpo_loss_gradient_direction(impl):
    pi_w = torch.tensor([-5.0], requires_grad=True)
    pi_l = torch.tensor([-5.0], requires_grad=True)
    loss, _ = impl.dpo.dpo_loss(pi_w, pi_l, torch.tensor([-5.0]), torch.tensor([-5.0]), beta=0.2)
    loss.backward()
    assert pi_w.grad.item() == pytest.approx(-0.1)     # -beta * sigmoid(0): raise chosen
    assert pi_l.grad.item() == pytest.approx(0.1)      # lower rejected


def test_freeze_copy(impl, tiny_model):
    ref = impl.dpo.freeze_copy(tiny_model)
    assert not ref.training and not any(p.requires_grad for p in ref.parameters())
    assert all(p.requires_grad for p in tiny_model.parameters())
    assert ref.get_input_embeddings().weight is not tiny_model.get_input_embeddings().weight
    assert torch.equal(ref.get_input_embeddings().weight, tiny_model.get_input_embeddings().weight)


def test_train_dpo_moves_policy_not_reference(impl, tiny_model, fns):
    examples = make_examples(impl, fns)
    ref = impl.dpo.freeze_copy(tiny_model)
    ref_before = [p.detach().clone() for p in ref.parameters()]
    hist = impl.dpo.train_dpo(tiny_model, ref, examples, examples[:2], pad_id=0, max_len=128,
                              beta=0.5, lr=1e-3, batch_size=4, epochs=15, warmup_steps=1,
                              log_every=5, device="cpu", log=lambda s: None)
    assert hist[0]["step"] == 0
    assert hist[0]["eval_loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert hist[-1]["step"] == 15
    assert hist[-1]["reward_acc"] == 1.0 and hist[-1]["margin"] > 0.5
    assert hist[-1]["loss"] < 0.5 and hist[-1]["eval_loss"] < 0.5
    assert all(torch.equal(a, b) for a, b in zip(ref_before, ref.parameters()))
