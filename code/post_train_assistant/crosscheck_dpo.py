"""Step 6 (optional): cross-check the from-scratch DPO loss on one fixed batch.

Check 1 always runs: a hand computation with explicit Python loops over tokens.
Check 2 runs if `trl` is installed: TRL's DPOTrainer tokenizes the same three pairs
with a ChatML template, and its loss is compared with the from-scratch loss computed
on TRL's own input_ids and completion mask. Uses a tiny random model: no download.
"""
import argparse
import copy
import math

import torch

from common import add_common_args, load_impl
from conftest import make_char_tokenizer, make_tiny_model

PAIRS = [
    {"messages": [{"role": "user", "content": "Name a color."}], "chosen": "Blue.", "rejected": "Blue blue blue blue"},
    {"messages": [{"role": "user", "content": "2+2=?"}], "chosen": "4", "rejected": "It is 5, I think."},
    {"messages": [{"role": "user", "content": "Capital of France?"}], "chosen": "Paris.", "rejected": "France France France"},
]
CHATML_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                   "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
BETA = 0.1


def hand_dpo_loss(policy, ref, ids, labels, beta):
    """Explicit loops, float64. ids/labels: (2B, T), chosen rows first."""
    def seq_logps(model):
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids).logits.double(), dim=-1)
        return [sum(logp[b, t - 1, labels[b, t]].item()
                    for t in range(1, ids.size(1)) if labels[b, t] != -100)
                for b in range(ids.size(0))]
    pi, rf = seq_logps(policy), seq_logps(ref)
    B = ids.size(0) // 2
    losses = []
    for b in range(B):
        margin = beta * ((pi[b] - rf[b]) - (pi[B + b] - rf[B + b]))
        losses.append(math.log(1 + math.exp(-margin)))
    return sum(losses) / B


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    impl = load_impl(args.impl)
    tok = make_char_tokenizer()
    ref = make_tiny_model(len(tok), seed=0)
    policy = copy.deepcopy(ref)
    torch.manual_seed(1)
    with torch.no_grad():                       # perturb so policy != reference
        for p in policy.parameters():
            p.add_(0.05 * torch.randn_like(p))

    encode = lambda s: tok.encode(s, add_special_tokens=False)
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    examples = [impl.dpo.build_dpo_example(p, encode, im_start, im_end, encode("\n")) for p in PAIRS]
    ids, labels = impl.dpo.dpo_batch(examples, tok.pad_token_id, 128)
    with torch.no_grad():
        ours, stats = impl.dpo.dpo_forward(policy, impl.dpo.freeze_copy(ref), ids, labels, BETA)
    hand = hand_dpo_loss(policy, ref, ids, labels, BETA)
    print(f"from-scratch loss {ours.item():.6f}   hand computation {hand:.6f}   "
          f"|diff| {abs(ours.item() - hand):.2e}")
    assert abs(ours.item() - hand) < 1e-4, "from-scratch loss disagrees with the hand computation"

    try:
        import trl
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("trl not installed: skipping the TRL comparison (uv pip install trl)")
        return
    tok.chat_template = CHATML_TEMPLATE
    ds = Dataset.from_list([{"prompt": p["messages"],
                             "chosen": [{"role": "assistant", "content": p["chosen"]}],
                             "rejected": [{"role": "assistant", "content": p["rejected"]}]} for p in PAIRS])
    cfg = DPOConfig(output_dir="out/trl_crosscheck", beta=BETA, loss_type="sigmoid", report_to="none",
                    per_device_train_batch_size=len(PAIRS), max_length=128, use_cpu=True)
    trainer = DPOTrainer(model=policy, ref_model=copy.deepcopy(ref), args=cfg,
                         train_dataset=ds, processing_class=tok)
    batch = next(iter(trainer.get_train_dataloader()))
    policy.eval()
    with torch.no_grad():
        trl_loss = trainer.compute_loss(policy, batch).item()
        # the from-scratch loss on TRL's own tensors
        t_ids = batch["input_ids"]
        t_labels = t_ids.masked_fill(batch["completion_mask"] == 0, -100)
        ours_on_trl, _ = impl.dpo.dpo_forward(policy, impl.dpo.freeze_copy(ref), t_ids, t_labels, BETA)
    print(f"TRL {trl.__version__} DPOTrainer loss {trl_loss:.6f}   from-scratch on the same tensors "
          f"{ours_on_trl.item():.6f}   |diff| {abs(trl_loss - ours_on_trl.item()):.2e}")
    assert abs(trl_loss - ours_on_trl.item()) < 1e-4, "from-scratch loss disagrees with TRL"


if __name__ == "__main__":
    main()
