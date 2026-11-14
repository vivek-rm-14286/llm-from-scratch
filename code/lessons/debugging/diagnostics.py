"""Small diagnostics used on the "Debugging a Training Run" page. Run this file for a demo on a toy MLP."""
import torch
import torch.nn as nn


def activation_stats(model, x, types=(nn.Linear, nn.LayerNorm)):
    """One forward pass; returns {module name: (mean, std, max |value|)} of each module's output."""
    stats, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, types):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, name=name: stats.__setitem__(
                    name, (out.mean().item(), out.std().item(), out.abs().max().item()))))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return stats


def update_ratios(model, optimizer):
    """Call INSTEAD of optimizer.step(). Returns {param name: ||update|| / ||weight||} for 2-D weights."""
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.ndim == 2}
    optimizer.step()
    return {n: ((p.detach() - before[n]).norm() / (before[n].norm() + 1e-12)).item()
            for n, p in model.named_parameters() if n in before}


def first_nonfinite(model, x):
    """Name of the first module whose output contains NaN or inf, or None."""
    found, hooks = [], []
    for name, m in model.named_modules():
        if not list(m.children()):                                  # leaf modules only
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, name=name: found.append(name)
                if not found and not torch.isfinite(out).all() else None))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return found[0] if found else None


if __name__ == "__main__":
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 10))
    x, y = torch.randn(16, 32), torch.randint(0, 10, (16,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nn.functional.cross_entropy(model(x), y).backward()
    for name, (mean, std, mx) in activation_stats(model, x).items():
        print(f"layer {name}: mean {mean:+.3f}  std {std:.3f}  max {mx:.2f}")
    for name, r in update_ratios(model, opt).items():
        print(f"{name}: update/weight {r:.1e}")
    with torch.no_grad():
        model[2].weight[0, 0] = float("nan")                        # plant a NaN in the second Linear
    print("first non-finite output:", first_nonfinite(model, x))
