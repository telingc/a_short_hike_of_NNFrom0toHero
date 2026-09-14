"""Black-box visualization helpers for the exploratory notebook.

These functions deliberately do not import the notebook's model or optimizer
definitions.  Dependencies that live in the notebook are passed in explicitly:

* ``optimizers``: a mapping from optimizer name to ``factory(params, lr)``.
* ``make_model``: a model factory for the pre-activation distribution plot.
* ``builders``: PyTorch model builders used by the optional live demo.
"""

import math
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, clear_output, display
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _toy_loss(theta):
    x, y = theta[0], theta[1]
    return x**2 + 10 * y**2


def _toy_grad(theta):
    x, y = theta[0], theta[1]
    return torch.tensor([2 * x, 20 * y], dtype=theta.dtype)


def make_landscape(xlim=(-4, 4), ylim=(-3, 3), n=200):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10 * Y**2
    return X, Y, Z


def run_optimizer(opt_name, lr, optimizers, steps=80, theta0=(-3.5, -2.0)):
    if opt_name not in optimizers:
        raise ValueError(f"unknown optimizer: {opt_name}")

    param = torch.tensor(theta0, dtype=torch.float32)
    param.grad = None
    opt = optimizers[opt_name]([param], lr=lr)

    path = [param.detach().clone().numpy()]
    losses = [float(_toy_loss(param))]

    for _ in range(steps):
        opt.zero_grad()
        param.grad = _toy_grad(param)
        opt.step()
        path.append(param.detach().clone().numpy())
        losses.append(float(_toy_loss(param)))

    return np.array(path), losses


def plot_contours_subplots(
    configs,
    optimizers,
    steps=80,
    theta0=(-3.5, -2.0),
    xlim=(-4, 4),
    ylim=(-3, 3),
    figsize=(10, 8),
):
    X, Y, Z = make_landscape(xlim, ylim)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (name, lr) in zip(axes, configs):
        path, _ = run_optimizer(name, lr, optimizers, steps=steps, theta0=theta0)
        cs = ax.contour(X, Y, Z, levels=30, cmap="coolwarm")
        ax.clabel(cs, inline=True, fontsize=7)
        ax.plot(path[:, 0], path[:, 1], "r-o", markersize=2, linewidth=1)
        ax.scatter(path[0, 0], path[0, 1], c="k", s=35, zorder=5, label="start")
        ax.scatter(0, 0, c="gold", marker="*", s=100, zorder=6, label="min")
        ax.set_title(f"{name}  lr={lr}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.supxlabel(r"$\theta_1$")
    fig.supylabel(r"$\theta_2$")
    fig.suptitle("toy model: loss contour & optimizer path", y=1.01)
    fig.tight_layout()
    return fig


def plot_surfaces_subplots(
    configs,
    optimizers,
    steps=80,
    theta0=(-3.5, -2.0),
    xlim=(-4, 4),
    ylim=(-3, 3),
    figsize=(11, 9),
    surface_alpha=0.40,
):
    X, Y, Z = make_landscape(xlim, ylim)
    fig = plt.figure(figsize=figsize)

    for i, (name, lr) in enumerate(configs, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.plot_surface(
            X, Y, Z, cmap="coolwarm", alpha=surface_alpha, linewidth=0, antialiased=True
        )

        path, _ = run_optimizer(name, lr, optimizers, steps=steps, theta0=theta0)
        pz = path[:, 0] ** 2 + 10 * path[:, 1] ** 2
        ax.plot(path[:, 0], path[:, 1], pz, "r-o", markersize=2, linewidth=1)
        ax.scatter(path[0, 0], path[0, 1], pz[0], c="k", s=30, label="start")
        ax.scatter(0, 0, 0, c="gold", marker="*", s=80, label="min")

        ax.set_title(f"{name}  lr={lr}")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_zlabel(r"$L$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.suptitle("toy model: 3d loss landscape & optimizer path", y=0.98)
    fig.tight_layout()
    return fig


def plot_loss_curves_subplots(
    configs,
    optimizers,
    steps=80,
    theta0=(-3.5, -2.0),
    figsize=(10, 6),
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.ravel()

    for ax, (name, lr) in zip(axes, configs):
        _, losses = run_optimizer(name, lr, optimizers, steps=steps, theta0=theta0)
        ax.plot(losses)
        ax.set_title(f"{name}  lr={lr}")
        ax.set_ylabel("L")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    axes[-2].set_xlabel("step")
    fig.suptitle("toy model: loss value - steps")
    fig.tight_layout()
    return fig


def plot_preact_four_gains(
    x,
    make_model,
    n_hidd=256,
    n_hidden_layers=3,
    fan_out=10,
    bins=80,
    xlim=(-4.0, 4.0),
    figsize=(16, 4),
):
    fan_in = x.shape[1]

    def init_with_gain(model, gain):
        for layer in model.layers:
            if not hasattr(layer, "weight"):
                continue
            fin = layer.weight.shape[0]
            bound = gain * math.sqrt(3.0 / fin)
            layer.weight.uniform_(-bound, bound)
            if layer.bias is not None:
                layer.bias.zero_()

    def hidden_preacts(model, x):
        preacts = []
        h = x
        for layer in model.layers:
            if hasattr(layer, "weight"):
                z = layer(h)
                preacts.append(z.detach())
                h = z
            else:
                h = layer(h)
        return preacts[:-1]

    def hist_line(t):
        t = t.detach().flatten().float().cpu()
        t = t[(t >= xlim[0]) & (t <= xlim[1])]
        if t.numel() < 10:
            return [xlim[0], xlim[1]], [0.0, 0.0], float("nan"), float("nan")
        hy, hx = torch.histogram(t, bins=bins, range=xlim, density=True)
        return hx[:-1].tolist(), hy.tolist(), float(t.mean()), float(t.std())

    setups = [
        (0.2, "too small gain (collapse)"),
        (1.0, "gain=1, no ReLU √2 (mild drift)"),
        (4.0, "too large gain (explosion)"),
        (math.sqrt(2.0), "Kaiming gain=√2 (stable)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    for ax, (gain, title) in zip(axes, setups):
        model = make_model(fan_in, n_hidd, n_hidden_layers, fan_out)
        init_with_gain(model, gain=gain)
        preacts = hidden_preacts(model, x)

        for i, z in enumerate(preacts):
            hx, hy, mean, std = hist_line(z)
            ax.plot(hx, hy, lw=1.5, label=f"L{i+1} std={std:.3g}")
            print(f"{title:40s} | L{i+1} mean={mean:+.4f} std={std:.4e}")

        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pre-activation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("density")
    fig.suptitle(
        f"Hidden pre-activations in [{xlim[0]}, {xlim[1]}] (before ReLU)",
        y=1.03,
    )
    fig.tight_layout()
    return fig


def plot_table(results, ax, title, highlight=True):
    """Draw one hyperparameter sweep as a matplotlib table."""
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=12)

    headers = [
        "n_layers",
        "n_hidd",
        "batch_size",
        "lr",
        "loss",
        "std",
        "train",
        "val",
        "gap",
        "sec",
    ]
    rows = [
        [
            f"{r['n_layers']}",
            f"{r['n_hidd']}",
            f"{r['batch_size']}",
            f"{r['lr']:g}",
            f"{r['loss_last50']:.4f}",
            f"{r['loss_std50']:.4f}",
            f"{r['train_acc']:.4f}",
            f"{r['val_acc']:.4f}",
            f"{r['train_acc'] - r['val_acc']:.4f}",
            f"{r['seconds']:.1f}",
        ]
        for r in results
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#d9d9d9")
        cell.set_text_props(weight="bold")

    for i in range(1, len(rows) + 1):
        table[i, 7].set_text_props(weight="bold")

    if highlight and results:
        best = max(range(len(results)), key=lambda i: results[i]["val_acc"])
        for j in range(len(headers)):
            table[best + 1, j].set_facecolor("#e6f4ea")

    return table


@dataclass
class InferenceContext:
    models: dict
    device: str
    temp: dict
    test_set: object
    classes: list
    mean: torch.Tensor
    std: torch.Tensor
    ymax: float = 1.0


def load_comparison_context(
    builders,
    weights_dir="weights",
    data_root="../assignment_1_1/datasets",
    device=None,
    temp=None,
    ymax=1.0,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
):
    weights = Path(weights_dir)
    if not weights.exists():
        raise FileNotFoundError(f"weights folder not found: {weights.resolve()}")
    print("weights folder:", weights.resolve())

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    temp = temp or {"mlp": 1.0, "cnn": 1.0}

    models = {}
    for name in ("mlp", "cnn"):
        checkpoint = torch.load(
            weights / f"{name}.pt", map_location="cpu", weights_only=True
        )
        model = builders[checkpoint["arch"]["kind"]](checkpoint["arch"])
        model.load_state_dict(checkpoint["state_dict"])
        models[name] = {
            "model": model.eval().to(device),
            "test_acc": checkpoint["metrics"]["test_acc"],
            "params": checkpoint["metrics"]["n_params"],
        }
        print(
            f"{name}: {models[name]['params']:,} params, "
            f"test acc {models[name]['test_acc']:.4f}"
        )

    test_set = datasets.CIFAR10(
        data_root,
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return InferenceContext(
        models=models,
        device=device,
        temp=temp,
        test_set=test_set,
        classes=test_set.classes,
        mean=mean_tensor,
        std=std_tensor,
        ymax=ymax,
    )


@torch.no_grad()
def predict_all(ctx: InferenceContext, x):
    out = {}
    for name, rec in ctx.models.items():
        logits = rec["model"](x.to(ctx.device))[0]
        out[name] = (logits / ctx.temp[name]).softmax(0).cpu()
    return out


def to_display(ctx: InferenceContext, img):
    return (img * ctx.std + ctx.mean).clamp(0, 1).permute(1, 2, 0).numpy()


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def build_figure(ctx: InferenceContext, idx=None, seed=None):
    if idx is None:
        idx = random.Random(seed).randrange(len(ctx.test_set))
    img, label = ctx.test_set[idx]
    probs = predict_all(ctx, img.unsqueeze(0))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.3),
        gridspec_kw={"width_ratios": [1.05, 1.5, 1.5]},
    )
    axes[0].imshow(to_display(ctx, img), interpolation="nearest")
    axes[0].set_title(
        f"random test image #{idx}\ntrue label: {ctx.classes[label]}",
        fontsize=10,
    )
    axes[0].axis("off")

    bars, heights, info = {}, {}, {"idx": idx, "label": label}
    for ax, name in zip(axes[1:], ("mlp", "cnn")):
        p = probs[name]
        order = torch.argsort(p, stable=True)
        h = p[order].numpy()
        pos_true = int((order == label).nonzero())
        correct = int(order[-1]) == label
        heights[name] = h

        bars[name] = ax.bar(range(10), np.zeros(10), color="#8ab4f8", edgecolor="none")
        bars[name][pos_true].set_edgecolor("#202124")
        bars[name][pos_true].set_linewidth(2.0)
        bars[name][-1].set_color("#34a853" if correct else "#ea4335")
        ax.set_xticks(range(10))
        ax.set_xticklabels(
            [ctx.classes[int(i)] for i in order],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0, ctx.ymax if ctx.ymax else max(h.max() * 1.18, 0.05))
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(
            f"{name.upper()}  ({ctx.models[name]['params']:,} params, "
            f"acc {ctx.models[name]['test_acc']:.3f})\n"
            f"top-1: {ctx.classes[int(order[-1])]} ({h[-1]:.3f}) "
            f"{'correct' if correct else 'wrong'}",
            fontsize=10,
        )
        info[name] = {
            "top1": ctx.classes[int(order[-1])],
            "correct": correct,
            "visible_bars": int((h > 0.01).sum()),
        }

    axes[1].set_ylabel(f"probability  (softmax(logits / {ctx.temp['mlp']:g}))")
    fig.suptitle(
        "same image, same axis: raw probabilities from both models "
        "(ringed bar = true class)",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, bars, heights, info


def frame_png(fig, dpi=100):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def cascade_start(heights_by_model, threshold=0.002):
    visible = [
        i
        for h in heights_by_model.values()
        for i, v in enumerate(h)
        if v > threshold
    ]
    return (min(visible) / 10.0) if visible else 0.0


def bar_step(bars, heights, progress):
    for name, group in bars.items():
        for i, bar in enumerate(group):
            bar.set_height(heights[name][i] * smoothstep(progress * 10 - i))


def run_live_demo(
    ctx: InferenceContext,
    n_images=10,
    frame_ms=110,
    rise_frames=24,
    hold_ms=2600,
    dpi=90,
):
    rng = random.Random(None)
    log = []

    for k in range(n_images):
        fig, bars, heights, info = build_figure(
            ctx, idx=rng.randrange(len(ctx.test_set))
        )
        start = cascade_start(heights)

        for frame in range(rise_frames + 1):
            t0 = time.time()
            bar_step(
                bars,
                heights,
                start + (1 - start) * frame / rise_frames,
            )

            clear_output(wait=True)
            display(Image(data=frame_png(fig, dpi=dpi)))

            wait = (frame_ms if frame < rise_frames else hold_ms) / 1000
            elapsed = time.time() - t0
            if elapsed < wait:
                time.sleep(wait - elapsed)

        plt.close(fig)

        log.append(
            f"[{k + 1}/{n_images}] image #{info['idx']}  "
            f"true={ctx.classes[info['label']]:<10} "
            f"| MLP top1={info['mlp']['top1']:<10} ok={info['mlp']['correct']} "
            f"bars>0.01={info['mlp']['visible_bars']} "
            f"| CNN top1={info['cnn']['top1']:<10} ok={info['cnn']['correct']} "
            f"bars>0.01={info['cnn']['visible_bars']}"
        )

    return log
