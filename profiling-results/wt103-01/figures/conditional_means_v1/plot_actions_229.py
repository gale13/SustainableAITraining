"""Run in results/wt103-01 after fit_profiles.py. Export a two-page figure."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("figures/conditional_means_v1")
BATCH = [4, 8, 10, 16, 24, 32, 48, 64]
H = 229
BLOCKS = [16, 17, 18]  # zero-based; common math attention backend
EPS = 1e-12
WIDTH = .008
with (ROOT / "phase_selection.csv").open() as f:
    grid = [r for r in csv.DictReader(f) if int(r["horizon"]) == H and int(r["eligible"])]
best = min(float(r["gain_rmse"]) for r in grid)
chosen = min((r for r in grid if float(r["gain_rmse"]) <= 1.05 * best), key=lambda r: int(r["K"]))
beta = [float(v) for v in chosen["beta"].split(";") if v]
assert int(chosen["K"]) == 3 and len(beta) == 2

data = np.full((len(BATCH), len(BLOCKS), H, 3), np.nan)
with open("training/all_epochs.csv") as f:
    for row in csv.DictReader(f):
        block = int(row["block_id"])
        if block not in BLOCKS:
            continue
        epoch = int(row["epoch_in_block"])
        assert 1 <= epoch <= H and int(row["global_epoch"]) == H * block + epoch
        dest = data[BATCH.index(int(row["batch_sequences"])), BLOCKS.index(block), epoch - 1]
        assert np.isnan(dest).all() and row["train_energy_valid"] == "True"
        dest[:] = [float(row[k]) for k in ("train_loss", "train_seconds", "train_energy_j")]
assert np.isfinite(data).all() and (data > 0).all()
loss, time, energy = (data[..., i] for i in range(3))
post = np.minimum.accumulate(loss, axis=2) / (loss[:, :, :1] + EPS)
pre, after = post[:, :, :-1], post[:, :, 1:]
change = (after - pre) / (pre + EPS)
phase = np.ones(pre.shape, dtype=int)
phase[:, :, 2:] += np.searchsorted(beta, (change[:, :, :-2] + change[:, :, 1:-1]) / 2)
phase = np.maximum.accumulate(phase, axis=2)
outcomes = [time, energy / 1000, time * post]


def refit(blocks):
    durations = time[:, blocks, :].mean(axis=(1, 2))
    demand = []
    ratios = []
    for i in range(len(BATCH)):
        j = pre[i, blocks]
        design = np.column_stack((np.ones(j.size), j.ravel() - 1))
        demand.append(np.linalg.lstsq(design, energy[i, blocks, 1:].ravel(), rcond=None)[0])
        p = phase[i, blocks]
        t = time[i, blocks, 1:]
        ratio = after[i, blocks] / j
        by_phase = [np.average(ratio[p == z], weights=t[p == z]) for z in range(1, 4)]
        ratios.append(by_phase)
    assert np.all((np.asarray(ratios) > 0) & (np.asarray(ratios) <= 1))
    return durations, np.asarray(demand), np.asarray(ratios)


def predict(params, i, j, z):
    t, d, rho = params
    return [t[i], (d[i, 0] + d[i, 1] * (j - 1)) / 1000, t[i] * j * rho[i, z - 1]]


params = refit(slice(None))
held = refit(slice(0, 2))
errors = []
for i in range(len(BATCH)):
    times, demands, penalties = predict(held, i, pre[i, 2], phase[i, 2])
    predictions = [times, demands, penalties]
    observed = [time[i, 2], outcomes[1][i, 2, 1:], outcomes[2][i, 2, 1:]]
    errors.append([np.sqrt(np.mean((p - v) ** 2)) / v.mean() for p, v in zip(predictions, observed)])


def select_state(z):
    active = phase == z
    limits = [(j[m].min(), j[m].max()) for j, m in zip(pre, active)]
    width = EPS if z == 1 else WIDTH

    def matched(j):
        samples = active & (np.abs(pre - j) <= width)
        support = np.array([lo - EPS <= j <= hi + EPS for lo, hi in limits]) & samples.any(axis=(1, 2))
        return samples & support[:, None, None], support

    if z == 1:
        return (1., *matched(1.))
    # Prioritize common action support and distinct blocks at those actions.
    # Resolve ties using the worst then total local sample count.
    def score(j):
        mask, support = matched(j)
        counts = mask.sum(axis=(1, 2))
        blocks = mask.any(axis=2).sum(axis=1)
        return support.sum(), blocks.min(), blocks.sum(), counts.min(), counts.sum()

    start, end = np.min(pre[active]), np.max(pre[active])
    values = np.arange(np.ceil(start * 1000), np.floor(end * 1000) + 1) / 1000
    j = max(values, key=score)
    return (j, *matched(j))


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#93a3a8", "axes.linewidth": .7,
                     "text.color": "#183339", "axes.labelcolor": "#183339"})
ink, teal = "#143842", "#248793"
positions = np.arange(len(BATCH))
rng = np.random.default_rng(229)
jitter = rng.uniform(-.19, .19, size=time.shape)
fields = ["Epoch duration", "GPU energy demand", "Time-weighted loss"]
units = ["Seconds per epoch", "kJ per epoch", r"$T_r\,J_{r+1}$  (s)"]
handles = [Line2D([], [], ls="", marker="o", mfc="none", mec=teal, label="Observed epoch"),
           Line2D([], [], ls="--", color=ink, marker="D", ms=5, label="Fit at each batch size")]


def panel(ax, m, fit, mask, aligned, show_x=True):
    offset = 1 if aligned else 0
    values = outcomes[m][:, :, offset:]
    for i in positions:
        idx = mask[i]
        ax.scatter(i + jitter[i, :, offset:][idx], values[i][idx], s=11,
                   facecolors="none", edgecolors=teal, linewidths=.5,
                   alpha=.28 if aligned else .15, rasterized=True)
    ax.plot(positions, fit, "--D", lw=1.35, ms=4.7, color=ink)
    ax.set(xticks=positions, xticklabels=BATCH, xlim=(-.52, 7.52), ylabel=units[m])
    if show_x:
        ax.set_xlabel("Batch size  $a$")
    ax.grid(axis="y", alpha=.4, lw=.55)
    ax.set_axisbelow(True)


out = ROOT / "action_profiles_229.pdf"
with PdfPages(out) as pdf:
    fig, axes = plt.subplots(3, 3, figsize=(12.7, 8.85), sharex=True, sharey="col")
    fig.subplots_adjust(left=.182, right=.984, bottom=.135, top=.830, hspace=.28, wspace=.34)
    fig.text(.045, .948, "Three conditional means across batch sizes", fontsize=20, weight="bold")
    fig.text(.045, .902, "WikiText-103   /   blocks 17–19   /   epochs 1–229   /   each row compares actions at one pre-epoch state", fontsize=11)
    for z in range(1, 4):
        j, mask, support = select_state(z)
        fitted = np.array([predict(params, i, j, z) for i in positions])
        fitted[~support] = np.nan
        for m, ax in enumerate(axes[z - 1]):
            panel(ax, m, fitted[:, m], mask, True, z == 3)
            if z == 1:
                ax.set_title(fields[m], loc="left", weight="bold", pad=12)
        row_center = axes[z - 1, 0].get_position().y0 + axes[z - 1, 0].get_position().height / 2
        fig.text(.029, row_center + .018, f"Phase {z}", fontsize=13, weight="bold")
        fig.text(.029, row_center - .014, f"$J={j:.3f}$", fontsize=11)
        fig.text(.029, row_center - .045, f"{mask.sum()} epochs", fontsize=9, color="#547078")
        print(f"Phase {z}  total={(phase == z).sum()}  J={j:.3f}  nearby={mask.sum()}  actions={support.sum()}/8")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .052), frameon=False, ncol=2)
    fig.text(.55, .032, "Circles match the indicated phase and J (exact in phase 1, within 0.008 in phases 2–3). Diamonds use the full-horizon fit.", ha="center", fontsize=9)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.85))
    fig.subplots_adjust(left=.073, right=.984, bottom=.26, top=.736, wspace=.32)
    fig.text(.05, .924, "Mean epoch outcomes over the full training trajectory", fontsize=18, weight="bold")
    fig.text(.05, .837, "WikiText-103   /   blocks 17–19   /   229 epochs per block   /   687 observations per batch size", fontsize=11)
    all_epochs = np.ones(time.shape, dtype=bool)
    for m, ax in enumerate(axes):
        marginal = outcomes[m].mean(axis=(1, 2))
        panel(ax, m, marginal, all_epochs, False)
        ax.set_title(fields[m], loc="left", weight="bold", pad=10)
        print(f"{fields[m]}  min={marginal.min():.4f} at {BATCH[marginal.argmin()]}  "
              f"max={marginal.max():.4f} at {BATCH[marginal.argmax()]}")
    fig.legend(handles=[handles[0], Line2D([], [], ls="--", color=ink, marker="D", ms=5,
                          label="Mean of all observed epochs at this batch size")],
               loc="lower center", bbox_to_anchor=(.5, .082), frameon=False, ncol=2)
    fig.text(.5, .05, "These means average over the training states visited by each fixed-batch trajectory.", ha="center", fontsize=9)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
print("Held-out block 19 macro NRMSE (%)  T / GPU demand / epsilon:",
      np.round(100 * np.mean(errors, axis=0), 3))
