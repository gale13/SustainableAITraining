"""Render one continuous, unsmoothed validation trajectory per trained model."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

root = Path("training")
out = Path("figures/raw_loss")
audit = json.loads((root / "audit.json").read_text())
assert audit["integrity_checks_passed"]
n = audit["common_completed_blocks"] * audit["epochs_per_block"]
series = defaultdict(list)
with (root / "all_epochs.csv").open() as stream:
    for r in csv.DictReader(stream):
        epoch = int(r["global_epoch"])
        assert epoch == int(r["block_id"])*audit["epochs_per_block"] + int(r["epoch_in_block"])
        series[int(r["batch_sequences"])].append((epoch, float(r["validation_loss"])))
assert len(series) == audit["batch_count"] and sum(map(len,series.values())) == audit["row_count"]
paths = {}
for b, pairs in series.items():
    x, y = np.array(sorted(pairs)).T
    assert np.array_equal(x, np.arange(1,n+1)) and np.isfinite(y).all()
    paths[b] = y
low = min(v.min() for v in paths.values())
high = max(v.max() for v in paths.values())
pad = .04 * (high-low)
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"pdf.fonttype":42,
                     "axes.spines.right":False,"axes.spines.top":False,"path.simplify":False})
out.mkdir(exist_ok=True, parents=True)
with PdfPages(out / "continuous_validation.pdf") as pdf:
    for b in sorted(paths):
        y = paths[b]
        fig, ax = plt.subplots(figsize=(11,4.25))
        fig.subplots_adjust(left=.077,right=.975,bottom=.17,top=.77)
        fig.text(.077,.903,"Validation loss over the full training run",fontsize=18,weight="bold",color="#163740")
        fig.text(.077,.822,f"Batch size {b}    ·    WikiText-103 validation    ·    {n:,} epoch measurements",fontsize=10,color="#3e535a")
        ax.plot(np.arange(1,n+1),y,color="#116c78",lw=.75)
        ax.set(xlim=(1,n),ylim=(low-pad,high+pad),xlabel="Global training epoch",
               ylabel="Validation cross-entropy",xticks=[1,500,1000,1500,2000,2500,3000,3500,4000,n])
        ax.grid(axis="y",color="#e0e8e8",lw=.55)
        ax.set_axisbelow(True)
        pdf.savefig(fig)
        plt.close(fig)
b4=paths[4]
print(f"batch4 epochs={n} first={b4[0]:.5f} final={b4[-1]:.5f} min={b4.min():.5f} min_epoch={b4.argmin()+1}")
print("batch4 epoch229 -> 230",b4[228:230])
