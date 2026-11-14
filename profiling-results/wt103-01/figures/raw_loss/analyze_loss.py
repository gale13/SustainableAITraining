"""Describe raw loss within each committed block of the formal wt103 profiling run."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

train = Path("training")
out = Path("figures/raw_loss")
out.mkdir(parents=True, exist_ok=True)
audit = json.loads((train/"audit.json").read_text())
assert audit["integrity_checks_passed"] and audit["common_completed_blocks"] == 19
window = 20
tail = 46
n = audit["epochs_per_block"]
assert n == 229
series = defaultdict(list)
with (train/"all_epochs.csv").open() as f:
    for r in csv.DictReader(f):
        key = int(r["batch_sequences"]), 1 + int(r["block_id"])
        series[key].append([int(r["epoch_in_block"]), float(r["train_loss"]), float(r["validation_loss"])])
assert sum(map(len,series.values())) == audit["row_count"] == 34808
batches = sorted({b for b,t in series})
assert len(batches)==8 and len(series)==152
names = ["Training", "Validation"]
stats = []
for (b,t), rows in sorted(series.items()):
    values = np.asarray(sorted(rows))
    assert np.array_equal(values[:,0], np.arange(1,n+1)) and np.isfinite(values).all()
    series[b,t] = values[:,1:]
    for m,name in enumerate(names):
        v = values[:,m+1]
        early, prev, end = [np.mean(q) for q in (v[:window],v[-2*window:-window],v[-window:])]
        stepdiff = np.diff(v)
        stats.append(dict(batch=b,task=t,metric=name,first=v[0],last=v[-1],early=early,
                          final=end,overall=end-early,late=end-prev,slope=np.polyfit(np.arange(tail),v[-tail:],1)[0],
                          step_up=np.mean(stepdiff>0),late_step_up=np.mean(stepdiff[-(tail-1):]>0),
                          minimum_epoch=1+np.argmin(v),late10=v[-10:].mean()-v[-20:-10].mean(),
                          late30=v[-30:].mean()-v[-60:-30].mean()))
with (out/"loss_trends.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)
with (out/"findings.txt").open("w") as log:
    for name in names:
        s=[r for r in stats if r["metric"]==name]
        def col(k): return np.array([r[k] for r in s])
        def rng(v): return np.round(np.percentile(v,[0,50,100]),5)
        print(name,"overall-down",sum(col("overall")<0),"first-last-down",sum(col("last")<col("first")),
              "overall delta range/median",rng(col("overall")),file=log)
        print("late delta",rng(col("late")),"late up",sum(col("late")>0),"slope up",sum(col("slope")>0),
              "both",sum((col("slope")>0)&(col("late")>0)),"10/30 up",sum(col("late10")>0),sum(col("late30")>0),file=log)
        print("up steps pooled",round(col("step_up").mean(),4),"curves with uptick",sum(col("step_up")>0),
              "min epoch",rng(col("minimum_epoch")),"early and final median",np.median(col("early")),np.median(col("final")),file=log)
        for b in batches:
            r=[a for a in s if a["batch"]==b]
            print(b,"late up",sum(a["late"]>0 for a in r),"both",sum(a["late"]>0 and a["slope"]>0 for a in r),
                  "median late delta",round(np.median([a["late"] for a in r]),5),file=log)
        for r in sorted(s,key=lambda a:-a["late"])[:2]:
            print("highest late rise",r["batch"],r["task"],round(r["late"],5),"late means",round(r["final"]-r["late"],5),round(r["final"],5),file=log)

plt.rcParams.update({"font.size":9,"font.family":"sans-serif","pdf.fonttype":42,"svg.fonttype":"none",
                     "axes.spines.top":False,"axes.spines.right":False})
def lines(ax,b,m):
    stack=np.stack([series[b,t][:,m] for t in range(1,20)])
    x=np.arange(1,n+1)
    for j,v in enumerate(stack):
        ax.plot(x,v,color="#458495" if j<16 else "#bc613c",lw=.65,alpha=.53 if j<16 else .85)
    ax.plot(x,np.median(stack,axis=0),color="#1a3039",lw=1.65)
    ax.set(xlim=(1,n),xticks=[1,50,100,150,200,229])
    ax.grid(axis="y",lw=.4,color="#dce2e3")
    ax.set_axisbelow(True)
def legend(fig):
    defs=[("Tasks 1–16","#458495"),("Tasks 17–19 (after repair)","#bc613c"),("Median across 19 tasks","#1a3039")]
    fig.legend([Line2D([],[],color=c,lw=1.8) for _,c in defs],[l for l,_ in defs],
               loc="lower center",ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.52,.005))
with PdfPages(out/"raw_loss.pdf") as pdf:
    for m,name in enumerate(names):
        fig,axs=plt.subplots(2,4,figsize=(11.4,6.3),sharex=True,sharey=True)
        fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.83,wspace=.17,hspace=.34)
        fig.text(.075,.928,f"{name} loss across data blocks",fontsize=18,weight="bold")
        fig.text(.075,.872,"Raw cross-entropy at each epoch   |   19 sequential blocks per batch size",fontsize=10)
        for ax,b in zip(axs.flat,batches):
            lines(ax,b,m)
            ax.set_title(f"Batch {b}",loc="left",fontsize=10,weight="bold")
        fig.text(.021,.49,"Cross-entropy",rotation=90)
        fig.text(.52,.106,"Epoch within data block",ha="center")
        legend(fig);pdf.savefig(fig);plt.close(fig)
