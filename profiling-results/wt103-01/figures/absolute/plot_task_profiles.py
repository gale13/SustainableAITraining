"""Plot each committed WT103 block as a task. Run with --help for input options."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"], "font.size": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.65, "text.color": "#193139", "axes.labelcolor": "#193139",
                     "xtick.color": "#40545c", "ytick.color": "#40545c", "legend.frameon": False})
METRICS = [
    ("normalized_loss", "Training loss", "Running minimum / first epoch", "Normalized training loss"),
    ("epoch_duration", "Training time", "Measured duration in seconds per epoch", "Training duration (s / epoch)"),
    ("epoch_energy", "GPU training energy", "Measured GPU-board energy in kilojoules per epoch", "Training energy (kJ / epoch)"),
]
FIELDS = ["batch_sequences", "block_id", "epoch_in_block", "train_loss", "train_seconds", "train_energy_j"]
EPS = 1e-12

def load_data(source, audit):
    opener = gzip.open if source.suffix == ".gz" else open
    assert audit["integrity_checks_passed"]
    groups = defaultdict(list)
    with opener(source, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            b, task, epoch = (int(row[k]) for k in FIELDS[:3])
            groups[b,task].append([epoch, *(float(row[k]) for k in FIELDS[3:])])
    batches = sorted({b for b,t in groups})
    tasks = range(audit["common_completed_blocks"])
    n = audit["epochs_per_block"]
    assert len(batches) == audit["batch_count"] and len(groups) == len(batches)*len(tasks)
    assert sum(map(len, groups.values())) == audit["row_count"]
    normalized, raw = {}, {}
    for b in batches:
        for task in tasks:
            data = np.array(sorted(groups[b,task]))
            assert np.array_equal(data[:,0], np.arange(1, n+1))
            values = data[:,1:]
            assert np.isfinite(values).all() and (values > 0).all()
            raw[b,task] = values
            result = values.copy()
            result[:,2] /= 1000  # J to kJ
            result[:,0] = np.minimum.accumulate(values[:,0]) / (values[0,0] + EPS)
            normalized[b,task] = result
    return batches, tasks, n, normalized, raw

def style(task, cut, count):
    if task < cut:
        return (plt.cm.viridis(0.13 + 0.55 * task / max(1,cut-1)), "-", 0.74)
    return (colors.to_rgba(["#c65931", "#aa3d20", "#7f281c"][(task-cut)%3]),
            ["--", "-.", ":"][(task-cut)%3], 0.97)

def plot_axis(ax, b, m, tasks, curve, n, cut, lim):
    for task in tasks:
        col, ls, alpha = style(task, cut, len(tasks))
        ax.plot(np.arange(1,n+1), curve[b,task][:,m], color=col, ls=ls, alpha=alpha,
                lw=0.75 if task < cut else 1, solid_capstyle="round")
    if m == 0:
        ax.axhline(1, color="#647177", lw=0.55, ls=(0,(1,3)), zorder=0)
    ax.set(xlim=(1,n), ylim=lim, xticks=[1, 50, 100, 150, n])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(length=2.5, width=0.6, labelsize=8)
    ax.grid(axis="y", color="#e1e6e6", linewidth=0.5)
    ax.set_axisbelow(True)

def legend(fig, tasks, cut):
    handles = []
    for t in tasks:
        c, ls, _ = style(t, cut, len(tasks))
        handles.append(Line2D([0],[0], color=c, ls=ls, lw=1.7, label=str(t+1)))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52,0.025),
               ncol=10, columnspacing=1.1, handlelength=1.8, handletextpad=.35,
               fontsize=8, title="Task (data block in arrival order)", title_fontsize=8)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit=json.loads(a.audit.read_text())
    cut=json.loads(a.recovery.read_text())["first_patched_block_id"]
    batches,tasks,n,curve,raw=load_data(a.source,audit)
    a.out.mkdir(parents=True,exist_ok=True)
    assert len(batches)==8
    lim=[]
    for m in range(3):
        vals=np.concatenate([z[:,m] for z in curve.values()])
        pad = max(.012, .045*np.ptp(vals))
        lim.append((max(0,vals.min()-pad),vals.max()+pad))
    note=f"Tasks 1–{cut}: original attention settings.   Tasks {cut+1}–{len(tasks)}: math attention backend."
    with PdfPages(a.out/"task_profiles_by_metric.pdf") as pdf:
        for m,(key,title,subtitle,ylabel) in enumerate(METRICS):
            fig,axs=plt.subplots(2,4,figsize=(11.3,6.65),sharex=True,sharey=True)
            fig.subplots_adjust(left=.08,right=.985,top=.825,bottom=.22,hspace=.32,wspace=.18)
            fig.text(.08,.937,title,fontsize=17,weight="bold")
            fig.text(.08,.891,subtitle+"   ·   "+note,fontsize=9)
            for ax,b in zip(axs.flat,batches):
                plot_axis(ax,b,m,tasks,curve,n,cut,lim[m])
                ax.set_title(f"Batch {b}",loc="left",fontsize=10,weight="bold",pad=5)
            fig.text(.022,.535,ylabel,rotation=90,va="center",fontsize=10)
            fig.text(.53,.144,"Epoch within task",ha="center",fontsize=10)
            legend(fig,tasks,cut)
            pdf.savefig(fig)
            fig.savefig(a.out/f"{key}.png",dpi=300)
            fig.savefig(a.out/f"{key}.svg")
            plt.close(fig)
    with PdfPages(a.out/"task_profiles_by_batch.pdf") as pdf:
        for b in batches:
            fig,axs=plt.subplots(1,3,figsize=(11.3,4.9))
            fig.subplots_adjust(left=.061,right=.984,top=.76,bottom=.33,wspace=.28)
            fig.text(.06,.91,f"Batch {b}",fontsize=18,weight="bold")
            fig.text(.06,.84,note,fontsize=9)
            for m,(key,title,subtitle,ylabel) in enumerate(METRICS):
                ax=axs[m]
                plot_axis(ax,b,m,tasks,curve,n,cut,lim[m])
                ax.set_title(title+"\n"+subtitle,fontsize=9,loc="left")
                ax.set_xlabel("Epoch within task")
            legend(fig,tasks,cut)
            pdf.savefig(fig)
            plt.close(fig)
    block_totals(a, batches, tasks, raw, n, cut)


def block_totals(args, batches, tasks, raw, n, cut):
    """Cross-check the sum of every epoch against the independently audited block table."""
    duration_s = np.array([[raw[b,t][:,1].sum() for t in tasks] for b in batches])
    energy_j = np.array([[raw[b,t][:,2].sum() for t in tasks] for b in batches])
    with (args.audit.parent / "block_summary.csv").open() as stream:
        check = {(int(r["batch_sequences"]), int(r["block_id"])): r for r in csv.DictReader(stream)}
    assert set(check) == set(raw)
    for i,b in enumerate(batches):
        for t in tasks:
            r = check[b,t]
            np.testing.assert_allclose([duration_s[i,t], energy_j[i,t]],
                                       [float(r["train_seconds"]), float(r["train_energy_j"])],
                                       rtol=1e-10, atol=1e-8)
    with (args.out/"block_totals.csv").open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_sequences","block_id","task_number","epochs","train_seconds","train_minutes","gpu_energy_j","gpu_energy_wh"])
        for i,b in enumerate(batches):
            for t in tasks:
                writer.writerow([b,t,t+1,n,duration_s[i,t],duration_s[i,t]/60,energy_j[i,t],energy_j[i,t]/3600])
    fig, axes = plt.subplots(2,1,figsize=(13.4,7.2))
    fig.subplots_adjust(left=.068,right=.986,bottom=.095,top=.855,hspace=.39)
    fig.text(.068,.942,"Training cost per data block",fontsize=18,weight="bold")
    fig.text(.068,.896,f"Each cell sums {n} measured training epochs on one GPU.  The divider marks the attention backend change.",fontsize=10)
    for ax,values,title,cmap in zip(axes,[duration_s/60,energy_j/3600],
                                      ["Training time (min per block)","GPU training energy (Wh per block)"],
                                      ["Blues","BuGn"]):
        im=ax.imshow(values,aspect="auto",cmap=cmap,interpolation="nearest")
        for i,b in enumerate(batches):
            for t in tasks:
                v=values[i,t]
                r,g,blue,_=im.cmap(im.norm(v))
                light=.2126*r+.7152*g+.0722*blue
                ax.text(t,i,f"{v:.1f}",fontsize=7.8,ha="center",va="center",
                        color="#11242b" if light>.53 else "white")
        ax.set(xticks=list(tasks),xticklabels=[str(t+1) for t in tasks],
               yticks=range(len(batches)),yticklabels=batches,ylabel="Batch size")
        ax.set_title(title,loc="left",fontsize=12,weight="bold",pad=6)
        ax.tick_params(length=0,pad=4)
        ax.axvline(cut-.5,color="#964f29",lw=1.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[1].set_xlabel("Task (data block in arrival order)",labelpad=7)
    fig.savefig(args.out/"block_totals.pdf")
    fig.savefig(args.out/"block_totals.svg")
    fig.savefig(args.out/"block_totals.png",dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
