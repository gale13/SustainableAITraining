"""OR manuscript figure from verified decreasing-weight LYA / PV70 tables.

Structural adaptation: the daily panel uses grayscale energy and outcome symbols
instead of tiny numeric annotations. Every date and scatter point is retained.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pv70-figure-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pv70-figure-font-cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
OUTPUT = ROOT / "figures"
POLICIES = ["LYA", "B32", "B128", "B512", "STA", "INC", "DEC"]
LABELS = {p: p for p in POLICIES}
MARKERS = dict(zip(POLICIES, ["o", "s", "^", "D", "v", "P", "X"]))
FIGSIZE = (6.5, 5.8)  # Actual manuscript text width, not a scaled poster.
FONT_NAME = "Times New Roman"
findfont(FontProperties(family=FONT_NAME), fallback_to_default=False)
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT_NAME],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    "hatch.linewidth": .5,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data():
    hashes = json.loads((ROOT / "figure_data.json").read_text())["source_sha256"]
    for name in ("daily_tiles.csv", "daily_scatter.csv", "algorithm_summary.csv"):
        assert sha(SOURCE/name) == hashes[name]
    tiles = pd.read_csv(SOURCE / "daily_tiles.csv")
    daily = pd.read_csv(SOURCE / "daily_scatter.csv")
    summary = pd.read_csv(SOURCE / "algorithm_summary.csv").set_index("policy").loc[POLICIES]
    assert len(tiles) == 240 and len(daily) == 1680 and len(summary) == 7
    assert tiles.date.is_monotonic_increasing and tiles.date.nunique() == 240
    assert daily.groupby("policy").size().eq(240).all()
    assert tiles.outcome.isin(["win", "loss", "tie"]).all()
    expected = np.where(tiles.LYA_advantage_W > 1e-10, "win", np.where(tiles.LYA_advantage_W < -1e-10, "loss", "tie"))
    assert np.array_equal(tiles.outcome, expected)
    assert tiles.PV_generation_kWh.between(0,30).all()
    assert daily.DWAE.between(0,1).all() and daily.TARS_W.ge(0).all()
    assert np.allclose(daily.groupby("policy").TARS_W.mean().loc[POLICIES], summary.mean_TARS_W)
    return tiles,daily,summary


def tile_panel(fig, tiles):
    ax = fig.add_axes([.105,.200,.430,.693])
    norm = Normalize(0,30)
    cmap = plt.get_cmap("Greys")
    for row in tiles.itertuples():
        x,y = row.grid_column-1,row.grid_row-1
        ax.add_patch(Rectangle((x+.02,y+.025),.96,.95,facecolor=cmap(norm(row.PV_generation_kWh)),edgecolor="#b6b6b6",linewidth=.28))
        if row.outcome == "loss":
            # Double stroke keeps the symbol visible on both dark and light cells.
            ax.plot(x+.5,y+.5,marker="x",ms=5.5,mew=2.0,color="white",ls="none",zorder=3)
            ax.plot(x+.5,y+.5,marker="x",ms=5.5,mew=.85,color="black",ls="none",zorder=4)
        elif row.outcome == "tie":
            ax.plot(x+.5,y+.5,marker="o",ms=5.0,mew=2.0,mfc="none",mec="white",ls="none",zorder=3)
            ax.plot(x+.5,y+.5,marker="o",ms=5.0,mew=.85,mfc="none",mec="black",ls="none",zorder=4)
    rows = [0,4,8,12,16,19]
    ax.set(xlim=(0,12),ylim=(20,0),xticks=[],yticks=np.array(rows)+.5,
           yticklabels=[f"{12*r+1}–{12*r+12}" for r in rows])
    ax.tick_params(axis="y",length=0,pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.105,.963,"(a) Daily PV generation",fontsize=10)
    fig.text(.105,.922,"Chronological day index",fontsize=9)
    cax=fig.add_axes([.105,.148,.430,.016])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal",ticks=[0,10,20,30])
    cb.ax.tick_params(labelsize=9,length=2,pad=3)
    cb.outline.set_linewidth(.5)
    cb.set_label("Daily PV generation (kWh)",fontsize=9.5,labelpad=3)
    handles=[Line2D([],[],color="black",marker="x",ls="none",ms=5.5,label="Baseline better"),
             Line2D([],[],color="black",marker="o",mfc="none",ls="none",ms=5,label="Tie")]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.100,.038),ncol=2,
               frameon=False,handlelength=.8,handletextpad=.45,columnspacing=1.0,borderaxespad=0)
    fig.text(.105,.018,"Unmarked: LYA better",fontsize=9)


def bar_panel(fig, summary):
    ax=fig.add_axes([.700,.640,.280,.262])
    y=np.arange(7)
    bars=ax.barh(y,summary.mean_TARS_W,height=.60,
                 color=["#333333"]+["#c8c8c8"]*6,edgecolor="#444444",linewidth=.5,zorder=3)
    ax.errorbar(summary.mean_TARS_W,y,
                xerr=np.array([summary.mean_TARS_W-summary.TARS_ci_low_W,summary.TARS_ci_high_W-summary.mean_TARS_W]),
                fmt="none",ecolor="black",elinewidth=.7,capsize=2,zorder=4)
    for b,r in zip(bars,summary.itertuples()):
        if r.violations:
            b.set_hatch("///")
    ax.set(xlim=(0,620),ylim=(6.65,-.65),yticks=y,yticklabels=[LABELS[p] for p in POLICIES],xticks=[0,200,400,600])
    ax.tick_params(axis="y",length=0,pad=5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean time-averaged renewable\nshortfall, TARS (W)",labelpad=4)
    fig.text(.645,.963,"(b) Mean renewable shortfall",fontsize=10)
    fig.legend(handles=[Patch(facecolor="#c8c8c8",edgecolor="#444444",hatch="///",label="DWAE > 0.24")],
               loc="lower left",bbox_to_anchor=(.665,.505),frameon=False,borderaxespad=0,
               handlelength=1.4,handletextpad=.5)


def scatter_panel(fig,daily,summary):
    ax=fig.add_axes([.700,.160,.280,.285])
    for policy in POLICIES:
        d=daily.loc[daily.policy.eq(policy)]
        # Every daily point is drawn, with no subsampling or artificial jitter.
        ax.scatter(d.DWAE,d.TARS_W,marker=MARKERS[policy],s=7,
                   color="#777777" if policy=="LYA" else "#b0b0b0",alpha=.30,linewidths=0,zorder=2)
    offsets={"LYA":(-13,-15),"B32":(-17,9),"B128":(-37,-1),"B512":(-28,11),
             "STA":(7,-3),"INC":(-35,1),"DEC":(-16,-15)}
    for policy,r in summary.iterrows():
        ax.scatter(r.mean_DWAE,r.mean_TARS_W,marker=MARKERS[policy],s=32,
                   facecolor="#222222" if policy=="LYA" else "white",edgecolor="#222222",linewidth=.8,zorder=4)
        ax.annotate(LABELS[policy],(r.mean_DWAE,r.mean_TARS_W),xytext=offsets[policy],textcoords="offset points",fontsize=9,zorder=5,
                    bbox={"facecolor":"white","edgecolor":"none","pad":.35} if policy=="STA" else None)
    ax.axvline(.24,color="#666666",ls="--",lw=.7,zorder=1)
    xlow=min(.224,float(daily.DWAE.min())-.001)
    xhigh=max(.25,float(daily.DWAE.max())+.001)
    ax.set(xlim=(xlow,xhigh),ylim=(-15,max(650,float(daily.TARS_W.max())+15)),xticks=[.225,.235,.245],yticks=[0,200,400,600])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Duration-weighted average\nerror, DWAE",labelpad=4)
    ax.set_ylabel("Time-averaged renewable\nshortfall, TARS (W)",labelpad=5)
    fig.text(.645,.482,"(c) DWAE–TARS comparison",fontsize=10)


def main():
    tiles,daily,summary=read_data()
    fig=plt.figure(figsize=FIGSIZE)
    tile_panel(fig,tiles)
    bar_panel(fig,summary)
    scatter_panel(fig,daily,summary)
    fig.savefig(OUTPUT/"figure8_pv70_mean_observed.pdf")
    fig.savefig(OUTPUT/"figure8_pv70_mean_observed.svg")
    fig.savefig(OUTPUT/"figure8_pv70_mean_observed.png",dpi=300)
    fig.savefig(OUTPUT/"figure8_pv70_mean_observed.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    provenance={
        "style":"OR manuscript, grayscale, Times New Roman",
        "size_inches":list(FIGSIZE),"minimum_configured_font_pt":9,
        "tiles":240,"daily_scatter_points":1680,"summary_points":7,
        "numeric_cells_replaced_by":"PV energy gray level + negative/tie symbols; exact values remain in source tables",
        "outcome_counts":tiles.outcome.value_counts().to_dict(),
        "source_sha256":{name:sha(SOURCE/name) for name in ("daily_tiles.csv","daily_scatter.csv","algorithm_summary.csv")},
        "same_PV_setting_as_parent":True,"alpha":float(tiles.PV_scale.iloc[0]),
        "decreasing_weights":[100,10,1],"V":json.loads((ROOT/'results/selected.json').read_text())['mean70']['V'],"sigma_e":0.24,
        "decision_information":"MLP means and latest real observed PV power; no noise bank or future PV integration",
        "legend":"LYA is the paper's phase-aware renewal–Lyapunov policy, here using decreasing phase weights; DEC is a separate batch-schedule baseline",
        "terminology":{"DWAE":"Duration-weighted average error","TARS":"Time-averaged renewable shortfall"},
    }
    (ROOT/"figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":
    main()
