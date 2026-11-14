"""Compare the finite actions at fixed training states using the stored final profile."""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

P=Path("figures/conditional_means_v1")
sys.path.insert(0,str(P.resolve()))
from predict import Profile
fit=Profile(P)
actions=sorted(fit.demand)
N=len(actions)
def read(file):
    with (P/file).open() as f: return list(csv.DictReader(f))
cfg,=[r for r in read("phase_selection.csv") if int(r["K"])==3 and int(r["horizon"])==10]
beta=[float(b) for b in cfg["beta"].split(";")]
data=np.full((N,3,10,3),np.nan)
with open("training/all_epochs.csv") as f:
    for row in csv.DictReader(f):
        block=int(row["block_id"])
        e=int(row["epoch_in_block"])
        if 16<=block<=18 and 1<=e<=10:
            b=actions.index(int(row["batch_sequences"]))
            assert np.isnan(data[b,block-16,e-1]).all() and row["train_energy_valid"]=="True"
            data[b,block-16,e-1]=[float(row[k]) for k in ("train_loss","train_seconds","train_energy_j")]
assert N==8 and np.isfinite(data).all()
J=np.minimum.accumulate(data[:,:,:,0],axis=2)/(data[:,:,:1,0]+1e-12)
x,y=J[:,:,:-1],J[:,:,1:]
delta=(y-x)/(x+1e-12)
ph=np.ones(x.shape,dtype=int)
ph[:,:,2:]=1+np.searchsorted(beta,(delta[:,:,:-2]+delta[:,:,1:-1])/2,side="left")
ph=np.maximum.accumulate(ph,axis=2)
for i,b in enumerate(actions):
    for z in (1,2,3):
        assert (ph[i]==z).sum()==int(fit.cells[b,z]["count"])
keys=["m_T_seconds","m_GPU_demand_joules","m_epsilon_seconds"]
observed=[data[:,:,1:,1],data[:,:,1:,2]/1000,data[:,:,1:,1]*y]
labels=["Epoch duration (s)","GPU demand (kJ)","Time-weighted loss (s)"]
titles=[r"$\widehat m_T(s,a)=\tau_a$",
        r"$\widehat m_D^{\rm GPU}(s,a)=d_{0a}+d_{1a}(J-1)$",
        r"$\widehat m_\varepsilon(s,a)=\tau_a\,J\,\rho_{a,\phi}$"]
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
                     "font.size":9.5,"pdf.fonttype":42,"svg.fonttype":"none",
                     "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.6,
                     "axes.edgecolor":"#738487","text.color":"#17323a","axes.labelcolor":"#17323a",
                     "xtick.color":"#4a6268","ytick.color":"#4a6268"})
def masks(z,j):
    support=np.array([float(fit.cells[b,z]["supported_J_min"])-1e-12<=j<=float(fit.cells[b,z]["supported_J_max"])+1e-12 for b in actions])
    width=1e-12 if z==1 else .008
    sample=(ph==z)&(np.abs(x-j)<=width)
    return support, sample
def anchor(z):
    if z==1:return 1.
    candidates=np.arange(830,1000)*.001
    def score(j):
        supported,obs=masks(z,j)
        return (supported.sum(),(supported & obs.any(axis=(1,2))).sum(),sum(obs[supported].ravel()))
    return max(candidates,key=score)
with PdfPages(P/"action_profiles.pdf") as pdf:
    for z in (1,2,3):
        j=anchor(z)
        support,sample=masks(z,j)
        if z==1: assert support.all() and sample.sum()==24
        fig,axs=plt.subplots(1,3,figsize=(12.3,4.6))
        fig.subplots_adjust(left=.068,right=.985,top=.72,bottom=.25,wspace=.42)
        fig.text(.068,.922,"Conditional means across batch-size actions",fontsize=17.5,weight="bold")
        fig.text(.068,.837,f"Fixed state:  J = {j:.3f},  phase = {z}     |     "+
                 ("Observations: first measured transition of each block (epoch 2)." if z==1 else
                  "Observations: same phase, within 0.008 of the indicated J."),fontsize=10)
        preds=np.full((N,3),np.nan)
        for i,b in enumerate(actions):
            if support[i]:
                p=fit(j,z,b)
                preds[i]=[p[keys[0]],p[keys[1]]/1000,p[keys[2]]]
        for m,ax in enumerate(axs):
            ax.plot(range(N),preds[:,m],color="#122e37",linestyle="--",lw=1.35,zorder=2)
            for i in range(N):
                if not support[i]:continue
                for t in range(3):
                    vals=observed[m][i,t][sample[i,t]]
                    ax.scatter(np.full(len(vals),i+[-.12,0,.12][t]),vals,marker="o",s=24,
                               facecolors="none",edgecolors="#228698",alpha=.78,lw=.95,zorder=3)
            ax.scatter(range(N),preds[:,m],c="#122e37",marker="D",s=34,zorder=4)
            ax.set(xlim=(-.5,N-.5),xticks=range(N),xticklabels=actions,xlabel="Action  a = B  (sequences / batch)",ylabel=labels[m])
            ax.set_title(titles[m],fontsize=12,pad=11,loc="left")
            ax.grid(axis="y",color="#dce4e5",lw=.55);ax.set_axisbelow(True)
        handles=[Line2D([],[],marker="o",mfc="none",mec="#228698",ls="",label="Observed epoch outcome"),
                 Line2D([],[],color="#122e37",marker="D",ls="--",label="Fitted value at each discrete action")]
        fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
        fig.text(.5,.033,f"Blocks 17–19, epochs 1–10 used for calibration.   Supported actions at this state: {int(support.sum())} / 8.",ha="center",color="#47636a",fontsize=8.5)
        pdf.savefig(fig)
        if z==1:
            fig.savefig(P/"action_profiles.png",dpi=300)
            fig.savefig(P/"action_profiles.svg")
        plt.close(fig)
        print(f"phase={z} J={j:.3f} supported={support.sum()} observations={sample[support].sum()} supported_actions={np.array(actions)[support]}")
