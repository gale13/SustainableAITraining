"""One-epoch relative loss progress by discrete batch action, at matched pre-epoch states."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p=Path("figures/conditional_means_v1")
with (p/"phase_selection.csv").open() as f:
    row,=[r for r in csv.DictReader(f) if (int(r["horizon"]),int(r["K"]))==(10,3)]
beta=np.array([float(t) for t in row["beta"].split(";")])
actions=[4,8,10,16,24,32,48,64]
loss=np.full((8,3,10),np.nan)
with open("training/all_epochs.csv") as f:
    for r in csv.DictReader(f):
        t,e=int(r["block_id"]),int(r["epoch_in_block"])
        if 16<=t<=18 and 1<=e<=10:
            idx=actions.index(int(r["batch_sequences"])),t-16,e-1
            assert np.isnan(loss[idx]) and int(r["global_epoch"])==229*t+e
            loss[idx]=float(r["train_loss"])
assert np.isfinite(loss).all()
j=np.minimum.accumulate(loss,axis=2)/(loss[:,:,:1]+1e-12)
pre=j[:,:,:-1]
delta=np.diff(j,axis=2)/(pre+1e-12)
phase=np.ones(pre.shape,dtype=int)
phase[:,:,2:]=1+np.searchsorted(beta,(delta[:,:,:-2]+delta[:,:,1:-1])/2,side="left")
phase=np.maximum.accumulate(phase,axis=2)
gain=-100*delta
assert (gain>=0).all()
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
                     "font.size":10,"pdf.fonttype":42,"svg.fonttype":"none",
                     "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,
                     "text.color":"#1e363c","axes.edgecolor":"#718085"})
fig,axs=plt.subplots(1,3,figsize=(12,4.55),sharey=True)
fig.subplots_adjust(left=.075,right=.984,top=.735,bottom=.255,wspace=.20)
fig.text(.075,.917,"Loss improvement in one epoch by batch size",fontsize=18,weight="bold")
fig.text(.075,.837,"WikiText-103   /   blocks 17–19   /   matched pre-epoch states within the first 10 epochs",fontsize=10,color="#435e67")
for z,(a,tol) in enumerate([(1.,1e-12),(.901,.008),(.865,.008)],1):
    ax=axs[z-1]
    mask=(phase==z)&(abs(pre-a)<=tol)
    means=[];counts=[]
    for i,b in enumerate(actions):
        g=gain[i][mask[i]]; assert len(g)
        means.append(g.mean()); counts.append(len(g))
        for t in range(3):
            vals=gain[i,t][mask[i,t]]
            ax.scatter(np.full(len(vals),i+(t-1)*.105),vals,s=29,facecolors="none",
                       edgecolors="#187f8e",linewidths=.9,alpha=.85,zorder=3)
    ax.plot(range(8),means,ls="--",lw=1.15,color="#273b46",marker="D",ms=5.7,zorder=4)
    ax.set_title(f"Phase {z}   ·   "+(r"$J_r=1.000$" if z==1 else f"$J_r={a:.3f}\\pm0.008$"),
                 loc="left",fontsize=11,weight="bold",pad=9)
    ax.set_xticks(range(8),actions);ax.set_xlim(-.45,7.45);ax.set_xlabel("Batch-size action  a = B")
    ax.set_ylim(bottom=0);ax.grid(axis="y",lw=.55,color="#dee5e6");ax.set_axisbelow(True)
    print(z,"n",counts,"means",[round(float(x),3) for x in means])
axs[0].set_ylabel("One-epoch relative improvement (%)")
handles=[Line2D([],[],marker="o",ls="none",mfc="none",mec="#187f8e",label="Observed epoch"),
         Line2D([],[],ls="--",lw=1.15,color="#273b46",marker="D",label="Mean of displayed epochs at each action")]
fig.legend(handles=handles,ncol=2,loc="lower center",bbox_to_anchor=(.52,.055),frameon=False)
fig.text(.52,.034,"Improvement is the percentage reduction in the block’s running-minimum training loss.",ha="center",fontsize=9)
fig.savefig(p/"progress_action.pdf");fig.savefig(p/"progress_action.svg");fig.savefig(p/"progress_action.png",dpi=300)
