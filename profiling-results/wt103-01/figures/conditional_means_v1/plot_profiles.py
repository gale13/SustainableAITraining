"""Render observed outcomes and the stored 10-epoch math-backend refit, in run root."""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

P = Path("figures/conditional_means_v1")
sys.path.insert(0, str(P.resolve()))
from predict import Profile
model = Profile(P)
bs = sorted(model.demand)
def rows(name):
    with (P/name).open() as f:
        return list(csv.DictReader(f))
phase_row, = [r for r in rows("phase_selection.csv") if int(r["horizon"])==10 and int(r["K"])==3]
beta = np.array([float(z) for z in phase_row["beta"].split(";")])
cost = {int(r["batch"]):float(r["m_T_seconds"]) for r in rows("action_costs.csv")
        if r["backend"]=="math" and int(r["horizon"])==10}
data = np.full((8,3,10,3),np.nan)
with open("training/all_epochs.csv") as f:
    for r in csv.DictReader(f):
        t,e,b = int(r["block_id"]), int(r["epoch_in_block"]),int(r["batch_sequences"])
        if t in (16,17,18) and e<=10:
            assert e>=1 and int(r["global_epoch"])==229*t+e and r["train_energy_valid"]=="True"
            i = (bs.index(b),t-16,e-1)
            assert np.isnan(data[i]).all()
            data[i]=[float(r[k]) for k in ("train_loss","train_seconds","train_energy_j")]
assert data.shape==(8,3,10,3) and np.isfinite(data).all()
T=data[:,:,:,1]
D=data[:,:,1:,2]
J=np.minimum.accumulate(data[:,:,:,0],axis=2)/(data[:,:,:1,0]+1e-12)
x,y=J[:,:,:-1],J[:,:,1:]
delta=(y-x)/(x+1e-12)
phase=np.ones(delta.shape,dtype=int)
phase[:,:,2:]=1+np.searchsorted(beta,(delta[:,:,:-2]+delta[:,:,1:-1])/2,side="left")
phase=np.maximum.accumulate(phase,axis=2)
err=T[:,:,1:]*y
for i,b in enumerate(bs):
    np.testing.assert_allclose(T[i].mean(),cost[b])
    np.testing.assert_allclose(np.polyfit(x[i].ravel()-1,D[i].ravel(),1),
                               [float(model.demand[b][k]) for k in ("coef_J_minus_1","intercept")])
    for z in (1,2,3):
        cell=model.cells[b,z]; mask=phase[i]==z
        assert int(cell["count"])==mask.sum()
        np.testing.assert_allclose([x[i][mask].min(),x[i][mask].max()],
                                   [float(cell[k]) for k in ("supported_J_min","supported_J_max")])
        np.testing.assert_allclose(np.average((y[i]/x[i])[mask],weights=T[i,:,1:][mask]),
                                   float(cell["time_weighted_retention"]))
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
                     "font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,
                     "axes.spines.right":False,"axes.linewidth":.7,"axes.edgecolor":"#69777c",
                     "text.color":"#182e36","axes.labelcolor":"#243940","xtick.color":"#40565b",
                     "ytick.color":"#40565b","axes.axisbelow":True,"path.simplify":False})
ink="#1e353e"; blue="#177082"; hues=["#147381","#a05226","#70548d"]
def dot(ax,xx,yy,t,color):
    ax.scatter(xx,yy,s=22 if t==2 else 16,marker="^" if t==2 else "o",
               facecolors="none" if t==2 else color,edgecolors=color,linewidths=.85 if t==2 else .35,
               alpha=.9 if t==2 else .54,zorder=2)
def setup(ax):
    ax.grid(axis="y",color="#e0e5e7",lw=.55)
    ax.tick_params(length=3,pad=3)
def times(ax):
    for i,b in enumerate(bs):
        for t in range(3):
            dot(ax,np.full(10,b),T[i,t],t,blue)
        ax.plot([b-1.25,b+1.25],[cost[b]]*2,color=ink,lw=2.25,zorder=4)
    ax.set(xlabel="Batch size, B", ylabel="Training time (s)",xlim=(0,68),xticks=bs)
    ax.tick_params(axis="x",labelrotation=45)
    setup(ax)
def costs(ax,i,kind):
    b=bs[i]
    if kind=="demand":
        for t in range(3):
            dot(ax,x[i,t],D[i,t]/1000,t,blue)
        s=model.demand[b]; xx=np.array([float(s["J_min"]),float(s["J_max"])])
        pred=(float(s["intercept"])+float(s["coef_J_minus_1"])*(xx-1))/1000
        ax.plot(xx,pred,color=ink,lw=2.05,zorder=3)
    else:
        for z,col in enumerate(hues,1):
            for t in range(3):
                keep=phase[i,t]==z
                dot(ax,x[i,t][keep],err[i,t][keep],t,col)
            cell=model.cells[b,z]
            xx=np.array([float(cell["supported_J_min"]),float(cell["supported_J_max"])])
            pred=cost[b]*xx*float(cell["time_weighted_retention"])
            ax.plot(xx,pred,color=col,lw=2.25,zorder=3)
    ax.set(xlim=(.837,1.008),xticks=[.85,.90,.95,1.0])
    setup(ax)
def legend(fig):
    h=[Line2D([],[],color=blue,marker="o",ls="",alpha=.62,label="Epoch from blocks 17–18"),
       Line2D([],[],color=blue,marker="^",mfc="white",ls="",label="Epoch from block 19"),
       Line2D([],[],color=ink,lw=2.3,label="Final fit (blocks 17–19)")]
    h.extend(Line2D([],[],color=c,lw=2,label=f"Phase {z}") for z,c in enumerate(hues,1))
    fig.legend(handles=h,loc="lower center",bbox_to_anchor=(.50,.012),ncol=6,frameon=False,
               handlelength=1.6,columnspacing=1.2,fontsize=8)
def head(fig,title,subtitle):
    fig.text(.069,.928,title,fontsize=17.5,weight="bold")
    fig.text(.069,.866,subtitle,fontsize=9.4,color="#46616a")

with PdfPages(P/"observed_vs_fitted.pdf") as pdf:
    fig,axs=plt.subplots(1,3,figsize=(12.0,4.5))
    fig.subplots_adjust(left=.069,right=.977,bottom=.234,top=.698,wspace=.40)
    head(fig,"Epoch measurements and fitted conditional means",
         "RTX 5090  /  first 10 epochs of each block  /  time across eight actions; state-dependent fits shown at B = 32")
    axs[0].set_title("(a)  Duration",loc="left",weight="bold");times(axs[0])
    for ax,kind,title,ylabel in [(axs[1],"demand","(b)  GPU demand","GPU energy (kJ)"),
                                 (axs[2],"error","(c)  Time-weighted loss","Time × post-epoch loss (s)")]:
        costs(ax,bs.index(32),kind);ax.set_title(title,loc="left",weight="bold")
        ax.set(xlabel="Pre-epoch normalized loss, J",ylabel=ylabel)
    legend(fig);pdf.savefig(fig);fig.savefig(P/"observed_vs_fitted.svg");plt.close(fig)
    for kind,title,ylabel in [("demand","GPU energy demand","GPU energy per epoch (kJ)"),
                              ("error","Time-weighted training loss","Time × post-epoch normalized loss (s)")]:
        fig,axs=plt.subplots(2,4,figsize=(12,7.1),sharex=True)
        fig.subplots_adjust(left=.085,right=.98,bottom=.16,top=.80,hspace=.38,wspace=.33)
        head(fig,title+" across batch sizes",
             "27 observed epoch outcomes per batch  /  lines span observed states only  /  each panel has its own vertical scale")
        for ax,i in zip(axs.flat,range(len(bs))):
            costs(ax,i,kind);ax.set_title(f"Batch {bs[i]}",loc="left",fontsize=10,weight="bold")
        fig.text(.024,.48,ylabel,rotation=90,ha="center",va="center")
        fig.text(.52,.081,"Pre-epoch normalized training loss, J",ha="center")
        if kind=="error":
            legend(fig)
        else:
            h=[Line2D([],[],marker="o",ls="",color=blue,label="Blocks 17–18"),
               Line2D([],[],marker="^",mfc="white",ls="",color=blue,label="Block 19"),
               Line2D([],[],color=ink,label="Final fit (blocks 17–19)")]
            fig.legend(handles=h,loc="lower center",bbox_to_anchor=(.50,.01),ncol=3,frameon=False)
        pdf.savefig(fig);plt.close(fig)
print("Verified 240 durations and 216 paired states each for demand and error; matched all saved fit coefficients and phase supports.")
