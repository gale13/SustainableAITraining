"""All 600 observations in fit plot; all 120 held-out residuals in distribution plots."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-profile04-refit')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/profile04-refit-cache')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import ROOT,ACTIONS,METRICS
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
NAMES=['Training duration','GPU energy demand','Time-weighted error']
UNITS=['s','J','s']
COLORS=['#447791','#67a1a6','#bd894f','#967a9d','#777777']

def save(fig,name):
    fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(ROOT/(name+'.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)

def hist(ax,r,unit,title):
    ax.hist(r,bins='fd',density=True,color='#6c9eac',alpha=.8,label='Observed')
    grid=np.linspace(min(r),max(r),400);sd=np.std(r,ddof=1)
    ax.plot(grid,stats.norm.pdf(grid,np.mean(r),sd),color='#b46d44',lw=1.2,label='Matched normal')
    ax.set(xlabel=f'Residual ({unit})',ylabel='Probability density',title=title)

def main():
    d=pd.read_csv(ROOT/'epochs.csv');te=d[d.split=='test'];n=pd.read_csv(ROOT/'noise_summary.csv').query("group == 'all'").set_index('metric')
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),layout='constrained')
    for i,k in enumerate(METRICS):
        bg=d[d.split!='test'];ax[i].scatter(bg[k],bg['pred_'+k],s=7,alpha=.3,color='#999999',label='Train / validation (480)')
        for a,c in zip(ACTIONS,COLORS):
            g=te[te.action==a];ax[i].scatter(g[k],g['pred_'+k],s=12,color=c,alpha=.85,label=f'Test batch {a}')
        lo=min(d[k].min(),d['pred_'+k].min());hi=max(d[k].max(),d['pred_'+k].max())
        ax[i].plot([lo,hi],[lo,hi],ls='--',lw=.7,color='black');ax[i].set(title=NAMES[i],xlabel=f'Observed ({UNITS[i]})',ylabel=f'MLP prediction ({UNITS[i]})')
    ax[0].legend(fontsize=5.7,loc='upper left');fig.suptitle('Three conditional means | 600 epochs, paired-seed split',fontsize=9)
    save(fig,'fit')
    fig,ax=plt.subplots(2,3,figsize=(10.5,6),layout='constrained')
    for i,k in enumerate(METRICS):
        r=te['res_'+k].to_numpy();hist(ax[0,i],r,UNITS[i],NAMES[i])
        ax[0,i].text(.98,.96,f"SD = {n.loc[k,'sd']:.3g}\nShapiro p = {n.loc[k,'shapiro_p']:.2g}",ha='right',va='top',transform=ax[0,i].transAxes,fontsize=6)
        q=stats.norm.ppf((np.arange(len(r))+.5)/len(r));ax[1,i].scatter(q,np.sort(r),s=9,color='#447791',alpha=.8)
        ax[1,i].plot(q,r.mean()+q*r.std(ddof=1),color='#b46d44',lw=1.)
        ax[1,i].set(xlabel='Standard normal quantile',ylabel=f'Observed residual quantile ({UNITS[i]})')
    ax[0,0].legend(fontsize=5.4,loc='upper left');fig.suptitle('Held-out residuals: measured − predicted | 120 epochs, all tails retained',fontsize=9)
    save(fig,'noise')
    # Random horizontal jitter only separates measured points; outcomes are never simulated here.
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained');rng=np.random.default_rng(20261119)
    for i,k in enumerate(METRICS):
        groups=[te.loc[te.action==a,'res_'+k].to_numpy() for a in ACTIONS]
        ax[i].boxplot(groups,labels=ACTIONS,showfliers=False,widths=.55,medianprops={'color':'black','linewidth':1},whis=1.5)
        for pos,(g,c) in enumerate(zip(groups,COLORS),1):ax[i].scatter(pos+rng.uniform(-.14,.14,len(g)),g,s=9,color=c,alpha=.75)
        ax[i].axhline(0,color='gray',lw=.6,ls=':');ax[i].set(title=NAMES[i],xlabel='Batch size (sequences / update)',ylabel=f'Test residual ({UNITS[i]})')
    fig.suptitle('Residual distributions differ across actions | 24 observations per batch',fontsize=9)
    save(fig,'noise_by_action')
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.7),layout='constrained')
    for i,k in enumerate(METRICS):hist(ax[i],te['z_'+k].to_numpy(),'validation SD',NAMES[i])
    ax[0].legend(fontsize=5.5);fig.suptitle('Location / scale calibrated per action on validation only',fontsize=9)
    save(fig,'standardized_noise')
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.5),layout='constrained');cp=pd.read_csv(ROOT/'common_proxy_points.csv')
    for i,k in enumerate(METRICS):hist(ax[i],cp.loc[cp.metric==k,'proxy'].to_numpy(),UNITS[i],NAMES[i])
    fig.suptitle('Matched seed–epoch mean residual | descriptive proxy, not identified exogenous noise',fontsize=8)
    save(fig,'common_proxy')
if __name__=='__main__':main()
