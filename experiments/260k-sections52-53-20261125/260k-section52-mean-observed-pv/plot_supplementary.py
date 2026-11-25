"""Regenerate Section 5.2 supplementary plots from decreasing-weight results."""
import numpy as np
import pandas as pd
from plot_figure8 import ROOT, OUTPUT, POLICIES, LABELS, MARKERS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

S=pd.read_csv(ROOT/'results/summary.csv')
C=pd.read_csv(ROOT/'results/comparison.csv')
D=pd.read_csv(ROOT/'results/daily_comparison.csv')
SHADES=dict(zip(POLICIES,['.10','.40','.60','.25','.70','.48','.78']))
# Shared OR typography is initialized by plot_figure8; keep editable exports explicit.
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,name):
    fig.savefig(OUTPUT/f'{name}.pdf')
    fig.savefig(OUTPUT/f'{name}.svg')
    fig.savefig(OUTPUT/f'{name}.png',dpi=300)
    plt.close(fig)


def comparison(scale,name):
    frame=S[(S.subset=='all_240')&(S.scale_name==scale)].set_index('policy').loc[POLICIES]
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.2))
    fig.subplots_adjust(left=.095,right=.975,bottom=.23,top=.89,wspace=.55)
    ax=axes[0];y=np.arange(7)
    bars=ax.barh(y,frame.mean_TARS_W,color=['.2']+['.78']*6,height=.6,edgecolor='.3',linewidth=.5)
    ax.errorbar(frame.mean_TARS_W,y,xerr=np.array([frame.mean_TARS_W-frame.TARS_ci_low_W,frame.TARS_ci_high_W-frame.mean_TARS_W]),fmt='none',ecolor='black',capsize=2,lw=.7)
    for bar,row in zip(bars,frame.itertuples()):
        if row.violations:bar.set_hatch('///')
    ax.set(yticks=y,yticklabels=[LABELS[p] for p in POLICIES],ylim=(6.65,-.65),xlim=(0,630),xticks=[0,200,400,600],xlabel='Time-averaged renewable\nshortfall, TARS (W)')
    ax.set_title('(a) Mean renewable shortfall',loc='left')
    ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    ax=axes[1]
    offsets={'LYA':(-25,-15),'B32':(-19,9),'B128':(-36,-2),'B512':(-29,10),'STA':(6,-2),'INC':(-35,1),'DEC':(-17,-15)}
    for policy,row in frame.iterrows():
        ax.scatter(row.mean_DWAE,row.mean_TARS_W,s=30,marker=MARKERS[policy],facecolor='.15' if policy=='LYA' else 'white',edgecolor='.2',lw=.7,zorder=3)
        ax.annotate(LABELS[policy],(row.mean_DWAE,row.mean_TARS_W),xytext=offsets[policy],textcoords='offset points',fontsize=9,
            bbox={'facecolor':'white','edgecolor':'none','pad':.35} if policy=='STA' else None)
    ax.axvline(.24,color='.5',ls='--',lw=.7)
    ax.set(xlim=(.224,.25),ylim=(0,650),xticks=[.225,.235,.245],yticks=[0,200,400,600],xlabel='Duration-weighted average\nerror, DWAE',ylabel='Time-averaged renewable\nshortfall, TARS (W)')
    ax.set_title('(b) DWAE–TARS comparison',loc='left')
    save(fig,name)


def sensitivity():
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.8))
    fig.subplots_adjust(left=.125,right=.975,bottom=.18,top=.73,wspace=.44)
    for policy in POLICIES:
        f=S[(S.subset=='all_240')&(S.policy==policy)&(S.scale_name!='mean70')].sort_values('alpha')
        assert f.alpha.gt(0).all()
        axes[0].plot(f.alpha,f.mean_TARS_W,marker=MARKERS[policy],ms=3.5,lw=.9,color=SHADES[policy],label=LABELS[policy])
    axes[0].set(ylim=(0,630),yticks=[0,200,400,600],ylabel='Time-averaged renewable\nshortfall, TARS (W)')
    axes[0].set_title('(a) PV capacity sensitivity',loc='left')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.985),ncol=4,frameon=False,columnspacing=1.2,handlelength=1.8)
    for subset,color,marker,label in [('all_240','.15','o','All 240 dates'),('heldout_192','.6','s','192 held-out dates')]:
        f=C[(C.subset==subset)&(C.scale_name!='mean70')].sort_values('alpha')
        axes[1].plot(f.alpha,f.gain_percent,color=color,marker=marker,ms=3.5,lw=.9,label=label)
        axes[1].fill_between(f.alpha,f.gain_ci_low_percent,f.gain_ci_high_percent,color=color,alpha=.12,lw=0)
    row=C[(C.subset=='all_240')&(C.scale_name=='mean70')].iloc[0]
    axes[1].scatter(row.alpha,row.gain_percent,marker='D',s=28,facecolors='white',edgecolors='black',zorder=4,label='PV70')
    axes[1].axhline(0,color='.5',ls='--',lw=.6)
    axes[1].set(ylim=(-1,19),ylabel='TARS reduction relative\nto B32 (%)')
    axes[1].set_title('(b) Benefit over B32',loc='left')
    axes[1].legend(loc='lower right',frameon=False,fontsize=9,handlelength=1.5)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16]);ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('PV capacity multiplier, α')
    save(fig,'scale_sensitivity')


def daily():
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.5))
    fig.subplots_adjust(left=.10,right=.975,bottom=.29,top=.88,wspace=.45)
    for ax,scale,title in zip(axes,['capacity_x1','capacity_x4'],['(a) Raw PV, α = 1','(b) Fourfold capacity, α = 4']):
        f=D[D.scale_name==scale];assert len(f)==240
        for good,marker,color,label in [(True,'o','.45','195 dates: gaps ≤5 min'),(False,'x','.10','45 dates: longer gaps')]:
            q=f[f.strict_complete==good]
            ax.scatter(q.unlimited_mean_W,q.gain_W,marker=marker,s=10,c=color,alpha=.6,linewidths=.6,label=label)
        ax.axhline(0,color='.5',ls='--',lw=.7)
        ax.set_title(title,loc='left')
        ax.set(xlabel='Raw daily mean PV power (W)',ylabel='Daily TARS: B32 − LYA (W)')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    save(fig,'all_240_daily_differences')


def quality():
    subsets=['all_240','heldout_192','strict_195','ten_minute_213','recommended_142']
    names=['All 240 dates','192 held-out dates','195 dates: ≤5 min gaps','213 dates: ≤10 min gaps','142 dense recommended dates']
    fig,ax=plt.subplots(figsize=(6.5,3.3))
    fig.subplots_adjust(left=.33,right=.975,bottom=.20,top=.79)
    for shift,scale,marker,color,label in [(-.18,'capacity_x1','o','.1','Raw PV'),(0,'capacity_x4','s','.5','Fourfold capacity'),(.18,'mean70','^','.7','PV70')]:
        f=C[C.scale_name==scale].set_index('subset').loc[subsets]
        ax.errorbar(f.gain_percent,np.arange(5)+shift,xerr=np.array([f.gain_percent-f.gain_ci_low_percent,f.gain_ci_high_percent-f.gain_percent]),fmt=marker,color=color,ms=3.5,lw=.8,capsize=2,label=label)
    ax.set(yticks=np.arange(5),yticklabels=names,xlabel='TARS reduction relative to B32 (%)')
    ax.invert_yaxis();ax.axvline(0,color='.6',ls='--',lw=.6)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    save(fig,'data_quality_sensitivity')


if __name__=='__main__':
    comparison('capacity_x1','figure8_raw_pv')
    comparison('capacity_x4','figure8_capacity_x4')
    comparison('mean70','figure8_mean70')
    sensitivity();daily();quality()
    print('Rendered six supplementary figure families.',flush=True)
