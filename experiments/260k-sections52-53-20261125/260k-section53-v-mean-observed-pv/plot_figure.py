"""The sole current Section 5.3 V figure: panels (a) and (b)."""
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'pv70-figure-mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext

ROOT=Path(__file__).resolve().parent


def main():
    p=json.loads((ROOT/'protocol.json').read_text())
    data=pd.read_csv(ROOT/'source_data/figure9_V_ab.csv')
    assert len(data)==33 and data.subset.eq('all_240').all()
    assert np.isfinite(data.select_dtypes('number')).all().all()
    assert data.V.gt(0).all() and data.Q_ci_low.gt(0).all()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,
        'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.0))
    fig.subplots_adjust(left=.115,right=.985,bottom=.22,top=.80,wspace=.40)
    ink=['#111111','#666666','#999999'];markers=['o','s','^']
    handles=[]
    for sigma,color,marker in zip(p['sigma_e'],ink,markers):
        g=data[data.sigma_e.eq(sigma)].sort_values('V')
        np.testing.assert_allclose(g.V,p['V'],rtol=1e-12)
        for ax,metric,prefix in [(axes[0],'mean_TARS_W','TARS'),(axes[1],'mean_Q_s','Q')]:
            ax.fill_between(g.V,g[prefix+'_ci_low'],g[prefix+'_ci_high'],color=color,alpha=.12,lw=0,zorder=1)
            ax.plot(g.V,g[metric],color=color,marker=marker,ms=3.6,lw=.9,zorder=2)
        handles.append(Line2D([],[],color=color,marker=marker,ms=3.6,lw=.9,label=rf'$\sigma_e={sigma:g}$'))
    for i,ax in enumerate(axes):
        ax.set_xscale('log');ax.set_xlim(7e-7,150)
        ax.set_xticks([1e-6,1e-4,1e-2,1,100]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xlabel('Lyapunov parameter, $V$')
        ax.set_title(f'({chr(97+i)})',loc='left',pad=7)
    axes[0].set_ylabel('Mean time-averaged renewable\nshortfall, TARS (W)')
    axes[0].margins(y=.08)
    axes[1].set_ylabel('Mean virtual queue (s)');axes[1].set_yscale('log');axes[1].margins(y=.10)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,handlelength=1.6,columnspacing=1.8)
    fig.canvas.draw()
    for ax in axes:
        lo,hi=ax.get_ylim()
        prefix='TARS' if ax is axes[0] else 'Q'
        assert lo<=data[prefix+'_ci_low'].min() and hi>=data[prefix+'_ci_high'].max()
    name=ROOT/'figures/figure9_V_ab'
    fig.savefig(name.with_suffix('.pdf'))
    fig.savefig(name.with_suffix('.svg'))
    fig.savefig(name.with_suffix('.png'),dpi=300)
    fig.savefig(name.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    (ROOT/'figure_data.json').write_text(json.dumps(dict(figures=1,panels=2,sigma_curves_per_panel=3,
        V_points_per_curve=11,total_points_per_panel=33,all_dates=240,raw_runs=253440,
        shaded_interval='Pointwise 95% paired calendar-week bootstrap, 2000 resamples'),indent=2)+'\n')
    print('Exported only figure9_V_ab: PDF, SVG, PNG, 600 dpi TIFF.',flush=True)


if __name__=='__main__':main()
