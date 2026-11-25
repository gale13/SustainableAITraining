"""Full same-V comparison in the requested compact single-panel format."""
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
from matplotlib.ticker import LogFormatterMathtext

ROOT=Path(__file__).resolve().parent
FAMILIES=['Linear','Exponential','Inverse-power','Random-monotone']


def main():
    p=json.loads((ROOT/'protocol.json').read_text());c=pd.read_csv(ROOT/'source_data/sameV_curves.csv')
    assert len(c)==68 and c.sigma_e.eq(.235).all() and c.V.gt(0).all()
    assert np.isfinite(c.select_dtypes('number')).all().all()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,
        'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'legend.frameon':False})
    fig,ax=plt.subplots(figsize=(3.5,3.2))
    fig.subplots_adjust(left=.22,right=.975,bottom=.21,top=.84)
    styles=[('#555555','--','s'),('#111111','-','^'),('#888888',':','D'),('#666666','-.','v')]
    for name,(color,ls,marker) in zip(FAMILIES,styles):
        g=c[c.decay_family.eq(name)].sort_values('V')
        np.testing.assert_allclose(g.V,p['V'],rtol=1e-12,atol=0)
        ax.fill_between(g.V,g.gain_ci_low_percent,g.gain_ci_high_percent,color=color,alpha=.09,lw=0)
        ax.plot(g.V,g.gain_percent,color=color,ls=ls,marker=marker,ms=3.4,lw=.9,label=name.replace('-',' '))
    ax.axhline(0,color='#aaaaaa',lw=.6,ls='--',zorder=0)
    ax.set_xscale('log');ax.set_xlim(7e-7,150)
    ax.set_xticks([1e-6,1e-4,1e-2,1,100]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel('Lyapunov parameter, $V$')
    ax.set_ylabel('TARS reduction vs. Fixed\nat the same $V$ (%)')
    ax.set_title(r'Same $V$, $\sigma_e=0.235$',loc='left',pad=10)
    ax.legend(loc='upper right',handlelength=1.8,labelspacing=.35,borderaxespad=.35,handletextpad=.45)
    ax.margins(y=.09)
    fig.canvas.draw()
    lo,hi=ax.get_ylim();assert lo<=c.gain_ci_low_percent.min() and hi>=c.gain_ci_high_percent.max()
    prefix=ROOT/'figures/weight_decay_sameV'
    fig.savefig(prefix.with_suffix('.pdf'));fig.savefig(prefix.with_suffix('.svg'))
    fig.savefig(prefix.with_suffix('.png'),dpi=300)
    fig.savefig(prefix.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    print('Exported full 17-point same-V curves for all four families.')


if __name__=='__main__':main()
