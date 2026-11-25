"""Compose the two verified same-V comparisons from unchanged source values."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'pv70-figure-mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import fitz
from PIL import Image

ROOT=Path(__file__).resolve().parent
FAMILIES=['Linear','Exponential','Inverse-power','Random-monotone']
SOURCES={'Fixed':'260k-section53-weight-sameV','Increasing':'260k-section53-weight-sameV-increasing'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    curves={};manifest={};protocols=[]
    for comparator,folder in SOURCES.items():
        source=ROOT.parent/folder
        assert json.loads((source/'verification.json').read_text())['status']=='PASS'
        p=json.loads((source/'protocol.json').read_text());protocols.append(p)
        path=source/'source_data/sameV_curves.csv';c=pd.read_csv(path)
        assert len(c)==68 and c.sigma_e.eq(.235).all() and c.comparator.eq(comparator).all()
        assert c.V.gt(0).all() and np.isfinite(c.select_dtypes('number')).all().all()
        assert c[f'{comparator.lower()}_TARS_W'].gt(0).all()
        for family in FAMILIES:
            g=c[c.decay_family.eq(family)].sort_values('V')
            np.testing.assert_allclose(g.V,p['V'],rtol=1e-12,atol=0)
            assert len(g)==17
        curves[comparator]=c
        copied=ROOT/f'source_data/{comparator.lower()}_curves.csv'
        shutil.copy2(path,copied);assert sha(path)==sha(copied)
        for n in ['source_data/sameV_curves.csv','protocol.json','verification.json']:
            manifest[f'{folder}/{n}']=sha(source/n)
    for key in ['V','sigma_e','evaluation','alpha','decision','environment']:
        assert protocols[0][key]==protocols[1][key],key
    a=curves['Fixed'].sort_values(['decay_family','V'])
    b=curves['Increasing'].sort_values(['decay_family','V'])
    np.testing.assert_array_equal(a.decreasing_TARS_W,b.decreasing_TARS_W)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,
        'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.982,bottom=.19,top=.77,wspace=.38)
    styles=[('#555555','--','s'),('#111111','-','^'),('#888888',':','D'),('#666666','-.','v')]
    low=min(0.,*(c.gain_ci_low_percent.min() for c in curves.values()))
    high=max(c.gain_ci_high_percent.max() for c in curves.values())
    pad=.09*(high-low);limits=(low-pad,high+pad)
    for letter,ax,(comparator,c) in zip('ab',axes,curves.items()):
        for name,(color,ls,marker) in zip(FAMILIES,styles):
            g=c[c.decay_family.eq(name)].sort_values('V')
            ax.fill_between(g.V,g.gain_ci_low_percent,g.gain_ci_high_percent,color=color,alpha=.09,lw=0)
            line,=ax.plot(g.V,g.gain_percent,color=color,ls=ls,marker=marker,ms=3.4,lw=.9,label=name.replace('-',' '))
            np.testing.assert_array_equal(line.get_ydata(),g.gain_percent)
        ax.axhline(0,color='#aaaaaa',lw=.6,ls='--',zorder=0)
        ax.set_xscale('log');ax.set_xlim(7e-7,150);ax.set_ylim(*limits)
        ax.set_xticks([1e-6,1e-4,1e-2,1,100]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xlabel('Lyapunov parameter, $V$')
        ax.set_ylabel(f'TARS reduction vs. {comparator}\nat the same $V$ (%)')
        ax.set_title(f'({letter}) '+r'$\sigma_e=0.235$',loc='left',pad=10)
        ax.tick_params(labelleft=True)
        assert limits[0]<=c.gain_ci_low_percent.min() and limits[1]>=c.gain_ci_high_percent.max()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.535,.995),ncol=4,
        handlelength=2.2,columnspacing=2.,handletextpad=.55)
    prefix=ROOT/'figures/weight_decay_sameV_ab'
    fig.savefig(prefix.with_suffix('.pdf'));fig.savefig(prefix.with_suffix('.svg'))
    fig.savefig(prefix.with_suffix('.png'),dpi=300)
    fig.savefig(prefix.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    with fitz.open(prefix.with_suffix('.pdf')) as doc:
        assert len(doc)==1;page=doc[0]
        spans=[x for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for x in line['spans']]
        outside=[x['text'] for x in spans if not page.rect.contains(fitz.Rect(x['bbox']))]
        assert not outside,outside
        assert abs(page.rect.width/72-7.2)<1e-5
        assert all(doc.extract_font(f[0])[3] for f in page.get_fonts())
        page.get_pixmap(dpi=200).save(ROOT/'qa/pdf_proof.png')
        fonts=sorted(set(x['font'] for x in spans))
    with Image.open(prefix.with_suffix('.tiff')) as im:
        assert im.size==(4320,1920) and im.info['dpi']==(600,600)
    for name,h in manifest.items():assert sha(ROOT.parent/name)==h
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'verification.json').write_text(json.dumps(dict(status='PASS',panels=2,points_per_panel=68,
        all_points_retained=True,source_csv_byte_identical=True,shared_decay_values_identical=True,
        shared_xlim=[7e-7,150],shared_ylim=limits,source_files_unchanged=True,
        width_inches=7.2,height_inches=3.2,fonts=fonts,fonts_embedded=True,outside_text=outside),indent=2)+'\n')
    print('Exported verified panels (a) Fixed, (b) Increasing; 136 unchanged points.')


if __name__=='__main__':main()
