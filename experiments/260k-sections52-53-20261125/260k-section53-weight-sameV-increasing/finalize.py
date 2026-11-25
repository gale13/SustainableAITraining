"""Final result/provenance and compact OR export audit."""
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'pv70-figure-mpl-cache'))
import fitz
import numpy as np
import pandas as pd
from PIL import Image
import runtime as rt
from experiment import checked,run_day


def main():
    p=checked();a=json.loads((rt.ROOT/'results/analysis_verification.json').read_text())
    e=json.loads((rt.ROOT/'results/engine_verification.json').read_text());assert a['status']==e['status']=='PASS'
    c=pd.read_csv(rt.ROOT/'source_data/sameV_curves.csv');s=pd.read_csv(rt.ROOT/'source_data/absolute_metrics.csv')
    assert len(c)==68 and len(s)==85 and s.violations.eq(0).all()
    assert c.groupby('decay_family').size().eq(17).all()
    for _,g in c.groupby('decay_family'):np.testing.assert_allclose(g.sort_values('V').V,p['V'],rtol=1e-12,atol=0)
    assert c.comparator.eq('Increasing').all() and c.increasing_TARS_W.gt(0).all()
    assert s.max_run_hours.le(24).all()
    run_day('2025-07-13') # exercise the saved-CSV cache path including 0.0003 and 0.3
    with fitz.open(rt.ROOT/'figures/weight_decay_sameV_increasing.pdf') as doc:
        assert len(doc)==1;page=doc[0]
        spans=[x for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for x in line['spans']]
        outside=[x['text'] for x in spans if not page.rect.contains(fitz.Rect(x['bbox']))]
        assert not outside,outside
        assert abs(page.rect.width/72-3.5)<1e-5
        assert all(len(doc.extract_font(f[0])[3])>0 for f in page.get_fonts())
        export=dict(width_inches=page.rect.width/72,height_inches=page.rect.height/72,
            min_font_pt=min(x['size'] for x in spans),fonts=sorted(set(x['font'] for x in spans)),embedded=True,outside_text=outside)
        page.get_pixmap(dpi=200).save(rt.ROOT/'qa/sameV_increasing_pdf_proof.png')
    with Image.open(rt.ROOT/'figures/weight_decay_sameV_increasing.tiff') as im:
        assert im.size==(2100,1920) and im.info['dpi']==(600,600)
    assert sorted(f.name for f in (rt.ROOT/'figures').iterdir())==[f'weight_decay_sameV_increasing.{s}' for s in ['pdf','png','svg','tiff']]
    a.update(engine_checks=e,exports=export,all_68_points_retained=True,one_figure=True,
        full_V_range_preserved=True,saved_csv_resume_validated=True)
    (rt.ROOT/'verification.json').write_text(json.dumps(a,indent=2)+'\n')
    (rt.ROOT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,matplotlib=__import__('matplotlib').__version__),indent=2)+'\n')
    manifest={str(f.relative_to(rt.ROOT)):rt.sha(f) for f in sorted(rt.ROOT.rglob('*')) if f.is_file() and f.name!='checksums.json' and f.suffix!='.log' and '__pycache__' not in f.parts}
    (rt.ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(a,indent=2),flush=True)


if __name__=='__main__':main()
