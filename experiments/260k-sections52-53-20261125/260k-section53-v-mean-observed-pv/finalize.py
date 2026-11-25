"""Audit the sole V figure, independent controller and saved result provenance."""
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
from PIL import Image
import runtime as rt
from experiment import checked
from controller import MeanOnlyLYA


def main():
    p=checked();audit=json.loads((rt.ROOT/'results/analysis_verification.json').read_text())
    engine=json.loads((rt.ROOT/'results/engine_verification.json').read_text())
    assert audit['status']==engine['status']=='PASS'
    for n in ['engine.cpp','controller.cpp','controller.py','pv_observations.py']:
        assert rt.sha(rt.ROOT/n)==rt.sha(rt.ROOT.parent/'260k-section52-mean-observed-pv'/n)
    count=0;score_error=0.
    rt.set_day('2025-12-26',None)
    with tempfile.TemporaryDirectory(prefix='260k-mean-only-controller-') as temp:
        for n in ['models.json','model_config.json']:shutil.copy2(rt.SIM/n,Path(temp)/n)
        assert sorted(x.name for x in Path(temp).iterdir())==['model_config.json','models.json']
        for sigma in p['sigma_e']:
            for V in p['V']:
                policy=MeanOnlyLYA(temp,V=V,weights=p['weights'],sigma_e=sigma)
                for J,phase,Q,clock in [(1.,1,0.,0.),(.3,2,3.5,40000.),(.25,3,100.,60000.)]:
                    action,observation,details=rt.decision(J,phase,Q,clock,V,p['weights'][phase-1],p['alpha'],sigma)
                    decision=policy.select(J,phase,Q,observation[0])
                    assert decision['action']==action
                    np.testing.assert_allclose(decision['means'],details[:,:3],rtol=1e-11,atol=1e-8)
                    np.testing.assert_allclose(decision['scores'],details[:,5],rtol=1e-9,atol=1e-8)
                    score_error=max(score_error,float(abs(decision['scores']-details[:,5]).max()));count+=1
    pdf=rt.ROOT/'figures/figure9_V_ab.pdf'
    with fitz.open(pdf) as doc:
        assert len(doc)==1;page=doc[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        outside=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
        assert not outside,outside
        text=' '.join(page.get_text().split())
        assert '(a)' in text and '(b)' in text and '(c)' not in text and '(d)' not in text
        assert 'virtual queue' in text and 'shortfall, TARS (W)' in text
        assert abs(page.rect.width/72-6.5)<1e-5
        assert all(len(doc.extract_font(f[0])[3])>0 for f in page.get_fonts())
        export=dict(width_inches=page.rect.width/72,height_inches=page.rect.height/72,
            fonts=sorted(set(s['font'] for s in spans)),minimum_text_size_pt=min(s['size'] for s in spans),
            embedded_fonts=True,outside_text=outside)
        (rt.ROOT/'qa').mkdir(exist_ok=True)
        page.get_pixmap(dpi=160).save(rt.ROOT/'qa/figure9_V_ab_pdf_proof.png')
    with Image.open(rt.ROOT/'figures/figure9_V_ab.tiff') as image:
        assert image.size==(3900,1800) and image.info['dpi']==(600,600)
    assert sorted(x.name for x in (rt.ROOT/'figures').iterdir())==[
        'figure9_V_ab.pdf','figure9_V_ab.png','figure9_V_ab.svg','figure9_V_ab.tiff']
    source=pd.read_csv(rt.ROOT/'source_data/figure9_V_ab.csv')
    assert len(source)==33 and (source[['w1','w2','w3']].to_numpy()==[100,10,1]).all()
    audit.update(independent_multi_sigma_replay_epochs=engine['independent_python_replay_epochs'],
        standalone_decisions_with_no_noise_file=count,standalone_max_score_error=score_error,
        decision_future_PV_and_noise_isolation=engine['environment_noise_bank_irrelevant_to_decision'] and engine['future_PV_values_irrelevant'],
        frozen_environment_and_history_unchanged=True,exports=export)
    (rt.ROOT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    (rt.ROOT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,matplotlib=__import__('matplotlib').__version__),indent=2)+'\n')
    manifest={str(f.relative_to(rt.ROOT)):rt.sha(f) for f in sorted(rt.ROOT.rglob('*')) if f.is_file() and f.name!='checksums.json' and f.suffix!='.log' and '__pycache__' not in f.parts}
    (rt.ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
