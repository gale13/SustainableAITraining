"""Audit saved evaluations, all-date causal traces and publication exports."""
import json
import os
import platform
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.gettempdir()+'/pv70-figure-mpl-cache')
import numpy as np
import pandas as pd
import fitz
from PIL import Image
import runtime as rt
from pv_observations import latest


def main():
    p=json.loads((rt.ROOT/'protocol.json').read_text());selected=json.loads((rt.ROOT/'results/selected.json').read_text())
    for name,digest in p['source_sha256'].items():assert rt.sha(rt.ROOT.parent/name)==digest,name
    for name,digest in p['core_sha256'].items():assert rt.sha(rt.ROOT/name)==digest,name
    assert all(rt.sha(rt.SIM/n)==v for n,v in p['preserved_simulator'].items())
    assert json.loads((rt.ROOT/'results/engine_verification.json').read_text())['status']=='PASS'
    assert json.loads((rt.ROOT/'results/standalone_controller_verification.json').read_text())['status']=='PASS'
    raw=pd.read_csv(rt.ROOT/'results/evaluation_runs.csv.gz')
    assert len(raw)==240*6*32*8 and raw.epochs.eq(120).all() and raw.hours.le(24).all()
    keys=['date','scale_name','policy','seed']
    assert not raw.duplicated(keys).any()
    assert set(raw.date)==set(p['evaluation']['all_dates']) and set(raw.seed)==set(p['evaluation']['seeds'])
    assert raw.groupby(['date','scale_name','policy']).size().eq(32).all()
    assert raw.groupby(['date','scale_name','seed']).donor_sha256.nunique().eq(1).all()
    columns=['TARS_W','DWAE','T_s','shortfall_J','mean_Q_s','demand_J','pv_J']+[f'action_{a}_count' for a in rt.ACTIONS]
    assert np.isfinite(raw[columns]).all().all()
    current=raw[raw.policy=='LYA']
    assert np.array_equal(current[['w1','w2','w3']],np.tile([100,10,1],(len(current),1)))
    assert np.isfinite(current.select_dtypes('number')).all().all()
    assert current.mean_observation_age_s.ge(0).all() and current.max_observation_age_s.ge(current.mean_observation_age_s).all()
    for scale,config in selected.items():
        if not scale.startswith('_'):assert current.loc[current.scale_name==scale,'V'].eq(config['V']).all()
    old=pd.read_csv(rt.ROOT.parent/'260k-section52-decreasing-weight/results/evaluation_runs.csv.gz')
    old=old[old.policy.isin(rt.POLICIES)].copy();old.loc[old.policy=='LYA','policy']='LYA-previous'
    original=old.set_index(keys).sort_index();untouched=raw[raw.policy!='LYA'].set_index(keys).sort_index()
    assert original.index.equals(untouched.index)
    np.testing.assert_allclose(untouched[columns],original[columns],rtol=1e-14,atol=1e-8)
    assert untouched.donor_sha256.equals(original.donor_sha256)
    means=raw.groupby(['date','scale_name','policy'])[['TARS_W','DWAE','mean_Q_s']].mean().sort_index()
    daily=pd.read_csv(rt.ROOT/'results/daily_metrics.csv').set_index(['date','scale_name','policy']).sort_index()
    error=float(abs(means-daily[means.columns]).to_numpy().max());assert error<1e-10

    traces=[];trace_error=0.;minimum_time_margin=1e100
    config=selected['mean70']
    for day in p['evaluation']['all_dates']:
        rt.set_day(day,None)
        row,trace=rt.run('LYA',config['alpha'],120000,V=config['V'],weights=config['weights'],trace=True)
        saved=current[(current.date==day)&(current.scale_name=='mean70')&(current.seed==120000)].iloc[0]
        trace_error=max(trace_error,abs(row['TARS_W']-saved.TARS_W))
        np.testing.assert_allclose(row['TARS_W'],saved.TARS_W,atol=1e-10,rtol=0)
        expected=np.array([latest(*rt.OBSERVATIONS,t,config['alpha']) for t in trace.clock_start_s])
        np.testing.assert_allclose(trace[['observed_power_W','observed_time_s','observation_age_s']],expected,atol=1e-9)
        assert trace.observed_time_s.le(trace.clock_start_s).all()
        minimum_time_margin=min(minimum_time_margin,float((trace.clock_start_s-trace.observed_time_s).min()))
        np.testing.assert_allclose(trace.estimated_PV_J,trace.observed_power_W*trace.mean_T_s,atol=1e-8,rtol=1e-12)
        np.testing.assert_allclose(trace.estimated_shortfall_J,np.maximum(trace.mean_demand_J-trace.estimated_PV_J,0),atol=1e-8)
        np.testing.assert_array_equal(trace.phase_weight,np.array([100,10,1])[trace.phase.astype(int)-1])
        trace.insert(0,'date',day);trace.insert(1,'seed',120000);traces.append(trace)
    trace=pd.concat(traces,ignore_index=True)
    trace.to_csv(rt.ROOT/'results/pv70_trace_seed120000.csv.gz',index=False)

    exports={}
    for path in sorted((rt.ROOT/'figures').glob('*.pdf')):
        with fitz.open(path) as doc:
            page=doc[0];text=' '.join(page.get_text().split())
            assert 'LYA-D' not in text and 'Duration-weighted normalized' not in text
            if path.stem.startswith('figure8'):assert 'Duration-weighted average error, DWAE' in text
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
            outside=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
            assert not outside,(path.name,outside)
            assert abs(page.rect.width/72-6.5)<1e-5
            assert all(len(doc.extract_font(f[0])[3])>0 for f in page.get_fonts())
            exports[path.name]=dict(width_inches=page.rect.width/72,height_inches=page.rect.height/72,
                min_font_pt=min(s['size'] for s in spans),fonts=sorted(set(s['font'] for s in spans)),embedded=True,outside_text=outside)
            if path.stem=='figure8_pv70_mean_observed':page.get_pixmap(dpi=150).save(rt.ROOT/'figures/figure8_pdf_proof.png')
    with Image.open(rt.ROOT/'figures/figure8_pv70_mean_observed.tiff') as im:
        assert im.size==(3900,3480) and im.info['dpi']==(600,600)
    figures=json.loads((rt.ROOT/'figure_data.json').read_text())
    report=dict(status='PASS',raw_runs=len(raw),new_LYA_evaluation_runs=len(current),tuning_runs=16128,
        simulated_epochs=int(raw.epochs.sum()),baseline_and_historical_control_rows_preserved=len(untouched),
        daily_mean_max_error=error,all_date_causal_trace_epochs=len(trace),max_saved_trace_TARS_error_W=trace_error,
        min_decision_time_minus_observation_time_s=minimum_time_margin,
        mean_only_decision_verified=True,last_observation_P_times_mT_verified=True,
        future_PV_and_noise_bank_counterfactuals_passed=True,standalone_controller_without_noise_file_passed=True,
        max_run_hours=float(raw.hours.max()),source_experiments_unchanged=True,frozen_simulator_unchanged=True,
        all_240_tiles_preserved=figures['tile_count']==240,all_1680_scatter_points_preserved=figures['scatter_points']==1680,
        exports=exports)
    (rt.ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=__import__('matplotlib').__version__)
    (rt.ROOT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    manifest={str(f.relative_to(rt.ROOT)):rt.sha(f) for f in sorted(rt.ROOT.rglob('*')) if f.is_file() and f.name!='checksums.json' and f.suffix!='.log' and '__pycache__' not in f.parts}
    (rt.ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='exports'},indent=2),flush=True)


if __name__=='__main__':main()
