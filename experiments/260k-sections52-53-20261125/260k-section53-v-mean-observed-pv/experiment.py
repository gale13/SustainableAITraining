"""One fixed-weight V sweep, with resumable, hash-checked daily results."""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import runtime as rt

BASE=rt.ROOT.parent/'260k-section52-mean-observed-pv'
OLD=rt.ROOT.parent/'260k-section53-all-days'
CORE=['runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py','experiment.py']


def write_frozen(path,value):
    if path.exists():assert json.loads(path.read_text())==value,f'Frozen configuration mismatch: {path}'
    else:path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    base=json.loads((BASE/'protocol.json').read_text())
    old=json.loads((OLD/'protocol.json').read_text())
    p=dict(name='Section 5.3 / V sensitivity only / current mean-only defaults',created_date='2026-11-25',
        tasks=30,epochs_per_task=4,start_s=0,timezone='Asia/Shanghai',deadline_s=86400,
        sigma_e=old['sigma_e'],V=old['V'],weights=[100.,10.,1.],alpha=4.1928776597152995,
        reference_power_W=base['reference_power_W'],scale_calibration_raw_mean_W=base['scale_calibration_raw_mean_W'],
        pv=base['pv'],evaluation=old['evaluation'],
        decision=dict(inputs='J, phase, Q, three MLP means, V, sigma_e, phase weight, latest observed PV power',
            PV='alpha * last valid raw observation with timestamp <= clock * m_T',
            shortfall='max(m_dem - estimated_PV,0); plug-in prediction',noise_bank_access=False,
            future_PV_access=False,PV_integration=False,observation_age_cap=None),
        environment='Frozen MLP mean plus calibrated joint empirical noise; realized-duration PV integral; no battery',
        resets='Training state each task; Q and clock only at the start of each daily run',
        queue_mean='Arithmetic mean of the 120 epoch-start queue values, in seconds',
        forecast='Latest observed power times MLP mean duration; no noise scenarios and no future integration',
        parameter_selection='No new parameter selection. Report all 33 predeclared sigma_e/V combinations.',
        analysis=dict(mean='Average 32 seeds within date, then equal-weight average over dates',
            uncertainty='2000 paired calendar-week cluster bootstrap replicates, pointwise 95% intervals',
            bootstrap_seed=2026112305,subsets=['all_240','heldout_192','strict_195','recommended_142'],
            feasibility='Keep every point. Report number of runs with DWAE > sigma_e + 1e-12'),
        figure='Only (a) mean TARS vs V and (b) mean virtual queue vs V, three sigma_e curves',
        preserved_simulator=base['preserved_simulator'],
        source_sha256={str((folder/name).relative_to(rt.ROOT.parent)):rt.sha(folder/name) for folder,names in [
            (BASE,['protocol.json','runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py','results/evaluation_runs.csv.gz','results/engine_verification.json']),
            (OLD,['protocol.json','results/summary.csv'])] for name in names},
        core_sha256={n:rt.sha(rt.ROOT/n) for n in CORE})
    p['evaluation']=dict(all_dates=old['evaluation']['all_dates'],heldout_dates=old['evaluation']['heldout_dates'],seeds=old['evaluation']['seeds'])
    assert len(p['evaluation']['all_dates'])==240 and len(p['V'])==11 and len(p['sigma_e'])==3
    assert all(rt.sha(rt.SIM/n)==h for n,h in p['preserved_simulator'].items())
    write_frozen(rt.ROOT/'protocol.json',p)
    write_frozen(rt.ROOT/'cache_manifest.json',dict(protocol_sha256=rt.sha(rt.ROOT/'protocol.json'),**p['core_sha256']))
    cfg=[dict(config_id=f's{si}_v{vi:02d}',sigma_e=sigma,V=V,w1=100.,w2=10.,w3=1.)
         for si,sigma in enumerate(p['sigma_e']) for vi,V in enumerate(p['V'])]
    pd.DataFrame(cfg).to_csv(rt.ROOT/'data/configurations.csv',index=False)
    print('Prepared 33 configurations, 240 dates, 32 paired seeds.',flush=True)


def checked():
    p=json.loads((rt.ROOT/'protocol.json').read_text())
    assert json.loads((rt.ROOT/'cache_manifest.json').read_text())==dict(protocol_sha256=rt.sha(rt.ROOT/'protocol.json'),**p['core_sha256'])
    assert all(rt.sha(rt.ROOT/n)==h for n,h in p['core_sha256'].items())
    assert all(rt.sha(rt.ROOT.parent/n)==h for n,h in p['source_sha256'].items())
    assert all(rt.sha(rt.SIM/n)==h for n,h in p['preserved_simulator'].items())
    assert rt.sha(rt.PV_PACKAGE/'src/opencem_pv/data/opencem_pv.sqlite')==p['pv']['database_sha256']
    return p


def day_worker(date):
    p=json.loads((rt.ROOT/'protocol.json').read_text());path=rt.ROOT/f'results/evaluate/{date}.csv.gz'
    if path.exists():frame=pd.read_csv(path)
    else:
        rt.set_day(date,None);rows=[]
        for si,sigma in enumerate(p['sigma_e']):
            for vi,V in enumerate(p['V']):
                for seed in p['evaluation']['seeds']:
                    row,_=rt.run('LYA',p['alpha'],seed,V=V,weights=p['weights'],sigma=sigma,tasks=p['tasks'])
                    row.update(date=date,config_id=f's{si}_v{vi:02d}');rows.append(row)
        frame=pd.DataFrame(rows)
        temporary=path.with_suffix('.tmp')
        frame.to_csv(temporary,index=False,compression='gzip');os.replace(temporary,path)
    assert len(frame)==1056 and frame.date.eq(date).all()
    assert not frame.duplicated(['config_id','seed']).any()
    assert frame.groupby('seed').donor_sha256.nunique().eq(1).all()
    assert frame.epochs.eq(120).all() and frame.hours.le(24).all()
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    assert frame.queue_bound_slack.ge(-1e-10).all()
    return date


def evaluate(workers=3):
    p=checked()
    assert json.loads((rt.ROOT/'results/engine_verification.json').read_text())['status']=='PASS'
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(day_worker,date) for date in p['evaluation']['all_dates']]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if i%20==0:print(f'Completed {i}/240 dates, {time.perf_counter()-start:.1f} s',flush=True)
    frames=[pd.read_csv(rt.ROOT/f'results/evaluate/{date}.csv.gz') for date in p['evaluation']['all_dates']]
    raw=pd.concat(frames,ignore_index=True)
    assert len(raw)==253440
    raw.to_csv(rt.ROOT/'results/evaluation_runs.csv.gz',index=False)
    print('Saved 253440 newly simulated runs; 30412800 epochs.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate']);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args()
    prepare() if a.stage=='prepare' else evaluate(a.workers)
