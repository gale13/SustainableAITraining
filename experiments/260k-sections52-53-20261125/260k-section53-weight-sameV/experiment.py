"""Predeclared weight comparison at one user-fixed V, with daily checkpoints."""
import argparse
import concurrent.futures
import json
import os
import time
import numpy as np
import pandas as pd
import runtime as rt

BASE=rt.ROOT.parent/'260k-section53-v-mean-observed-pv'
RANDOM_SEED=2026112305
RANDOM_U=float(np.random.default_rng(RANDOM_SEED).uniform(0.,1.))
SCHEDULES={
    'Linear':[100.,50.5,1.],
    'Exponential':[100.,10.,1.],
    'Inverse-power':[100.,float(100*2**(-np.log(100)/np.log(3))),1.],
    'Random-monotone':[100.,float(100*np.exp(-np.log(100)*RANDOM_U)),1.],
    'Fixed':[100.,100.,100.],
}
V_GRID=[v for exponent in range(-6,2) for v in (10.**exponent,3.*10.**exponent)]+[100.]
CORE=['runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py','experiment.py']


def frozen(path,value):
    if path.exists():assert json.loads(path.read_text())==value,f'Frozen configuration changed: {path}'
    else:path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    base=json.loads((BASE/'protocol.json').read_text())
    p={k:base[k] for k in ['tasks','epochs_per_task','start_s','timezone','deadline_s','sigma_e','alpha',
        'reference_power_W','scale_calibration_raw_mean_W','pv','evaluation','decision','environment','resets','queue_mean','forecast','analysis','preserved_simulator']}
    p.update(name='Section 5.3 / four decay families vs Fixed at the same V',created_date='2026-11-25',V=V_GRID,sigma_e=[.235],weights=SCHEDULES,
        schedule_design='All schedules start at 100; four decreasing families share endpoint 1. K=3, so they differ only in phase 2. Sums are not matched.',
        random_schedule=dict(seed=RANDOM_SEED,u=RANDOM_U,law='w2=100*exp(-ln(100)*u), u~Uniform(0,1), one fixed draw'),
        formulas={'Linear':'100-99*(k-1)/2','Exponential':'100*exp(-ln(100)*(k-1)/2)',
            'Inverse-power':'100*k**(-ln(100)/ln(3))','Random-monotone':'[100,100*exp(-ln(100)*u),1]',
            'Fixed':'100'},
        parameter_selection='No weight or date tuning. User-requested sigma=.235 and same-V design; predeclared complete 17-value logarithmic grid from 1e-6 to 100.',
        followup_context='Exploratory follow-up after the fixed V=.1 comparison; sigma=.235 is specified by the user-provided figure, not a new independent confirmatory selection.',
        source_sha256={str((BASE/n).relative_to(rt.ROOT.parent)):rt.sha(BASE/n) for n in [
            'protocol.json','runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py','verification.json','results/evaluation_runs.csv.gz']},
        core_sha256={n:rt.sha(rt.ROOT/n) for n in CORE})
    p['analysis']['feasibility']='Report every setting, observed violations and paired gains; do not claim dominance on the basis of TARS alone.'
    for n in CORE[:-1]:assert rt.sha(rt.ROOT/n)==rt.sha(BASE/n)
    frozen(rt.ROOT/'protocol.json',p)
    frozen(rt.ROOT/'cache_manifest.json',dict(protocol_sha256=rt.sha(rt.ROOT/'protocol.json'),**p['core_sha256']))
    cfg=[dict(config_id=f'v{vi:02d}_w{wi}',sigma_e=.235,weight_schedule=name,V=V,w1=w[0],w2=w[1],w3=w[2])
         for vi,V in enumerate(V_GRID) for wi,(name,w) in enumerate(SCHEDULES.items())]
    pd.DataFrame(cfg).to_csv(rt.ROOT/'data/configurations.csv',index=False)
    print('Frozen 85 configurations; same-V comparison, sigma=.235, 240 dates, 32 paired seeds.',flush=True)
    print(json.dumps(SCHEDULES,indent=2),flush=True)


def checked():
    p=json.loads((rt.ROOT/'protocol.json').read_text())
    assert json.loads((rt.ROOT/'cache_manifest.json').read_text())==dict(protocol_sha256=rt.sha(rt.ROOT/'protocol.json'),**p['core_sha256'])
    assert all(rt.sha(rt.ROOT/n)==h for n,h in p['core_sha256'].items())
    assert all(rt.sha(rt.ROOT.parent/n)==h for n,h in p['source_sha256'].items())
    assert all(rt.sha(rt.SIM/n)==h for n,h in p['preserved_simulator'].items())
    assert rt.sha(rt.PV_PACKAGE/'src/opencem_pv/data/opencem_pv.sqlite')==p['pv']['database_sha256']
    return p


def run_day(date):
    p=json.loads((rt.ROOT/'protocol.json').read_text());path=rt.ROOT/f'results/evaluate/{date}.csv.gz'
    if path.exists():frame=pd.read_csv(path)
    else:
        rt.set_day(date,None);rows=[]
        for vi,V in enumerate(p['V']):
            for wi,(name,weights) in enumerate(p['weights'].items()):
                for seed in p['evaluation']['seeds']:
                    row,_=rt.run('LYA',p['alpha'],seed,V=V,weights=weights,sigma=.235)
                    row.update(date=date,config_id=f'v{vi:02d}_w{wi}',weight_schedule=name);rows.append(row)
        frame=pd.DataFrame(rows);temporary=path.with_suffix('.tmp')
        frame.to_csv(temporary,index=False,compression='gzip');os.replace(temporary,path)
    assert len(frame)==2720 and not frame.duplicated(['config_id','seed']).any()
    assert frame.groupby('seed').donor_sha256.nunique().eq(1).all()
    assert frame.sigma_e.eq(.235).all()
    np.testing.assert_allclose(sorted(frame.V.unique()),V_GRID,rtol=1e-12,atol=0)
    assert frame.epochs.eq(120).all() and frame.hours.le(24).all()
    assert np.isfinite(frame.select_dtypes('number')).all().all() and frame.queue_bound_slack.ge(-1e-10).all()
    return date


def evaluate(workers=3):
    p=checked();start=time.perf_counter()
    assert json.loads((BASE/'verification.json').read_text())['status']=='PASS'
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_day,date) for date in p['evaluation']['all_dates']]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if i%40==0:print(f'Completed {i}/240 dates; {time.perf_counter()-start:.1f}s',flush=True)
    raw=pd.concat([pd.read_csv(rt.ROOT/f'results/evaluate/{date}.csv.gz') for date in p['evaluation']['all_dates']],ignore_index=True)
    assert len(raw)==652800
    raw.to_csv(rt.ROOT/'results/evaluation_runs.csv.gz',index=False)
    print('Saved 652800 runs, 78336000 epochs.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate']);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();prepare() if a.stage=='prepare' else evaluate(a.workers)
