"""Incremental Increasing comparator; reuse frozen matched decay runs."""
import argparse
import concurrent.futures
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import runtime as rt

BASE=rt.ROOT.parent/'260k-section53-weight-sameV'
OLD=rt.ROOT.parent/'260k-section53-weight-families-fixedV'
CORE=['runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py','experiment.py']
V_GRID=[v for e in range(-6,2) for v in (10.**e,3.*10.**e)]+[100.]


def frozen(path,value):
    if path.exists():assert json.loads(path.read_text())==value,f'Frozen configuration changed: {path}'
    else:path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    p=copy.deepcopy(json.loads((BASE/'protocol.json').read_text()))
    p['name']='Section 5.3 / four decay families vs Increasing at the same V'
    p['weights'].pop('Fixed');p['weights']['Increasing']=[100.,1000.,10000.]
    p['formulas'].pop('Fixed');p['formulas']['Increasing']='100*10**(k-1)'
    p['schedule_design']='Same four decay functions; Increasing=(100,1000,10000) from the prior fixed-V experiment. Match initial weight, not total weight.'
    p['parameter_selection']='User-requested Increasing companion; unchanged 17 V values, sigma=.235, all dates, weights and PV scale.'
    p['reuse']={'decay_runs':522240,'increasing_V01_runs':7680,'new_increasing_runs':122880,
        'description':'Reuse all four decay schedules from the Fixed companion and Increasing V=.1 from the previous fixed-V study; evaluate only the 16 missing Increasing settings.'}
    p['source_sha256']={str((folder/n).relative_to(rt.ROOT.parent)):rt.sha(folder/n)
        for folder,names in [(BASE,['protocol.json','verification.json','results/evaluation_runs.csv.gz',*CORE[:-1]]),
                            (OLD,['protocol.json','verification.json','results/evaluation_runs.csv.gz'])] for n in names}
    p['core_sha256']={n:rt.sha(rt.ROOT/n) for n in CORE}
    for n in CORE[:-1]:assert rt.sha(rt.ROOT/n)==rt.sha(BASE/n)==rt.sha(OLD/n)
    prior=json.loads((OLD/'protocol.json').read_text())
    assert prior['weights']['Increasing']==p['weights']['Increasing']
    for key in ['evaluation','pv','alpha','tasks','epochs_per_task','decision','environment']:
        assert prior[key]==p[key],key
    frozen(rt.ROOT/'protocol.json',p)
    frozen(rt.ROOT/'cache_manifest.json',dict(protocol_sha256=rt.sha(rt.ROOT/'protocol.json'),**p['core_sha256']))
    cfg=[dict(config_id=f'v{vi:02d}_w{wi}',sigma_e=.235,weight_schedule=name,V=V,w1=w[0],w2=w[1],w3=w[2])
         for vi,V in enumerate(V_GRID) for wi,(name,w) in enumerate(p['weights'].items())]
    pd.DataFrame(cfg).to_csv(rt.ROOT/'data/configurations.csv',index=False)
    source=pd.read_csv(OLD/'results/evaluation_runs.csv.gz')
    source=source[source.weight_schedule.eq('Increasing')&source.sigma_e.eq(.235)&source.V.eq(.1)].copy()
    assert len(source)==7680
    source.to_csv(rt.ROOT/'data/reused_increasing_V01.csv.gz',index=False)
    print('Prepared 85 comparison settings; 16 missing Increasing settings to evaluate.',flush=True)


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
        reuse=pd.read_csv(rt.ROOT/'data/reused_increasing_V01.csv.gz');reuse=reuse[reuse.date.eq(date)]
        assert len(reuse)==32
        for vi,V in enumerate(p['V']):
            if V==.1:
                part=reuse.copy();part['config_id']=f'v{vi:02d}_w4';rows.extend(part.to_dict('records'))
            else:
                for seed in p['evaluation']['seeds']:
                    row,_=rt.run('LYA',p['alpha'],seed,V=V,weights=p['weights']['Increasing'],sigma=.235)
                    row.update(date=date,config_id=f'v{vi:02d}_w4',weight_schedule='Increasing');rows.append(row)
        frame=pd.DataFrame(rows);temporary=path.with_suffix('.tmp')
        frame.to_csv(temporary,index=False,compression='gzip');os.replace(temporary,path)
    assert len(frame)==544 and not frame.duplicated(['config_id','seed']).any()
    assert frame.groupby('seed').donor_sha256.nunique().eq(1).all()
    assert frame.sigma_e.eq(.235).all() and frame.weight_schedule.eq('Increasing').all()
    assert frame.w1.eq(100).all() and frame.w2.eq(1000).all() and frame.w3.eq(10000).all()
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
    assert len(raw)==130560
    raw.to_csv(rt.ROOT/'results/increasing_runs.csv.gz',index=False)
    print('Saved 130560 Increasing runs (122880 new; 7680 reused).',flush=True)


def combined_runs():
    old=pd.read_csv(BASE/'results/evaluation_runs.csv.gz')
    decay=old[~old.weight_schedule.eq('Fixed')]
    assert len(decay)==522240
    increasing=pd.read_csv(rt.ROOT/'results/increasing_runs.csv.gz')
    assert len(increasing)==130560
    return pd.concat([decay,increasing],ignore_index=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate']);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();prepare() if a.stage=='prepare' else evaluate(a.workers)
