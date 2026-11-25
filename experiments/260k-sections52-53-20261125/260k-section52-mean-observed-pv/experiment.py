"""All-date Section 5.2 with mean-only LYA and last-observation PV forecast."""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import runtime as rt

PREVIOUS=rt.ROOT.parent/'260k-section52-decreasing-weight'
WEIGHTS=[100.,10.,1.]
CORE=['engine.cpp','controller.cpp','runtime.py','pv_observations.py']


def write_frozen(path,obj):
    if path.exists():assert json.loads(path.read_text())==obj,f'Frozen configuration changed: {path}'
    else:path.write_text(json.dumps(obj,indent=2)+'\n')


def prepare():
    original=json.loads((PREVIOUS/'protocol.json').read_text());p=copy.deepcopy(original)
    p['revision']={
        'date':'2026-11-25','request':'Decision uses only MLP means and most recent observed PV power times mean duration',
        'decision_information':'J, phase, Q, three MLP means for each action, V, sigma_e, last observed PV power',
        'decision_PV_J':'alpha * last valid raw power measured at timestamp <= current time * m_T',
        'decision_shortfall_J':'max(m_dem - decision_PV_J, 0); plug-in forecast, not exact expected shortfall',
        'no_decision_noise_bank':True,'no_decision_future_PV':True,'no_decision_PV_integration':True,
        'observation_initialization':'Public observations() from current date plus last earlier valid real measurement; no interpolated midnight value',
        'measurement_availability':'Available at measurement timestamp; no additional telemetry latency modeled',
        'measurement_hold':'Keep latest valid measurement across gaps; no age cap; report staleness',
        'environment':'Unchanged calibrated empirical joint noise; actual duration/demand/error and PVDatabase-integrated actual-window PV',
        'weights':WEIGHTS,'sigma_e':.24,
        'previous_control':'LYA-previous = previous decreasing-weight LYA using noise-distribution and future-integrated PV information',
        'reuse':'Only six baselines and historical control; rerun every LYA tuning and evaluation observation',
    }
    p['tuning']['weights']=[WEIGHTS]
    p['tuning']['selection']='Fix weights (100,10,1); re-evaluate original seven V candidates on the original 48 days and 8 seeds. Select lowest equal-day TARS with DWAE<=0.24 in every tuning run; ties within 1e-10 W select smaller V.'
    p['tuning']['scale_selection']='No new scale choice. All original scales reported; primary Figure 8 remains original PV70.'
    p['source_sha256']={str((PREVIOUS/f).relative_to(rt.ROOT.parent)):rt.sha(PREVIOUS/f) for f in ['protocol.json','results/evaluation_runs.csv.gz','results/summary.csv','results/selected.json','data/date_quality.csv']}
    p['core_sha256']={f:rt.sha(rt.ROOT/f) for f in CORE}
    assert all(rt.sha(rt.SIM/f)==v for f,v in p['preserved_simulator'].items())
    assert rt.sha(rt.PV_PACKAGE/'src/opencem_pv/data/opencem_pv.sqlite')==p['pv']['database_sha256']
    write_frozen(rt.ROOT/'protocol.json',p)
    analysis=json.loads((PREVIOUS/'analysis_protocol.json').read_text())
    analysis['parameter_selection']=p['tuning']['selection']
    analysis['original_policy_control']=p['revision']['previous_control']
    analysis['revision_scope']='Replace decision information only; environmental responses and six baselines unchanged.'
    write_frozen(rt.ROOT/'analysis_protocol.json',analysis)
    print('Protocol frozen; six scales, original dates and seeds.',flush=True)


def checked_protocol():
    p=json.loads((rt.ROOT/'protocol.json').read_text())
    for f,h in p['core_sha256'].items():assert rt.sha(rt.ROOT/f)==h,f
    for f,h in p['source_sha256'].items():assert rt.sha(rt.ROOT.parent/f)==h,f
    assert json.loads((rt.ROOT/'results/engine_verification.json').read_text())['status']=='PASS'
    return p


def tune():
    p=checked_protocol();frames=[];start=time.perf_counter()
    for k,date in enumerate(p['tuning']['dates']):
        path=rt.ROOT/f'results/tune_{date}.csv.gz'
        if path.exists():frame=pd.read_csv(path)
        else:
            rt.set_day(date,None);rows=[]
            for scale in p['scales']:
                for V in p['tuning']['V']:
                    for seed in p['tuning']['seeds']:
                        row,_=rt.run('LYA',scale['alpha'],seed,V=V,weights=WEIGHTS)
                        row.update(date=date,scale_name=scale['name'],data_origin='New mean-only observed-PV tuning');rows.append(row)
            frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
        assert len(frame)==6*7*8 and not frame.duplicated(['scale_name','V','seed']).any()
        frames.append(frame)
        if (k+1)%12==0:print('Tuned',k+1,'/48 days;',round(time.perf_counter()-start,1),'s',flush=True)
    raw=pd.concat(frames,ignore_index=True);raw.to_csv(rt.ROOT/'results/tuning_runs.csv.gz',index=False)
    summary=raw.groupby(['scale_name','alpha','V'],as_index=False).agg(mean_TARS_W=('TARS_W','mean'),mean_DWAE=('DWAE','mean'),max_DWAE=('DWAE','max'),n=('seed','size'))
    summary['feasible_all_runs']=summary.max_DWAE.le(.24)
    summary.to_csv(rt.ROOT/'results/tuning_summary.csv',index=False)
    selected={}
    for scale in p['scales']:
        candidates=summary[(summary.scale_name==scale['name'])&summary.feasible_all_runs]
        assert len(candidates),f'No feasible candidate for {scale}'
        best=candidates[candidates.mean_TARS_W<=candidates.mean_TARS_W.min()+1e-10].sort_values('V').iloc[0]
        selected[scale['name']]=dict(alpha=scale['alpha'],V=float(best.V),weights=WEIGHTS,
            tuning_mean_TARS_W=float(best.mean_TARS_W),tuning_max_DWAE=float(best.max_DWAE))
    selected['_protocol_sha256']=rt.sha(rt.ROOT/'protocol.json')
    write_frozen(rt.ROOT/'results/selected.json',selected)
    print(json.dumps(selected,indent=2),flush=True)


def evaluate():
    p=checked_protocol();selected=json.loads((rt.ROOT/'results/selected.json').read_text())
    assert selected['_protocol_sha256']==rt.sha(rt.ROOT/'protocol.json')
    write_frozen(rt.ROOT/'cache_manifest.json',{'protocol':rt.sha(rt.ROOT/'protocol.json'),'selected':rt.sha(rt.ROOT/'results/selected.json'),**p['core_sha256']})
    previous=pd.read_csv(PREVIOUS/'results/evaluation_runs.csv.gz')
    previous=previous[previous.policy.isin(rt.POLICIES)].copy()
    previous.loc[previous.policy=='LYA','policy']='LYA-previous'
    previous['data_origin']='Unchanged historical baseline/control'
    frames=[];start=time.perf_counter()
    for k,date in enumerate(p['evaluation']['all_dates']):
        path=rt.ROOT/f'results/eval_{date}.csv.gz'
        if path.exists():frame=pd.read_csv(path)
        else:
            rt.set_day(date,None);rows=[]
            for scale in p['scales']:
                config=selected[scale['name']]
                for seed in p['evaluation']['seeds']:
                    row,_=rt.run('LYA',scale['alpha'],seed,V=config['V'],weights=WEIGHTS)
                    row.update(date=date,scale_name=scale['name'],data_origin='New mean-only observed-PV LYA');rows.append(row)
            frame=pd.concat([previous[previous.date==date],pd.DataFrame(rows)],ignore_index=True)
            frame.to_csv(path,index=False)
        assert len(frame)==6*32*8 and not frame.duplicated(['scale_name','policy','seed']).any()
        assert frame.groupby(['scale_name','seed']).donor_sha256.nunique().eq(1).all()
        assert frame.hours.le(24).all() and frame.epochs.eq(120).all()
        frames.append(frame)
        if (k+1)%30==0:print('Evaluated',k+1,'/240 days;',round(time.perf_counter()-start,1),'s',flush=True)
    raw=pd.concat(frames,ignore_index=True);raw.to_csv(rt.ROOT/'results/evaluation_runs.csv.gz',index=False)
    print('Completed',len(raw),'records; new LYA',int(raw.policy.eq('LYA').sum()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','tune','evaluate'])
    globals()[ap.parse_args().stage]()
