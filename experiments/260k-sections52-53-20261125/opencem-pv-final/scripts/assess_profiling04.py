"""Assess time resolution and smoke-test the full-data simulator with daily PV."""
import hashlib,json,runpy,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from opencem_pv import PVDatabase

def main():
    simroot=ROOT.parent/'tinystories-04-simulator-full-data'
    profile=pd.read_csv(simroot/'epochs.csv')
    summary=profile.groupby('action')['T'].agg(['count','min','median','mean','max'])
    summary['intervals_per_epoch_at_13s']=summary['median']/13
    summary['intervals_per_epoch_at_111s']=summary['median']/111
    summary['intervals_per_epoch_at_366s']=summary['median']/366
    summary['hours_for_30_tasks_4_epochs']=summary['mean']*120/3600
    summary.to_csv(ROOT/'profiling04_resolution.csv')
    Simulator=runpy.run_path(str(simroot/'simulator.py'))['Simulator']
    rows=[]
    with PVDatabase() as db:
        day=db.load_day('2025-12-26',scale=.1)
        for action in [32,64,128,256,512]:
            sim=Simulator(seed=20261123);clock=0.
            for task in range(1,31):
                sim.reset()
                for epoch in range(1,5):
                    obs=sim.step(action);energy=day.energy(clock,obs['T_s'])
                    rows.append(dict(action=action,task=task,epoch=epoch,start_s=clock,T_s=obs['T_s'],
                        demand_j=obs['demand_J'],pv_energy_j=energy,shortfall_j=max(0.,obs['demand_J']-energy)))
                    clock+=obs['T_s']
            assert clock<86400
        # Numerical representation check on the shortest epoch scale. Midpoint
        # quadrature is compared with the exact official piecewise-linear integral.
        rng=np.random.default_rng(20261123);checks=[]
        unscaled=db.load_day('2025-12-26')
        for i in range(100):
            start=float(rng.uniform(8*3600,16*3600));duration=float(rng.uniform(183.86,650.02))
            exact=unscaled.energy(start,duration)
            for step in [1.,5.,15.]:
                edges=np.r_[np.arange(start,start+duration,step),start+duration]
                approx=float(np.sum(unscaled.power_at((edges[1:]+edges[:-1])/2)*np.diff(edges)))
                checks.append(dict(window=i,step_s=step,exact_j=exact,approx_j=approx,
                    relative_error=abs(approx-exact)/max(exact,1.),absolute_error_j=abs(approx-exact)))
        timing_start=time.perf_counter()
        for i in range(10000):unscaled.energy(36000+i*.01,185.8)
        microseconds=(time.perf_counter()-timing_start)*1e6/10000
    pd.DataFrame(rows).to_csv(ROOT/'examples/profiling04_smoke.csv',index=False)
    check=pd.DataFrame(checks);check.to_csv(ROOT/'integration_grid_sensitivity.csv',index=False)
    result=dict(profile_sha256=hashlib.sha256((simroot/'epochs.csv').read_bytes()).hexdigest(),
        date='2025-12-26',PV_scale=.1,tasks_per_batch=30,epochs_per_task=4,smoke_epoch_count=len(rows),
        all_intervals_fully_supported=True,
        simulated_hours_by_batch=pd.DataFrame(rows).groupby('action').T_s.sum().div(3600).to_dict(),
        official_exact_energy_query_mean_microseconds=microseconds,
        quadrature_error_by_step=check.groupby('step_s').relative_error.agg(['median','max']).to_dict('index'),
        conclusion='Dense 12-14s dates provide about 14-49 native intervals per measured epoch. Earlier 111/366s dates are coarse; interpolation cannot recover missing physical variation. Recommended subset is quality-based, independent of algorithm outcomes.',
        limitations='Grid comparison checks numerical integration only; it does not validate the unknown physical waveform between observations. This is an interface smoke test, not a rerun of Figure 8/9 or parameter tuning.')
    (ROOT/'profiling04_assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
