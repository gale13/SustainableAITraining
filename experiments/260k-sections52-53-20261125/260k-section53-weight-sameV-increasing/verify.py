"""Verify Increasing schedules against independent MLP/controller/environment code."""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import runtime as rt
from experiment import checked,BASE
from controller import MeanOnlyLYA
from pv_observations import latest


def main():
    p=checked();prior=json.loads((BASE/'verification.json').read_text())
    assert prior['status']=='PASS' and prior['engine_checks']['unchanged_engine_inherits_future_PV_and_noise_isolation_checks']
    for n in ['runtime.py','engine.cpp','controller.cpp','controller.py','pv_observations.py']:
        assert rt.sha(rt.ROOT/n)==rt.sha(BASE/n)
    spec=importlib.util.spec_from_file_location('frozen260k',rt.SIM/'simulator.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);reference=mod.Simulator(seed=42)
    import pandas as pd
    stored=pd.read_csv(rt.ROOT/'results/increasing_runs.csv.gz')
    max_saved_error=0.
    epochs=0;max_score_error=0.;max_PV_error=0.;nonzero_noise=0;standalone=0
    with tempfile.TemporaryDirectory(prefix='260k-weights-no-noise-') as temp:
        for n in ['models.json','model_config.json']:shutil.copy2(rt.SIM/n,Path(temp)/n)
        assert len(list(Path(temp).iterdir()))==2
        for date in ['2025-07-13','2025-12-26','2026-04-11']:
            pv=rt.set_day(date,None);times,powers=rt.OBSERVATIONS
            sigma=.235
            for V in [1e-6,3e-5,1e-4,.1,100.]:
                for name,weights in [('Increasing',p['weights']['Increasing'])]:
                    controller=MeanOnlyLYA(temp,V=V,weights=weights,sigma_e=sigma)
                    result,trace=rt.run('LYA',p['alpha'],120000,V=V,weights=weights,sigma=sigma,trace=True)
                    saved=stored[stored.date.eq(date)&stored.seed.eq(120000)&np.isclose(stored.V,V,rtol=1e-12,atol=0)]
                    assert len(saved)==1
                    metric_names=['TARS_W','DWAE','mean_Q_s','T_s','demand_J','shortfall_J']
                    expected=np.array([result[k] for k in metric_names]);actual_saved=saved[metric_names].to_numpy()[0]
                    np.testing.assert_allclose(expected,actual_saved,rtol=1e-12,atol=1e-7)
                    max_saved_error=max(max_saved_error,float(abs(expected-actual_saved).max()))
                    Q=0.;clock=0.
                    for row in trace.itertuples():
                        e=int(row.epoch)-1
                        if e==0:reference.reset()
                        np.testing.assert_allclose([row.J,row.phase,row.Q_before_s,row.clock_start_s],[reference.J,reference.phase,Q,clock],rtol=1e-10,atol=1e-8)
                        power,observed,age=latest(times,powers,clock,p['alpha'])
                        decision=controller.select(reference.J,reference.phase,Q,power)
                        assert row.action==decision['action'];standalone+=1
                        candidates=np.array([list(reference.means(reference.J,reference.phase,a).values()) for a in rt.ACTIONS])
                        np.testing.assert_allclose(decision['means'],candidates,atol=1e-8,rtol=1e-10)
                        idx=rt.ACTIONS.index(row.action);score=decision['scores'][idx]
                        max_score_error=max(max_score_error,abs(row.theta-score))
                        np.testing.assert_allclose(row.theta,score,rtol=1e-8,atol=1e-7)
                        assert row.phase_weight==weights[int(row.phase)-1] and observed<=clock
                        np.testing.assert_allclose(row.estimated_PV_J,power*candidates[idx,0],rtol=1e-10,atol=1e-8)
                        actual=reference.scenarios(reference.J,reference.phase,row.action,e)[int(row.donor_task)-1]
                        np.testing.assert_allclose([row.T_s,row.demand_J,row.epsilon_s,row.J_next],actual,rtol=1e-10,atol=1e-8)
                        ren=p['alpha']*pv.energy(clock,row.T_s);max_PV_error=max(max_PV_error,abs(row.renewable_J-ren))
                        np.testing.assert_allclose(row.renewable_J,ren,rtol=1e-8,atol=.002)
                        nonzero_noise+=int(not np.allclose(actual[:3],candidates[idx],rtol=1e-7,atol=1e-7))
                        Q=max(0,Q+actual[2]-sigma*actual[0])
                        reference.phase=mod.phase_after(reference.phase,reference.J,actual[3],reference.config)
                        reference.J=actual[3];clock+=actual[0];epochs+=1
                    np.testing.assert_allclose([result['TARS_W'],result['mean_Q_s'],result['DWAE']],
                        [trace.shortfall_J.sum()/trace.T_s.sum(),trace.Q_before_s.mean(),trace.epsilon_s.sum()/trace.T_s.sum()])
    report=dict(status='PASS',independent_python_replay_epochs=epochs,standalone_no_noise_file_decisions=standalone,
        increasing_verified=True,inherited_decay_replay_epochs=9000,max_saved_metric_error=max_saved_error,verified_sigma=.235,verified_V=[1e-6,3e-5,1e-4,.1,100.],nonzero_noise_epochs=nonzero_noise,
        max_score_error=max_score_error,max_environment_PV_error_J=max_PV_error,
        mean_only_and_past_only_controller_verified=True,
        unchanged_engine_inherits_future_PV_and_noise_isolation_checks=True)
    (rt.ROOT/'results/engine_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
