"""Validate causal mean-only decisions and independent noisy environment replay."""
import importlib.util
import json
import numpy as np
import pandas as pd
import runtime as rt
from pv_observations import latest


def main():
    spec=importlib.util.spec_from_file_location('frozen260k',rt.SIM/'simulator.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    reference=mod.Simulator(seed=42);rng=np.random.default_rng(20261125)
    dates=pd.read_csv(rt.ROOT/'data/date_quality.csv').query('unlimited_complete').date.tolist()
    pv_error=0.;lookup_error=0.;windows=0;lookups=0;midnight_age=[]
    for day in dates:
        pv=rt.set_day(day,None);times,powers=rt.OBSERVATIONS
        candidates=np.r_[0.,times[(times>=0)&(times<86400)][:2],rng.uniform(0,85000,8)]
        for t in candidates:
            actual=np.zeros(3);rt.LIB.latest_observation(float(t),4.1928776597152995,actual)
            i=np.flatnonzero(times<=t)[-1]
            expected=np.array([powers[i]*4.1928776597152995,times[i],t-times[i]])
            lookup_error=max(lookup_error,float(abs(actual-expected).max()));lookups+=1
            np.testing.assert_array_equal(actual,expected)
            assert actual[1]<=t and actual[2]>=0
        midnight_age.append(latest(times,powers,0)[2])
        for t,d in [(0.,86400.),(43200.125,632.271),(85000.25,1000.5),(float(rng.uniform(0,80000)),185.8)]:
            expected=pv.energy(t,d)*4.1928776597152995
            actual=rt.LIB.pv_energy(t,d,4.1928776597152995)
            pv_error=max(pv_error,abs(actual-expected));windows+=1
            np.testing.assert_allclose(actual,expected,atol=.002,rtol=1e-9)
    states=[(r.J,int(r.phi),int(r.action),int(r.epoch)-1) for r in pd.read_csv(rt.SIM/'epochs.csv').itertuples()]
    states.extend((rng.uniform(.15,1),int(rng.integers(1,4)),int(rng.choice(rt.ACTIONS)),int(rng.integers(0,4))) for _ in range(150))
    rt.set_day('2025-12-26',None);model_error=0.;mean_preservation_error=0.
    for j,phase,a,e in states:
        expected=np.array(list(reference.means(j,phase,a).values()))
        actual=rt.means(j,phase,a)
        model_error=max(model_error,float(abs(actual-expected).max()))
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-11)
        _,cases=rt.inspect(j,phase,a,e,43200.,4.1928776597152995)
        np.testing.assert_allclose(cases,reference.scenarios(j,phase,a,e),atol=2e-8,rtol=1e-10)
        mean_preservation_error=max(mean_preservation_error,float(abs(cases[:,:3].mean(axis=0)-actual).max()))
        np.testing.assert_allclose(cases[:,:3].mean(axis=0),actual,atol=2e-8,rtol=1e-10)

    # No nuisance realization, empirical-bank distribution or future PV is in
    # the pure controller signature. Exercise the full decision wrapper too.
    decision_args=(.31,2,3.5,43200.25,.1,10.,4.1928776597152995)
    pv=rt.set_day('2025-12-26',None);times,powers=[x.copy() for x in rt.OBSERVATIONS]
    baseline=rt.decision(*decision_args)
    def same():
        current=rt.decision(*decision_args)
        assert current[0]==baseline[0]
        np.testing.assert_array_equal(current[1],baseline[1]);np.testing.assert_array_equal(current[2],baseline[2])
    changed=powers.copy();changed[times>decision_args[3]]=10000+7*changed[times>decision_args[3]]
    rt.LIB.set_observations(times,changed,len(times))
    rt.LIB.set_pv(np.array([0.,43200.25,86400.]),np.array([0.,1.,1e8]),3)
    same()
    past=times<=decision_args[3]
    rt.LIB.set_observations(np.ascontiguousarray(times[past]),np.ascontiguousarray(powers[past]),int(past.sum()))
    rt.LIB.set_pv(np.array([0.,43200.25]),np.array([0.,0.]),2)
    same()  # no future PV support exists, yet decision still works
    rt.set_day('2025-12-26',None)
    _,before_cases=rt.inspect(.31,2,128,2,43200.25,4.1928776597152995)
    changed_bank=rt.BANK.copy();changed_bank[0,:,:,0]*=2.;changed_bank[0,:,:,1]*=3.;changed_bank[0,:,:,2]+=.05
    rt.load(changed_bank);same()
    _,after_cases=rt.inspect(.31,2,128,2,43200.25,4.1928776597152995)
    assert not np.allclose(before_cases,after_cases)
    rt.load();rt.set_day('2025-12-26',None)

    epochs=0;nonzero_noise=0;observation_ages=[];score_error=0.
    for date in ['2025-07-13','2025-12-26','2026-04-11']:
        pv=rt.set_day(date,None);times,powers=rt.OBSERVATIONS
        for V in [.0001,.01,.1,1.]:
            result,trace=rt.run('LYA',4.1928776597152995,120000,V=V,trace=True)
            Q=0.;clock=0.
            for row in trace.itertuples():
                e=int(row.epoch)-1
                if e==0:reference.reset()
                np.testing.assert_allclose([row.J,row.phase,row.Q_before_s,row.clock_start_s],[reference.J,reference.phase,Q,clock],rtol=1e-10,atol=1e-9)
                power,observed_time,age=latest(times,powers,clock,4.1928776597152995)
                candidates=np.array([list(reference.means(reference.J,reference.phase,a).values()) for a in rt.ACTIONS])
                predicted=np.maximum(0,candidates[:,1]-power*candidates[:,0])
                w=[100,10,1][reference.phase-1]
                scores=(V*predicted+w*Q*(candidates[:,2]-.24*candidates[:,0]))/candidates[:,0]
                a=rt.ACTIONS[int(np.argmin(scores))];assert row.action==a
                idx=rt.ACTIONS.index(a)
                np.testing.assert_allclose(row.theta,scores[idx],rtol=1e-8,atol=1e-8)
                score_error=max(score_error,abs(row.theta-scores[idx]))
                np.testing.assert_allclose([row.observed_power_W,row.observed_time_s,row.observation_age_s],[power,observed_time,age],atol=1e-8)
                np.testing.assert_allclose(row.estimated_PV_J,power*candidates[idx,0],rtol=1e-10)
                np.testing.assert_allclose(row.estimated_shortfall_J,predicted[idx],rtol=1e-10,atol=1e-8)
                actual=reference.scenarios(reference.J,reference.phase,a,e)[int(row.donor_task)-1]
                np.testing.assert_allclose([row.T_s,row.demand_J,row.epsilon_s,row.J_next],actual,rtol=1e-10,atol=1e-8)
                np.testing.assert_allclose(row.renewable_J,4.1928776597152995*pv.energy(clock,row.T_s),rtol=1e-8,atol=.002)
                nonzero_noise+=int(not np.allclose(actual[:3],candidates[idx],rtol=1e-7,atol=1e-7))
                Q=max(0,Q+row.epsilon_s-.24*row.T_s)
                reference.phase=mod.phase_after(reference.phase,reference.J,row.J_next,reference.config)
                reference.J=row.J_next;clock+=row.T_s;epochs+=1;observation_ages.append(age)
            np.testing.assert_allclose(result['TARS_W'],trace.shortfall_J.sum()/trace.T_s.sum())
    assert nonzero_noise>0

    old=pd.read_csv(rt.ROOT.parent/'260k-section52-decreasing-weight/results/evaluation_runs.csv.gz')
    regression=0.
    for date in ['2025-07-13','2025-12-26','2026-04-11']:
        rt.set_day(date,None)
        for policy in rt.POLICIES[1:]:
            row,_=rt.run(policy,4.1928776597152995,120000)
            previous=old[(old.date==date)&(old.policy==policy)&(old.scale_name=='mean70')&(old.seed==120000)].iloc[0]
            for key in ['TARS_W','DWAE','T_s','shortfall_J','mean_Q_s']:
                regression=max(regression,abs(row[key]-previous[key]))
                np.testing.assert_allclose(row[key],previous[key],atol=1e-7,rtol=1e-12)
    report=dict(status='PASS',eligible_days=len(dates),past_observation_checks=lookups,max_observation_error=lookup_error,
        PV_windows=windows,max_environment_PV_energy_error_J=pv_error,MLP_states=len(states),max_MLP_error=model_error,
        max_mean_preservation_error=mean_preservation_error,future_PV_values_irrelevant=True,future_PV_support_not_required=True,
        environment_noise_bank_irrelevant_to_decision=True,environment_response_still_noise_sensitive=True,
        independent_python_replay_epochs=epochs,nonzero_environment_noise_epochs=nonzero_noise,max_score_error=score_error,
        baseline_regression_max_error=regression,midnight_observation_age_max_s=max(midnight_age),
        decision_inputs='MLP means, Q, phase weight, V, sigma_e, latest observed scaled PV power only')
    (rt.ROOT/'results/engine_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
