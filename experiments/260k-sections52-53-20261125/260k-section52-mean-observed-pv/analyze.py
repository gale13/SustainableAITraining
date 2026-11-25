"""Equal-day summaries for mean-only LYA, six baselines and previous-rule control."""
import json
import numpy as np
import pandas as pd
import runtime as rt

P=json.loads((rt.ROOT/'protocol.json').read_text())

def bootstrap(values,dates,n=2000):
    """Pair policies within date; resample whole calendar weeks, not epoch rows."""
    values=np.asarray(values);values=values[:,None] if values.ndim==1 else values
    iso=pd.to_datetime(dates).isocalendar();week=np.asarray(iso.year.astype(str)+'-'+iso.week.astype(str))
    keys=np.unique(week);sums=np.array([values[week==k].sum(axis=0) for k in keys]);counts=np.array([(week==k).sum() for k in keys])
    rng=np.random.default_rng(20261123);draw=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=n)
    return draw@sums/(draw@counts)[:,None]

def main():
    data=pd.read_csv(rt.ROOT/'results/evaluation_runs.csv.gz')
    quality=pd.read_csv(rt.ROOT/'data/date_quality.csv')
    assert len(data)==240*6*32*8 and data.hours.le(24).all()
    assert data.groupby(['date','scale_name','seed']).donor_sha256.nunique().eq(1).all()
    numerical=['TARS_W','DWAE','hours','shortfall_J','demand_J','pv_J','T_s','epsilon_s','terminal_Q_s','mean_Q_s']
    counts=[f'action_{a}_count' for a in rt.ACTIONS]
    daily=data.groupby(['date','scale_name','alpha','policy'],as_index=False)[numerical+counts].mean()
    observed=data.groupby(['date','scale_name','policy'],as_index=False).agg(max_DWAE=('DWAE','max'),violations=('DWAE',lambda a:int((a>.24).sum())))
    daily=daily.merge(observed,on=['date','scale_name','policy'])
    daily.to_csv(rt.ROOT/'results/daily_metrics.csv',index=False)
    strict=P['pv']['strict_dates'];holdout=P['evaluation']['heldout_dates']
    subsets={'all_240':P['evaluation']['all_dates'],'heldout_192':holdout,'strict_195':strict,
       'ten_minute_213':P['pv']['ten_minute_dates'],'recommended_142':P['pv']['recommended_dates'],
       'strict_heldout':[x for x in strict if x in holdout]}
    summary=[];comparisons=[];daily_diff=[]
    for subset,dates in subsets.items():
        for (scale,alpha),g in daily[daily.date.isin(dates)].groupby(['scale_name','alpha']):
            times=g.pivot(index='date',columns='policy',values='TARS_W').sort_index()
            assert len(times)==len(dates)
            boot=bootstrap(times.to_numpy(),times.index)
            for index,policy in enumerate(times.columns):
                rows=g[g.policy.eq(policy)].set_index('date').loc[times.index]
                summary.append(dict(subset=subset,scale_name=scale,alpha=alpha,policy=policy,days=len(rows),
                    noise_replicates=32,mean_TARS_W=rows.TARS_W.mean(),TARS_ci_low_W=np.quantile(boot[:,index],.025),TARS_ci_high_W=np.quantile(boot[:,index],.975),
                    mean_DWAE=rows.DWAE.mean(),pooled_TARS_W=rows.shortfall_J.sum()/rows.T_s.sum(),
                    pooled_DWAE=rows.epsilon_s.sum()/rows.T_s.sum(),max_run_DWAE=rows.max_DWAE.max(),
                    violations=int(rows.violations.sum()),mean_hours=rows.hours.mean(),mean_shortfall_kWh=rows.shortfall_J.mean()/3.6e6,
                    mean_demand_kWh=rows.demand_J.mean()/3.6e6,mean_PV_kWh=rows.pv_J.mean()/3.6e6,
                    mean_Q_s=rows.mean_Q_s.mean(),mean_terminal_Q_s=rows.terminal_Q_s.mean()))
            lya=g[g.policy.eq('LYA')]
            baselines=g[g.policy.isin(rt.POLICIES[1:])].groupby('policy').agg(tars=('TARS_W','mean'),maxd=('max_DWAE','max'))
            feasible=baselines[baselines.maxd.le(.24)]
            best=feasible.tars.idxmin();unconstrained=baselines.tars.idxmin()
            i=times.columns.get_loc('LYA');j=times.columns.get_loc(best)
            diff=times[best]-times.LYA;gainboot=100*(1-boot[:,i]/boot[:,j])
            all_better=times.LYA.lt(times[list(rt.POLICIES[1:])].min(axis=1)-1e-9)
            feasible_better=times.LYA.lt(times[list(feasible.index)].min(axis=1)-1e-9)
            comparisons.append(dict(subset=subset,scale_name=scale,alpha=alpha,days=len(times),
                LYA_TARS_W=times.LYA.mean(),LYA_DWAE=lya.DWAE.mean(),LYA_max_DWAE=lya.max_DWAE.max(),
                LYA_violations=int(lya.violations.sum()),best_feasible_baseline=best,baseline_TARS_W=times[best].mean(),
                gain_percent=100*(1-times.LYA.mean()/times[best].mean()),gain_W=diff.mean(),
                gain_ci_low_percent=np.quantile(gainboot,.025),gain_ci_high_percent=np.quantile(gainboot,.975),
                best_any_baseline=unconstrained,best_any_baseline_TARS_W=times[unconstrained].mean(),
                mean_rank_all=int((times[list(rt.POLICIES[1:])].mean()<times.LYA.mean()-1e-9).sum())+1,
                days_beating_global_best=int((diff>1e-9).sum()),days_beating_every_feasible=int(feasible_better.sum()),
                days_beating_every_baseline=int(all_better.sum())))
            if subset=='all_240':
                for date,value in diff.items():daily_diff.append(dict(date=date,scale_name=scale,alpha=alpha,
                  best_global_baseline=best,gain_W=value,LYA_TARS_W=times.loc[date,'LYA'],baseline_TARS_W=times.loc[date,best]))
    summary=pd.DataFrame(summary);comparisons=pd.DataFrame(comparisons)
    summary.to_csv(rt.ROOT/'results/summary.csv',index=False)
    comparisons.to_csv(rt.ROOT/'results/comparison.csv',index=False)
    pd.DataFrame(daily_diff).merge(quality,on='date').to_csv(rt.ROOT/'results/daily_comparison.csv',index=False)
    controls=[]
    for subset,dates in subsets.items():
        for scale,g in daily[daily.date.isin(dates)].groupby('scale_name'):
            values=g.pivot(index='date',columns='policy',values='TARS_W').sort_index()[['LYA','LYA-previous']]
            boot=bootstrap(values.to_numpy(),values.index)
            gain=100*(1-boot[:,0]/boot[:,1])
            controls.append(dict(subset=subset,scale_name=scale,
                current_TARS_W=values.LYA.mean(),previous_TARS_W=values['LYA-previous'].mean(),
                current_gain_percent=100*(1-values.LYA.mean()/values['LYA-previous'].mean()),
                ci_low_percent=np.quantile(gain,.025),ci_high_percent=np.quantile(gain,.975)))
    pd.DataFrame(controls).to_csv(rt.ROOT/'results/comparison_to_previous.csv',index=False)
    current=data[data.policy=='LYA']
    ages=current.groupby(['date','scale_name'],as_index=False).agg(
        mean_observation_age_s=('mean_observation_age_s','mean'),
        max_observation_age_s=('max_observation_age_s','max'),
        epochs_over_300s=('epochs_observation_age_over_300s','sum'),
        epochs_over_600s=('epochs_observation_age_over_600s','sum'),
        total_epochs=('epochs','sum'))
    ages['fraction_epochs_over_300s']=ages.epochs_over_300s/ages.total_epochs
    ages.to_csv(rt.ROOT/'results/observation_age_daily.csv',index=False)
    # The user asked for the average of day-wise experiments. Never average PV
    # curves first and run one surrogate day: shortfall and policy are nonlinear.
    audit=dict(status='PASS',days=240,noise_replicates=32,settings=6,policy_variants=8,total_runs=len(data),
       total_simulated_epochs=int(data.epochs.sum()),max_run_hours=float(data.hours.max()),
       paired_donors=True,all_runs_start_at_midnight=True,all_runs_finish_same_day=True,
       all_dates_equal_weight=True,selected_parameters_frozen=True,PV_gap_rule='None (explicit official unlimited interpolation)',
       decreasing_weights=[100,10,1],control='LYA-previous = historical noise-distribution and future-integrated-PV LYA',
       decision='MLP means and last real observed PV power times m_T; no noise scenarios or PV integrals',
       data_origin_counts=data.data_origin.value_counts().to_dict(),
       simulator_unchanged=all(rt.sha(rt.SIM/n)==h for n,h in P['preserved_simulator'].items()))
    assert audit['simulator_unchanged']
    (rt.ROOT/'results/audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(comparisons[comparisons.subset.isin(['all_240','heldout_192'])][['subset','scale_name','LYA_TARS_W','best_feasible_baseline','baseline_TARS_W','gain_percent','gain_ci_low_percent','gain_ci_high_percent','mean_rank_all','LYA_violations']].to_string(index=False))

if __name__=='__main__':main()
