"""Equal-date V curves and paired week-bootstrap intervals; no point selection."""
import json
import numpy as np
import pandas as pd
import runtime as rt
from experiment import checked


def bootstrap(values,dates):
    iso=pd.to_datetime(dates).isocalendar()
    week=np.asarray(iso.year.astype(str)+'-'+iso.week.astype(str));keys=np.unique(week)
    sums=np.array([values[week==k].sum(axis=0) for k in keys]);counts=np.array([(week==k).sum() for k in keys])
    rng=np.random.default_rng(2026112305)
    draws=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    return draws@sums/(draws@counts)[:,None]


def main():
    p=checked();raw=pd.read_csv(rt.ROOT/'results/evaluation_runs.csv.gz')
    configs=pd.read_csv(rt.ROOT/'data/configurations.csv').set_index('config_id');ids=configs.index.tolist()
    assert len(raw)==253440 and raw.groupby(['date','config_id']).size().eq(32).all()
    assert not raw.duplicated(['date','config_id','seed']).any()
    assert raw.groupby(['date','seed']).donor_sha256.nunique().eq(1).all()
    assert raw.queue_bound_slack.ge(-1e-10).all() and raw.hours.le(24).all()
    assert np.isfinite(raw.select_dtypes('number')).all().all()
    metrics=['TARS_W','DWAE','mean_Q_s','terminal_Q_s','max_Q_s','hours','T_s','demand_J','pv_J','shortfall_J',
             'mean_observation_age_s','max_observation_age_s','epochs_observation_age_over_300s']
    daily=raw.groupby(['date','config_id'])[metrics].mean()
    daily=daily.join(raw.groupby(['date','config_id']).agg(n=('seed','size'),violations=('violated','sum'),
        max_DWAE=('DWAE','max'),max_run_hours=('hours','max'),min_queue_bound_slack=('queue_bound_slack','min'))).reset_index()
    daily.to_csv(rt.ROOT/'results/daily_metrics.csv',index=False)
    subsets=dict(all_240=p['evaluation']['all_dates'],heldout_192=p['evaluation']['heldout_dates'],
                 strict_195=p['pv']['strict_dates'],recommended_142=p['pv']['recommended_dates'])
    tables=[];changes=[]
    for subset,dates in subsets.items():
        d=daily[daily.date.isin(dates)]
        assert len(d)==len(dates)*33
        table=d.groupby('config_id')[metrics].mean().loc[ids]
        table=table.rename(columns={k:'mean_'+k for k in metrics if not k.startswith('mean_')}).join(configs)
        table=table.join(d.groupby('config_id').agg(violations=('violations','sum'),max_DWAE=('max_DWAE','max'),
            max_run_hours=('max_run_hours','max'),min_queue_bound_slack=('min_queue_bound_slack','min')))
        table.insert(0,'subset',subset);table['days']=len(dates);table['runs_per_config']=32*len(dates)
        table['violation_fraction']=table.violations/table.runs_per_config
        boots={}
        for metric,prefix in [('TARS_W','TARS'),('mean_Q_s','Q'),('DWAE','DWAE')]:
            matrix=d.pivot(index='date',columns='config_id',values=metric).loc[dates,ids]
            boot=bootstrap(matrix.to_numpy(),matrix.index);boots[metric]=boot
            table[prefix+'_ci_low']=np.quantile(boot,.025,axis=0);table[prefix+'_ci_high']=np.quantile(boot,.975,axis=0)
        for si,sigma in enumerate(p['sigma_e']):
            for vi in range(len(p['V'])-1):
                left=f's{si}_v{vi:02d}';right=f's{si}_v{vi+1:02d}'
                i=ids.index(left);j=ids.index(right)
                gain=100*(1-boots['TARS_W'][:,j]/boots['TARS_W'][:,i])
                dq=boots['mean_Q_s'][:,j]-boots['mean_Q_s'][:,i]
                changes.append(dict(subset=subset,sigma_e=sigma,V_from=p['V'][vi],V_to=p['V'][vi+1],
                    TARS_gain_percent=100*(1-table.loc[right,'mean_TARS_W']/table.loc[left,'mean_TARS_W']),
                    gain_ci_low=np.quantile(gain,.025),gain_ci_high=np.quantile(gain,.975),
                    Q_change_s=table.loc[right,'mean_Q_s']-table.loc[left,'mean_Q_s'],
                    Q_change_ci_low=np.quantile(dq,.025),Q_change_ci_high=np.quantile(dq,.975)))
        tables.append(table.reset_index())
    summary=pd.concat(tables,ignore_index=True)
    summary.to_csv(rt.ROOT/'results/summary.csv',index=False)
    pd.DataFrame(changes).to_csv(rt.ROOT/'results/adjacent_V_changes.csv',index=False)
    summary[summary.subset.eq('all_240')].to_csv(rt.ROOT/'source_data/figure9_V_ab.csv',index=False)
    daily.to_csv(rt.ROOT/'source_data/daily_metrics.csv',index=False)
    previous=pd.read_csv(rt.ROOT.parent/'260k-section52-mean-observed-pv/results/evaluation_runs.csv.gz')
    previous=previous[(previous.policy=='LYA')&(previous.scale_name=='mean70')].set_index(['date','seed']).sort_index()
    current=raw[(raw.sigma_e==.24)&(raw.V==.1)].set_index(['date','seed']).sort_index()
    assert previous.index.equals(current.index)
    regression=float(abs(previous[metrics]-current[metrics]).to_numpy().max())
    np.testing.assert_allclose(previous[metrics],current[metrics],rtol=1e-12,atol=1e-7)
    report=dict(status='PASS',dates=240,configurations=33,runs=len(raw),epochs=int(raw.epochs.sum()),
        all_parameter_points_retained=True,figure_points_per_panel=33,
        all_day_all_seed_section52_regression_max_error=regression,max_run_hours=float(raw.hours.max()),
        min_queue_bound_slack=float(raw.queue_bound_slack.min()),mean_only_decisions=True,
        weights=[100,10,1],alpha=p['alpha'],figure_count=1,panels=['a','b'])
    (rt.ROOT/'results/analysis_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary[summary.subset.eq('all_240')][['sigma_e','V','mean_TARS_W','mean_Q_s','mean_DWAE','violations']].to_string(index=False),flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
