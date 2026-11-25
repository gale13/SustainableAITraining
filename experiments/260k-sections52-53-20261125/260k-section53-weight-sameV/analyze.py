"""Same-V paired comparisons over the entire declared grid, with no row selection."""
import json
import numpy as np
import pandas as pd
import runtime as rt
from experiment import checked,BASE


def bootstrap(values,dates):
    iso=pd.to_datetime(dates).isocalendar()
    weeks=np.asarray(iso.year.astype(str)+'-'+iso.week.astype(str));keys=np.unique(weeks)
    sums=np.array([values[weeks==k].sum(axis=0) for k in keys]);counts=np.array([(weeks==k).sum() for k in keys])
    draws=np.random.default_rng(2026112305).multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    return draws@sums/(draws@counts)[:,None]


def main():
    p=checked();raw=pd.read_csv(rt.ROOT/'results/evaluation_runs.csv.gz')
    configs=pd.read_csv(rt.ROOT/'data/configurations.csv').set_index('config_id');ids=configs.index.tolist()
    assert len(raw)==652800 and len(configs)==85 and raw.groupby(['date','config_id']).size().eq(32).all()
    assert raw.groupby(['date','seed']).donor_sha256.nunique().eq(1).all()
    assert raw.sigma_e.eq(.235).all() and not raw.duplicated(['date','config_id','seed']).any()
    metrics=['TARS_W','DWAE','mean_Q_s','terminal_Q_s','max_Q_s','hours','T_s','demand_J','shortfall_J','pv_J',
             'mean_observation_age_s','epochs_observation_age_over_300s']
    daily=raw.groupby(['date','config_id'])[metrics].mean()
    daily=daily.join(raw.groupby(['date','config_id']).agg(violations=('violated','sum'),n=('seed','size'),
        max_DWAE=('DWAE','max'),max_run_hours=('hours','max'),min_queue_bound_slack=('queue_bound_slack','min'))).reset_index()
    daily.to_csv(rt.ROOT/'results/daily_metrics.csv',index=False)
    subsets=dict(all_240=p['evaluation']['all_dates'],heldout_192=p['evaluation']['heldout_dates'],
        strict_195=p['pv']['strict_dates'],recommended_142=p['pv']['recommended_dates'])
    summaries=[];comparisons=[]
    for subset,dates in subsets.items():
        d=daily[daily.date.isin(dates)];assert len(d)==len(dates)*85
        table=d.groupby('config_id')[metrics].mean().loc[ids]
        table=table.rename(columns={c:'mean_'+c for c in metrics if not c.startswith('mean_')}).join(configs)
        table=table.join(d.groupby('config_id').agg(violations=('violations','sum'),max_DWAE=('max_DWAE','max'),max_run_hours=('max_run_hours','max')))
        table['subset']=subset;table['days']=len(dates);table['runs_per_config']=len(dates)*32
        table['violation_fraction']=table.violations/table.runs_per_config
        boots={};matrices={}
        for metric,tag in [('TARS_W','TARS'),('DWAE','DWAE'),('mean_Q_s','Q'),('hours','hours')]:
            matrix=d.pivot(index='date',columns='config_id',values=metric).loc[dates,ids]
            boot=bootstrap(matrix.to_numpy(),matrix.index);boots[metric]=boot;matrices[metric]=matrix
            table[tag+'_ci_low']=np.quantile(boot,.025,axis=0);table[tag+'_ci_high']=np.quantile(boot,.975,axis=0)
        for vi,V in enumerate(p['V']):
            fixed=f'v{vi:02d}_w4';j=ids.index(fixed)
            for fi,family in enumerate(list(p['weights'])[:4]):
                decay=f'v{vi:02d}_w{fi}';i=ids.index(decay)
                delta=boots['TARS_W'][:,j]-boots['TARS_W'][:,i];gain=100*delta/boots['TARS_W'][:,j]
                diff=matrices['TARS_W'][fixed]-matrices['TARS_W'][decay]
                comparisons.append(dict(subset=subset,sigma_e=.235,V=V,decay_family=family,comparator='Fixed',
                    decreasing_TARS_W=table.loc[decay,'mean_TARS_W'],fixed_TARS_W=table.loc[fixed,'mean_TARS_W'],
                    TARS_reduction_W=table.loc[fixed,'mean_TARS_W']-table.loc[decay,'mean_TARS_W'],
                    reduction_ci_low_W=np.quantile(delta,.025),reduction_ci_high_W=np.quantile(delta,.975),
                    gain_percent=100*(1-table.loc[decay,'mean_TARS_W']/table.loc[fixed,'mean_TARS_W']),
                    gain_ci_low_percent=np.quantile(gain,.025),gain_ci_high_percent=np.quantile(gain,.975),
                    decreasing_DWAE=table.loc[decay,'mean_DWAE'],fixed_DWAE=table.loc[fixed,'mean_DWAE'],
                    decreasing_Q_s=table.loc[decay,'mean_Q_s'],fixed_Q_s=table.loc[fixed,'mean_Q_s'],
                    decreasing_violations=int(table.loc[decay,'violations']),fixed_violations=int(table.loc[fixed,'violations']),
                    days_won=int((diff>1e-9).sum()),days_lost=int((diff< -1e-9).sum()),days_tied=int((abs(diff)<=1e-9).sum()),days=len(dates)))
        summaries.append(table.reset_index())
    summary=pd.concat(summaries,ignore_index=True);pairs=pd.DataFrame(comparisons)
    summary.to_csv(rt.ROOT/'results/summary.csv',index=False);pairs.to_csv(rt.ROOT/'results/paired_comparisons.csv',index=False)
    primary=pairs[pairs.subset.eq('all_240')].copy()
    primary.to_csv(rt.ROOT/'source_data/sameV_curves.csv',index=False)
    summary[summary.subset.eq('all_240')].to_csv(rt.ROOT/'source_data/absolute_metrics.csv',index=False)
    daily.to_csv(rt.ROOT/'source_data/daily_metrics.csv',index=False)
    peaks=primary.loc[primary.groupby('decay_family').gain_percent.idxmax()].copy()
    peaks['interpretation']='Descriptive maximum on the complete predeclared grid, not independently validated optimum'
    peaks.to_csv(rt.ROOT/'results/descriptive_peaks.csv',index=False)
    regression_error=0.;regression_runs=0
    old=pd.read_csv(BASE/'results/evaluation_runs.csv.gz')
    old=old[old.sigma_e.eq(.235)]
    for V in sorted(old.V.unique()):
        a=old[np.isclose(old.V,V,rtol=1e-12,atol=0)].set_index(['date','seed']).sort_index()
        b=raw[raw.weight_schedule.eq('Exponential')&np.isclose(raw.V,V,rtol=1e-12,atol=0)].set_index(['date','seed']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[metrics],b[metrics],rtol=1e-12,atol=1e-7)
        regression_error=max(regression_error,float(abs(a[metrics]-b[metrics]).to_numpy().max()));regression_runs+=len(a)
    oldpath=rt.ROOT.parent/'260k-section53-weight-families-fixedV/results/evaluation_runs.csv.gz'
    old=pd.read_csv(oldpath);old=old[old.sigma_e.eq(.235)&~old.weight_schedule.isin(['Increasing','Exponential'])]
    for family in old.weight_schedule.unique():
        a=old[old.weight_schedule.eq(family)].set_index(['date','seed']).sort_index()
        b=raw[raw.weight_schedule.eq(family)&raw.V.eq(.1)].set_index(['date','seed']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[metrics],b[metrics],rtol=1e-12,atol=1e-7)
        regression_error=max(regression_error,float(abs(a[metrics]-b[metrics]).to_numpy().max()));regression_runs+=len(a)
    audit=dict(status='PASS',dates=240,configurations=85,runs=len(raw),epochs=int(raw.epochs.sum()),sigma_e=.235,
        V_values=len(p['V']),all_curves_and_dates_retained=True,paired_effects=68,
        regression_runs=regression_runs,regression_max_error=regression_error,
        additional_regression_source_sha256=rt.sha(oldpath),max_run_hours=float(raw.hours.max()),
        max_DWAE=float(raw.DWAE.max()),total_violations=int(raw.violated.sum()),
        min_queue_bound_slack=float(raw.queue_bound_slack.min()))
    (rt.ROOT/'results/analysis_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(primary.pivot(index='V',columns='decay_family',values='gain_percent').to_string())
    print(peaks[['decay_family','V','gain_percent','gain_ci_low_percent','gain_ci_high_percent','decreasing_violations','fixed_violations']].to_string(index=False))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
