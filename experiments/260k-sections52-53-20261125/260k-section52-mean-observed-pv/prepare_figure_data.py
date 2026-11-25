"""Build all plotted daily values directly from the revised result tables."""
import json
import numpy as np
import pandas as pd
import runtime as rt


def main():
    p=json.loads((rt.ROOT/'protocol.json').read_text())
    config=json.loads((rt.ROOT/'results/selected.json').read_text())['mean70']
    assert config['weights']==[100.,10.,1.]
    dates=p['evaluation']['all_dates'];alpha=config['alpha']
    daily=pd.read_csv(rt.ROOT/'results/daily_metrics.csv')
    daily=daily[(daily.scale_name=='mean70') & daily.policy.isin(rt.POLICIES)].sort_values(['date','policy'])
    assert len(daily)==240*7 and daily.groupby('policy').size().eq(240).all()
    assert not daily.duplicated(['date','policy']).any()
    summary=pd.read_csv(rt.ROOT/'results/summary.csv')
    summary=summary[(summary.subset=='all_240')&(summary.scale_name=='mean70')&summary.policy.isin(rt.POLICIES)].set_index('policy').loc[rt.POLICIES]
    means=daily.groupby('policy')[['TARS_W','DWAE']].mean().loc[rt.POLICIES]
    np.testing.assert_allclose(summary.mean_TARS_W,means.TARS_W,atol=1e-10)
    np.testing.assert_allclose(summary.mean_DWAE,means.DWAE,atol=1e-12)
    feasible=daily[(daily.policy!='LYA') & (daily.max_DWAE<=.24)]
    best=feasible.sort_values(['TARS_W','policy']).groupby('date').first().loc[dates]
    lya=daily[daily.policy=='LYA'].set_index('date').loc[dates]
    assert len(best)==240 and lya.max_DWAE.le(.24).all()
    tiles=pd.DataFrame(dict(date=dates,grid_row=np.arange(240)//12+1,grid_column=np.arange(240)%12+1,
        best_feasible_baseline=best.policy.values,best_feasible_baseline_TARS_W=best.TARS_W.values,
        LYA_TARS_W=lya.TARS_W.values,LYA_DWAE=lya.DWAE.values))
    tiles['LYA_advantage_W']=tiles.best_feasible_baseline_TARS_W-tiles.LYA_TARS_W
    tiles['outcome']=np.where(tiles.LYA_advantage_W>1e-10,'win',np.where(tiles.LYA_advantage_W< -1e-10,'loss','tie'))
    quality=pd.read_csv(rt.ROOT/'data/date_quality.csv')
    tiles=tiles.merge(quality[['date','strict_complete','max_support_gap_s','unlimited_mean_W']],on='date',validate='one_to_one')
    assert rt.sha(rt.PV_PACKAGE/'src/opencem_pv/data/opencem_pv.sqlite')==p['pv']['database_sha256']
    with rt.PVDatabase() as db:
        tiles['PV_generation_kWh']=[db.load_day(d,scale=alpha,max_gap_s=None).energy(0,86400)/3.6e6 for d in dates]
    error=float(abs(tiles.PV_generation_kWh-tiles.unlimited_mean_W*24/1000*alpha).max())
    assert error<1e-7
    tiles['PV_scale']=alpha
    tiles['PV_mean_power_W']=tiles.PV_generation_kWh*1000/24
    tiles['used_for_tuning']=tiles.date.isin(p['tuning']['dates'])
    assert not tiles.isna().any().any()
    source=rt.ROOT/'source_data'
    tiles.to_csv(source/'daily_tiles.csv',index=False)
    daily.to_csv(source/'daily_scatter.csv',index=False)
    summary.to_csv(source/'algorithm_summary.csv')
    meta=dict(LYA_parameters=config,sigma_e=.24,tile_count=len(tiles),scatter_points=len(daily),
        outcome_counts=tiles.outcome.value_counts().to_dict(),api_energy_max_error_kWh=error,
        all_dates_PV_percent=100*tiles.PV_mean_power_W.mean()/p['reference_power_W'],
        source_sha256={f.name:rt.sha(f) for f in source.glob('*.csv')})
    (rt.ROOT/'figure_data.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
