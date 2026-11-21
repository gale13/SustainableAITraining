"""Verify full-data usage, causal state, mean/noise consistency and preservation."""
import hashlib,json
from zipfile import ZipFile
import numpy as np
import pandas as pd
from simulator import ROOT,ACTIONS,METRICS,EPS,features,predict,phase_after,Simulator
from fitting_core import rebuild_states

def run():
    cfg=json.loads((ROOT/'model_config.json').read_text())
    recipe=json.loads((ROOT/'calibration_recipe.json').read_text())
    data=pd.read_csv(ROOT/'epochs.csv');lib=pd.read_csv(ROOT/'noise_library.csv')
    models=json.loads((ROOT/'models.json').read_text());sim=Simulator(seed=42)
    assert len(data)==len(lib)==600 and sim.bank.shape==(30,5,4,3)
    assert data.calibration_role.eq('full_data').all() and 'split' not in data
    assert data.original_split.value_counts().to_dict()=={'train':360,'test':120,'validation':120}
    assert cfg['fit_tasks']==cfg['noise_tasks']==list(range(1,31))
    assert cfg['held_out_observations']==0
    np.testing.assert_array_equal(sim.donor_tasks,np.arange(1,31))
    assert data.groupby(['action','epoch']).size().eq(30).all()
    for key,model in models.items():
        assert model['converged']
        for field in ['width','penalty','parameters']:assert model[field]==recipe['fixed_models'][key][field]
    np.testing.assert_array_equal(cfg['beta_internal'],recipe['beta_internal'])
    with ZipFile(ROOT/'source.zip') as z:
        raw=pd.read_csv(z.open('epochs.csv')).rename(columns={'batch':'action'})
        initial={(r['task'],r['batch']):r['train_probe_loss'] for r in json.load(z.open('baselines.json'))}
    rebuilt=rebuild_states(raw,initial,cfg['beta_internal'])
    cols=['J','J_next','phi','phi_next_within_task','T','dem','epsilon']
    np.testing.assert_allclose(data[cols],rebuilt[cols],rtol=1e-12)
    changed=raw.copy();changed.loc[(changed.task==1)&(changed.action==32)&(changed.epoch>=2),'train_loss']*=2
    alt=rebuild_states(changed,initial,cfg['beta_internal']);before=(data.task==1)&(data.action==32)&(data.epoch<=2)
    np.testing.assert_allclose(data.loc[before,['J','phi']],alt.loc[before,['J','phi']])
    # Separately rederive the running-minimum input/output state for each task.
    for _,group in data.groupby(['task','action']):
        minimum=group.iloc[0].L0;phase=1
        for row in group.itertuples():
            j=minimum/(row.L0+EPS);minimum=min(minimum,row.train_loss);jp=minimum/(row.L0+EPS)
            np.testing.assert_allclose([row.J,row.J_next,row.epsilon],[j,jp,row.T*jp])
            assert row.phi==phase;phase=phase_after(phase,j,jp,cfg)
    np.testing.assert_allclose([cfg['j_mean'],cfg['j_sd'],cfg['initial_loss_mean_calibration']],
        [data.J.mean(),data.J.std(ddof=0),data.loc[data.epoch.eq(1),'L0'].mean()])
    prediction=predict(models,features(data.J,data.phi,data.action,cfg),data.J.to_numpy())
    np.testing.assert_allclose(prediction,data[['pred_'+k for k in METRICS]],rtol=1e-10)
    for row in pd.read_csv(ROOT/'metrics.csv').itertuples():
        error=data[row.metric]-data['pred_'+row.metric]
        assert row.dataset=='all_calibration' and row.n==600
        np.testing.assert_allclose([row.mse,row.rmse],[np.mean(error**2),np.sqrt(np.mean(error**2))])
    ordered=data.sort_values(['task','action','epoch'])
    expected=np.column_stack([ordered['T']/ordered.pred_T,ordered.dem/ordered.pred_dem,
        ordered.J_next/ordered.J-ordered.pred_epsilon/(ordered.pred_T*ordered.J)])
    np.testing.assert_allclose(sim.bank,expected.reshape(30,5,4,3),rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(lib[['time_ratio','demand_ratio','retention_residual']],expected,rtol=1e-9,atol=1e-12)
    assert lib[['task','action','epoch']].equals(ordered[['task','action','epoch']])
    max_mean_error=0.
    # Check all 600 measured states, including the states formerly held out.
    for row in data.itertuples():
        cases=sim.scenarios(row.J,row.phi,row.action,row.epoch-1)
        mean=np.asarray(list(sim.means(row.J,row.phi,row.action).values()))
        np.testing.assert_allclose(cases[:,:3].mean(0),mean,rtol=1e-10,atol=1e-10)
        max_mean_error=max(max_mean_error,float(np.max(abs(cases[:,:3].mean(0)-mean))))
        assert (cases[:,:2]>0).all() and ((cases[:,3]>0)&(cases[:,3]<=row.J)).all()
        np.testing.assert_allclose(cases[:,2],cases[:,0]*cases[:,3])
    for action in ACTIONS:
        for mode in ['empirical','independent','trajectory','pooled','none']:
            sim.reset()
            for epoch in range(4):
                row=sim.step(action,mode)
                assert 0<row['J_next']<=row['J'] and row['phase_next']>=row['phase']
                for key in ['T_s','demand_J','epsilon_s']:
                    np.testing.assert_allclose(row[key],row['mean_'+key]+row['noise_'+key])
                if mode!='none':
                    assert row['noise_source']=='full_data_calibration'
                    assert row['noise_donor_task'] in range(1,31) and row['noise_donor_epoch']==epoch+1
            try:sim.step(action)
            except ValueError:pass
            else:raise AssertionError('Fifth epoch accepted.')
    default,alias=Simulator(seed=20261121),Simulator(seed=20261121)
    for action in [32,128,512,64]:assert default.step(action)==alias.step(action,'independent')
    old=ROOT.parent/recipe['feasibility_directory'];preserved='not present in this portable installation'
    if old.exists():
        snapshot=json.loads((ROOT/'feasibility_snapshot_sha256.json').read_text())
        current={str(f.relative_to(old)):hashlib.sha256(f.read_bytes()).hexdigest() for f in old.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        assert current==snapshot,'Feasibility experiment changed.'
        preserved=True
    result=dict(status='passed',fit_rows=600,noise_rows=600,noise_bank_shape=list(sim.bank.shape),
        noise_donors_per_action_epoch=30,held_out_rows=0,all_selected_models_converged=True,
        feasibility_files_unchanged=preserved,maximum_scenario_mean_absolute_error=max_mean_error,
        checks=['all original splits included','frozen recipe','causal paper state and scalar running minimum',
                'all-data feature scales and MLP exports','all-data empirical bank reconstruction',
                'all 600 conditional means and physical bounds','seeded default sampling and donor provenance'])
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
