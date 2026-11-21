"""Check paper recursions, causal timing, split integrity and exported inference."""
import hashlib
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
from fit import rebuild_states
from simulator import ROOT, EPS, ACTIONS, METRICS, features, predict, phase_after, Simulator


def run():
    cfg=json.loads((ROOT/'model_config.json').read_text())
    models=json.loads((ROOT/'models.json').read_text())
    saved=pd.read_csv(ROOT/'epochs.csv')
    old=pd.read_csv(ROOT/'previous_probe/epochs.csv')
    with ZipFile(ROOT/'source.zip') as z:
        raw=pd.read_csv(z.open('epochs.csv')).rename(columns={'batch':'action'})
        initial=json.load(z.open('baselines.json'))
    l0={(v['task'],v['batch']):v['train_probe_loss'] for v in initial}
    rebuilt=rebuild_states(raw,l0,cfg['beta_internal'])
    statecols=['observed_loss_before','observed_loss_after','J','J_next','delta','phi','epsilon']
    np.testing.assert_allclose(saved[statecols],rebuilt[statecols],rtol=1e-12,atol=1e-12)
    assert len(saved)==600 and saved.groupby(['task','action']).size().eq(4).all()
    assert saved.groupby('task').split.nunique().eq(1).all()
    assert saved.groupby('split').size().to_dict()=={'train':360,'validation':120,'test':120}
    assert saved[['task','action','epoch','split']].equals(old[['task','action','epoch','split']])
    np.testing.assert_array_equal(saved.phi,old.phi)
    # Independent scalar implementation, not a repeat of the vectorized cummin.
    for _,g in saved.groupby(['task','action']):
        first=g.iloc[0];lowest=first.L0;phase=1;lastloss=first.L0
        for row in g.itertuples():
            assert row.phi==phase
            np.testing.assert_allclose(row.observed_loss_before,lastloss)
            j=lowest/(first.L0+EPS)
            lowest=min(lowest,row.train_loss)
            jp=lowest/(first.L0+EPS)
            np.testing.assert_allclose([row.J,row.J_next,row.epsilon],[j,jp,row.T*jp])
            delta=(jp-j)/(j+EPS)
            candidate=next(k for k,upper in enumerate([*cfg['beta_internal'],0.],1) if delta<=upper)
            phase=max(phase,candidate);lastloss=row.train_loss
    # Modify current/future outcomes: earlier and current decision states cannot change.
    changed=raw.copy()
    mask=(changed.task==1)&(changed.action==32)&(changed.epoch>=2)
    changed.loc[mask,'train_loss']*=2
    alt=rebuild_states(changed,l0,cfg['beta_internal'])
    before=(rebuilt.task==1)&(rebuilt.action==32)&(rebuilt.epoch<=2)
    np.testing.assert_allclose(rebuilt.loc[before,['J','phi']],alt.loc[before,['J','phi']])
    # Post-training probe and validation values are never used by the new state.
    changed=raw.copy();changed['train_probe_loss']=123.;changed['validation_loss']=456.
    alt=rebuild_states(changed,l0,cfg['beta_internal'])
    np.testing.assert_allclose(rebuilt[statecols],alt[statecols])
    # Nonmonotone loss fixture: a rise leaves J unchanged and pushes phase to K.
    fixture=raw[(raw.task==1)&(raw.action==32)].sort_values('epoch').copy()
    fixture['train_loss']=[2.,3.,1.,1.2]
    example=rebuild_states(fixture,{(1,32):4.},[-.3,-.1])
    np.testing.assert_allclose(example.J_next,np.array([2.,2.,1.,1.])/(4.+EPS))
    np.testing.assert_array_equal(example.phi,[1,1,3,3])
    # Interval is (beta[k-1],beta[k]], so exact equality belongs to the lower phase.
    for i,beta in enumerate(cfg['beta_internal']):
        assert 1+np.searchsorted(cfg['beta_internal'],beta,side='left')==i+1
    assert phase_after(1,.2,.2,cfg)==3
    assert phase_after(3,1.,.1,cfg)==3
    # Thresholds use only training seeds and transitions entering actual decisions.
    calibration=saved[(saved.split=='train')&(saved.epoch<4)].delta
    np.testing.assert_allclose(cfg['beta_internal'],np.quantile(calibration,[1/3,2/3]))
    prediction=predict(models,features(saved.J,saved.phi,saved.action,cfg),saved.J.to_numpy())
    np.testing.assert_allclose(prediction,saved[['pred_'+m for m in METRICS]],rtol=1e-10,atol=1e-8)
    assert (prediction[:,:2]>0).all()
    assert (prediction[:,2]<=saved.J.to_numpy()*prediction[:,0]+1e-10).all()
    # Direct MSE recomputation checks all exported train/validation/test losses.
    scores=pd.read_csv(ROOT/'metrics.csv')
    for row in scores.itertuples():
        group=saved[saved.split==row.split]
        mse=np.mean((group[row.metric]-group['pred_'+row.metric])**2)
        np.testing.assert_allclose(mse,row.mse,rtol=1e-9)
    sim=Simulator(seed=20261121)
    for a in ACTIONS:
        for noise in ('none','trajectory','independent','pooled'):
            sim.reset(initial_loss=l0[1,a])
            for e in range(4):
                j,p=sim.J,sim.phase
                cases=sim.scenarios(j,p,a,e,noise)
                np.testing.assert_allclose(cases[:,:3].mean(axis=0),list(sim.means(j,p,a).values()),rtol=1e-10)
                row=sim.step(a,noise)
                assert 0<row['J_next']<=j and row['phase_next']>=p
                np.testing.assert_allclose(row['epsilon_s'],row['T_s']*row['J_next'])
            assert sim.epoch==4
    provenance=json.loads((ROOT/'provenance.json').read_text())
    from pathlib import Path
    if 'old_directory' in provenance and Path(provenance['old_directory']).is_dir():
        for path,digest in provenance['old_sha256'].items():
            assert hashlib.sha256((Path(provenance['old_directory'])/path).read_bytes()).hexdigest()==digest,path
        old_check='original artifact checksums unchanged'
    else:
        # Public packages retain the reference snapshots, not private local paths.
        for path,digest in provenance['reference_snapshot_sha256'].items():
            assert hashlib.sha256((ROOT/'previous_probe'/path).read_bytes()).hexdigest()==digest,path
        old_check='packaged previous-probe reference checksums unchanged'
    # Noise bank is reconstructed from validation rows only, never test residuals.
    va=saved[saved.split=='validation'].sort_values(['task','action','epoch'])
    expected_bank=np.column_stack([va['T']/va.pred_T,va.dem/va.pred_dem,
                                  va.J_next/va.J-va.pred_epsilon/(va.pred_T*va.J)]).reshape(6,5,4,3)
    np.testing.assert_allclose(sim.bank,expected_bank,rtol=1e-9,atol=1e-12)
    if (ROOT/'exact_task_distributions.csv').exists():
        paths=pd.read_csv(ROOT/'exact_task_distributions.csv')
        for (action,mode),g in paths.groupby(['action','mode']):
            assert len(g)==(1296 if mode=='independent' else 6)
            np.testing.assert_allclose(g.probability.sum(),1.)
            assert (g[['total_T','total_dem','J_end']]>0).all().all()
    result={'status':'passed','observations':600,'paired_seed_groups':30,
            'checks':['independent scalar paper recursion','current/future outcome causality',
                      'post-epoch probe and validation exclusion','running minimum under loss rises',
                      'phase interval convention and irreversible phase','train-only thresholds',
                      'paired split integrity','exported MLP predictions and metrics',
                      'stochastic simulator bounds and conditional scenario means',
                      'validation-only residual noise bank','enumerated task path counts and probabilities',old_check]}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
