"""Refit the validated recipe on all observations, then calibrate empirical noise.

There is deliberately no train/validation/test partition for this final fit.
Historical split labels are retained solely for provenance. The independent
feasibility experiment remains unchanged in its own directory.
"""
import hashlib,json
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from simulator import ROOT,ACTIONS,METRICS,EPS,features,predict
from fitting_core import rebuild_states,fit_one

def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def prepare():
    recipe=json.loads((ROOT/'calibration_recipe.json').read_text())
    with ZipFile(ROOT/'source.zip') as z:
        raw=pd.read_csv(z.open('epochs.csv')).rename(columns={'batch':'action'})
        baselines=json.load(z.open('baselines.json'))
    assert len(raw)==600 and not raw.duplicated(['task','action','epoch']).any()
    assert raw.groupby(['task','action']).size().eq(4).all()
    assert raw.groupby('task').initial_weights_sha256.nunique().eq(1).all()
    assert raw.groupby(['task','epoch']).order_sha256.nunique().eq(1).all()
    initial={(r['task'],r['batch']):r['train_probe_loss'] for r in baselines}
    data=rebuild_states(raw,initial,recipe['beta_internal'])
    oldsplit={t:s for s,tasks in recipe['original_split_tasks'].items() for t in tasks}
    data['original_split']=data.task.map(oldsplit)
    assert data.original_split.notna().all()
    data['calibration_role']='full_data'
    tasks=sorted(data.task.unique().tolist())
    assert tasks==list(range(1,31)) and sorted(data.action.unique())==list(ACTIONS)
    cfg=dict(version='profiling04_full_data',data_role='full_data_calibration_no_holdout',
        K=recipe['K'],W_ma=recipe['W_ma'],beta_internal=recipe['beta_internal'],
        beta_calibration='Frozen from the independently evaluated feasibility recipe; not reselected on full data.',
        j_mean=float(data.J.mean()),j_sd=float(data.J.std(ddof=0)),actions=list(ACTIONS),task_epochs=4,
        state_loss='Completed online epoch train_loss; normalized within-task running minimum, measured pretraining probe anchor.',
        initial_loss_source='Pretraining train_probe_loss on 12800 fixed training sequences; no post-epoch probe enters state.',
        initial_loss_mean_calibration=float(data.loc[data.epoch.eq(1),'L0'].mean()),
        epsilon=EPS,analysis_seed=recipe['analysis_seed'],fit_tasks=tasks,noise_tasks=tasks,
        observations=600,noise_donors_per_action_epoch=30,noise_source='full_data_calibration',
        default_noise='empirical',noise_residuals='In-sample relative residual triples after full-data refit.',
        original_split_tasks=recipe['original_split_tasks'],held_out_observations=0,
        measurement_scope='GPU training windows only; excludes validation/probe/checkpoint/preprocessing.',
        source_sha256=hashlib.sha256((ROOT/'source.zip').read_bytes()).hexdigest(),
        feasibility_commit=recipe['feasibility_commit'])
    save('model_config.json',cfg)
    return data,cfg,recipe

def scores(y,pred):
    error=y-pred;mse=float(np.mean(error**2))
    return dict(mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(error))),
        mean_residual=float(np.mean(error)),r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)))

def run():
    data,cfg,recipe=prepare();x=features(data.J,data.phi,data.action,cfg);j=data.J.to_numpy()
    models={};pred=np.zeros((len(data),3));attempts=[];metrics=[];byaction=[]
    for index,key in enumerate(METRICS):
        y=data[key].to_numpy();bound=pred[:,0]*j if key=='epsilon' else None
        spec=recipe['fixed_models'][key]
        candidates=[]
        for seed in recipe['initialization_seeds']:
            model=fit_one(x,y,bound,spec['width'],spec['penalty'],seed)
            candidates.append(model)
            attempts.append(dict(metric=key,fit_rows=len(data),**{k:model[k] for k in
                ['width','penalty','initialization_seed','optimization_loss','iterations','converged','message']}))
            print('FIT',key,'seed',seed,'objective',model['optimization_loss'],
                  'iterations',model['iterations'],'converged',model['converged'],flush=True)
        model=min(candidates,key=lambda m:m['optimization_loss'])
        if not model['converged']:raise RuntimeError(f'Best {key} fit did not converge; inspect before publishing artifacts.')
        models[key]=model
        z=np.tanh(x@np.array(model['W'])+model['b'])@np.array(model['v'])+model['bias']
        pred[:,index]=bound*expit(z) if bound is not None else model['scale']*np.logaddexp(0,z)
        data['pred_'+key]=pred[:,index];data['res_'+key]=y-pred[:,index]
        metrics.append(dict(metric=key,dataset='all_calibration',n=len(data),**scores(y,pred[:,index]),
            standardized_mse=float(np.mean((y-pred[:,index])**2)/model['target_sd']**2),
            regularized_optimization_loss=model['optimization_loss']))
        for action,g in data.groupby('action'):
            byaction.append(dict(metric=key,action=action,dataset='all_calibration',n=len(g),
                                 **scores(g[key].to_numpy(),g['pred_'+key].to_numpy())))
    np.testing.assert_allclose(pred,predict(models,x,j),rtol=1e-12,atol=1e-10)
    save('models.json',models)
    pd.DataFrame(attempts).to_csv(ROOT/'optimization_runs.csv',index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(byaction).to_csv(ROOT/'metrics_by_action.csv',index=False)
    data.to_csv(ROOT/'epochs.csv',index=False)
    # Every original observation contributes exactly one joint noise donor entry.
    ordered=data.sort_values(['task','action','epoch'])
    bank=np.column_stack([ordered['T']/ordered.pred_T,ordered.dem/ordered.pred_dem,
        ordered.J_next/ordered.J-ordered.pred_epsilon/(ordered.pred_T*ordered.J)])
    np.savez_compressed(ROOT/'noise_bank.npz',bank=bank.reshape(30,5,4,3),tasks=cfg['noise_tasks'])
    library=ordered[['task','action','epoch','original_split','J','phi','J_next','T','dem','epsilon',
                     'pred_T','pred_dem','pred_epsilon','res_T','res_dem','res_epsilon']].copy()
    library[['time_ratio','demand_ratio','retention_residual']]=bank
    library.to_csv(ROOT/'noise_library.csv',index=False)
    save('noise_metadata.json',dict(source='all 600 full-data calibration observations',
        residual_type='In-sample relative residual triples from the final all-data MLPs.',
        shape=[30,5,4,3],axes=['task','action','epoch','joint residual component'],
        components=['T / predicted_T','demand / predicted_demand','J_next / J - predicted_epsilon / (predicted_T * J)'],
        default='empirical: new uniform task draw per epoch, joint across metrics, conditional on action/epoch',
        gaussian_assumption=False,fit_rows=600,noise_rows=600,held_out_rows=0,
        original_split_counts={str(k):int(v) for k,v in data.original_split.value_counts().items()},
        caution='Calibration fit and noise may be optimistic because the same observations fit the mean and residual bank. Independent feasibility evidence is preserved separately.'))
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
