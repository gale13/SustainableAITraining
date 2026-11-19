"""Split by paired initialization seed, calibrate phases on train only, fit 3 small MLPs."""
import json,hashlib
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from simulator import ROOT,ACTIONS,METRICS,EPS,features,phase_after,predict
SEED=20261119

def save(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def prepare():
    with ZipFile(ROOT/'source.zip') as z:
        d=pd.read_csv(z.open('epochs.csv')).rename(columns={'batch':'action'})
        b=json.load(z.open('baselines.json'));meta=json.load(z.open('snapshot.json'))
    assert len(d)==600 and not d.duplicated(['task','action','epoch']).any()
    assert d.groupby(['task','action']).size().eq(4).all()
    assert d.groupby('task').initial_weights_sha256.nunique().eq(1).all()
    assert d.groupby(['task','epoch']).order_sha256.nunique().eq(1).all()
    assert np.isfinite(d.select_dtypes('number')).all().all()
    l0={(v['task'],v['batch']):v['train_probe_loss'] for v in b}
    d=d.sort_values(['task','action','epoch']).reset_index(drop=True)
    d['L0']=[l0[t,a] for t,a in zip(d.task,d.action)]
    d['J_next']=np.minimum(d.groupby(['task','action']).train_probe_loss.cummin(),d.L0)/(d.L0+EPS)
    d['J']=d.groupby(['task','action']).J_next.shift().fillna(d.L0/(d.L0+EPS))
    d['T']=d.train_seconds;d['dem']=d.train_energy_j;d['epsilon']=d['T']*d.J_next
    d['delta']=(d.J_next-d.J)/(d.J+EPS)
    # Six temporal strata of five paired seeds; 3 train, 1 validation, 1 test per stratum.
    rng=np.random.default_rng(SEED);splits={}
    for chunk in np.array_split(np.arange(1,31),6):
        order=rng.permutation(chunk)
        for t,s in zip(order,['train']*3+['validation','test']):splits[int(t)]=s
    d['split']=d.task.map(splits);tr=d.split.eq('train')
    # Only past changes that can enter one of the four decision states.
    betas=np.quantile(d.loc[tr & (d.epoch<4),'delta'],[1/3,2/3])
    cfg=dict(analysis_seed=SEED,K=3,W_ma=1,beta_internal=betas.tolist(),j_mean=float(d.loc[tr,'J'].mean()),
             j_sd=float(d.loc[tr,'J'].std(ddof=0)),actions=list(ACTIONS),task_epochs=4,
             state_loss='fixed 12800-sequence eval-mode train_probe_loss; running minimum / initial probe loss',
             split_tasks={s:sorted(t for t,v in splits.items() if v==s) for s in ['train','validation','test']},
             snapshot=meta,source_sha256=hashlib.sha256((ROOT/'source.zip').read_bytes()).hexdigest())
    d['phi']=1
    for _,g in d.groupby(['task','action']):
        p=1
        for r in g.itertuples():
            d.loc[r.Index,'phi']=p;p=phase_after(p,r.J,r.J_next,cfg)
    save('model_config.json',cfg)
    return d,cfg

def fit_one(x,y,bound,width,penalty,seed):
    rng=np.random.default_rng(seed);n,p=x.shape;bounded=bound is not None
    scale=float(y.mean());sd=float(y.std());q0=np.r_[rng.normal(0,.3,p*width),np.zeros(width),rng.normal(0,.1,width),0.]
    def unpack(q):return q[:p*width].reshape(p,width),q[p*width:(p+1)*width],q[-width-1:-1],q[-1]
    def objective(q):
        W,b,v,bias=unpack(q);h=np.tanh(x@W+b);z=h@v+bias
        pred=bound*expit(z) if bounded else scale*np.logaddexp(0,z)
        deriv=pred*expit(-z) if bounded else scale*expit(z)
        err=(pred-y)/sd;delta=2/n*err*deriv/sd;dh=delta[:,None]*v*(1-h*h)
        fac=penalty/(W.size+v.size)
        loss=np.mean(err**2)+fac*((W*W).sum()+v@v)
        grad=np.r_[(x.T@dh+2*fac*W).ravel(),dh.sum(0),h.T@delta+2*fac*v,delta.sum()]
        return loss,grad
    res=minimize(objective,q0,jac=True,method='L-BFGS-B',options={'maxiter':1800,'ftol':1e-11,'gtol':1e-7})
    W,b,v,bias=unpack(res.x)
    return dict(W=W.tolist(),b=b.tolist(),v=v.tolist(),bias=float(bias),scale=scale,width=width,
                penalty=penalty,initialization_seed=seed,optimization_loss=float(res.fun),target_sd=sd,
                iterations=int(res.nit),converged=bool(res.success),message=str(res.message),parameters=len(q0))

def run():
    d,cfg=prepare();x=features(d.J,d.phi,d.action,cfg);j=d.J.to_numpy()
    tr=d.split.eq('train').to_numpy();va=d.split.eq('validation').to_numpy()
    models={};search=[];pred=np.zeros((len(d),3))
    for i,k in enumerate(METRICS):
        y=d[k].to_numpy();best=np.inf
        bound=pred[:,0]*j if k=='epsilon' else None
        for width in (4,8,16):
            for penalty in (1e-4,1e-2):
                candidates=[fit_one(x[tr],y[tr],bound[tr] if bound is not None else None,width,penalty,SEED+s) for s in (0,1)]
                m=min(candidates,key=lambda m:m['optimization_loss'])
                z=np.tanh(x@np.array(m['W'])+m['b'])@np.array(m['v'])+m['bias']
                v=bound*expit(z) if bound is not None else m['scale']*np.logaddexp(0,z)
                score=float(np.mean((v[va]-y[va])**2));search.append(dict(metric=k,width=width,penalty=penalty,validation_mse=score,**{a:m[a] for a in ['iterations','converged','optimization_loss']}))
                if score<best:best=score;models[k]=m;pred[:,i]=v
        print('SELECTED',k,models[k]['width'],models[k]['penalty'],best,flush=True)
    save('models.json',models);pd.DataFrame(search).to_csv(ROOT/'model_selection.csv',index=False)
    np.testing.assert_allclose(pred,predict(models,x,j))
    rows=[];by_action=[]
    for i,k in enumerate(METRICS):
        d['pred_'+k]=pred[:,i];d['res_'+k]=d[k]-pred[:,i]
        baseline=d.loc[tr].groupby('action')[k].mean()
        for name in ('train','validation','test'):
            mask=d.split.eq(name);yy=d.loc[mask,k].to_numpy();err=d.loc[mask,'res_'+k].to_numpy()
            rows.append(dict(metric=k,split=name,n=len(yy),mse=np.mean(err**2),rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),r2=1-err@err/np.sum((yy-yy.mean())**2),action_only_rmse=np.sqrt(np.mean((yy-d.loc[mask,'action'].map(baseline))**2))))
        for a,g in d[d.split=='test'].groupby('action'):
            err=g['res_'+k].to_numpy();by_action.append(dict(metric=k,action=a,rmse=np.sqrt(np.mean(err**2)),mean_error=np.mean(err),r2=1-np.sum(err**2)/np.sum((g[k]-g[k].mean())**2)))
    pd.DataFrame(rows).to_csv(ROOT/'metrics.csv',index=False);pd.DataFrame(by_action).to_csv(ROOT/'metrics_by_action.csv',index=False)
    d.to_csv(ROOT/'epochs.csv',index=False)
    # Retain paired metrics and all four epoch positions from each validation trajectory.
    va=d[d.split=='validation'].sort_values(['task','action','epoch'])
    bank=np.column_stack([va['T']/va.pred_T,va.dem/va.pred_dem,va.J_next/va.J-va.pred_epsilon/(va.pred_T*va.J)])
    np.savez_compressed(ROOT/'noise_bank.npz',bank=bank.reshape(6,5,4,3),tasks=cfg['split_tasks']['validation'])
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
