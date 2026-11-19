"""Test-only residual diagnostics. Validation alone supplies simulation innovations."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from simulator import ROOT,ACTIONS,METRICS,Simulator
SEED=20261119

def summary(r):
    r=np.asarray(r);q=np.quantile(r,[0,.01,.05,.5,.95,.99,1]);w,p=stats.shapiro(r)
    return dict(n=len(r),mean=r.mean(),sd=r.std(ddof=1),min=q[0],q01=q[1],q05=q[2],median=q[3],q95=q[4],q99=q[5],max=q[6],
                skew=stats.skew(r,bias=False),excess_kurtosis=stats.kurtosis(r,bias=False),shapiro_w=w,shapiro_p=p)

def holm(values):
    values=np.asarray(values);order=np.argsort(values);out=np.empty(len(values));out[order]=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return out

def main():
    d=pd.read_csv(ROOT/'epochs.csv');te=d[d.split=='test'];va=d[d.split=='validation'];rng=np.random.default_rng(SEED)
    rows=[];tests=[];commons=[];common_rows=[];standardized=[];lag=[]
    perms=np.array([[rng.permutation(5) for _ in range(6)] for _ in range(4999)])
    for k in METRICS:
        col='res_'+k
        mat=te.pivot(index=['task','epoch'],columns='action',values=col).loc[:,list(ACTIONS)].to_numpy().reshape(6,4,5)
        boot=np.array([mat[rng.integers(0,6,6)].ravel().std(ddof=1) for _ in range(2000)])
        rows.append(dict(metric=k,group='all',**summary(te[col]),sd_ci_low=np.quantile(boot,.025),sd_ci_high=np.quantile(boot,.975)))
        for a,g in te.groupby('action'):rows.append(dict(metric=k,group=str(a),**summary(g[col])))
        moments=va.groupby('action')[col].agg(['mean','std']).reindex(ACTIONS)
        scaled=(mat-moments['mean'].to_numpy())/moments['std'].to_numpy()
        for mode,arr in [('raw',mat),('initial_state_only',mat[:,:1]),('validation_standardized',scaled)]:
            cuts=np.quantile(arr,[.05,.1,.25,.5,.75,.9,.95]);cdf=(arr[:,:,:,None]<=cuts).mean(axis=1)
            def stat(c):
                v=c.mean(axis=0);return np.mean((v-v.mean(axis=0))**2)
            obs=stat(cdf);count=sum(stat(cdf[np.arange(6)[:,None],p])>=obs-1e-15 for p in perms)
            tests.append(dict(metric=k,scope=mode,statistic=obs,p=(1+count)/5000,paired_tasks=6,permutations=4999))
        standardized.append(dict(metric=k,**summary(scaled.ravel())))
        # Descriptive common proxy. Per-action centering on test is diagnostic only, never calibration.
        centered=mat.reshape(24,5)-mat.reshape(24,5).mean(axis=0)
        proxy=centered.mean(axis=1);loo=(centered.sum(axis=1,keepdims=True)-centered)/4
        cov=np.cov(centered,rowvar=False);off=cov[np.triu_indices(5,1)].mean()
        commons.append(dict(metric=k,**summary(proxy),mean_offdiagonal_covariance=off,loo_corr=np.corrcoef(loo.ravel(),centered.ravel())[0,1],
                            loo_skill_vs_zero=1-np.mean((loo-centered)**2)/np.mean(centered**2)))
        for (t,e),v in zip(te[['task','epoch']].drop_duplicates().itertuples(index=False,name=None),proxy):common_rows.append(dict(metric=k,task=t,epoch=e,proxy=v))
        for ai,a in enumerate(ACTIONS):
            v=mat[:,:,ai];lag.append(dict(metric=k,action=a,lag1_corr=np.corrcoef(v[:,:-1].ravel(),v[:,1:].ravel())[0,1],pairs=18))
        # Exactly all test residuals, including tails, are used in every diagnostic.
        for ai,a in enumerate(ACTIONS):
            mask=(d.split=='test')&(d.action==a)
            d.loc[mask,'z_'+k]=(d.loc[mask,col]-moments.loc[a,'mean'])/moments.loc[a,'std']
    n=pd.DataFrame(rows);mask=n.group.eq('all');n.loc[mask,'shapiro_p_holm_three']=holm(n.loc[mask,'shapiro_p'])
    n.to_csv(ROOT/'noise_summary.csv',index=False)
    t=pd.DataFrame(tests)
    for scope,g in t.groupby('scope'):t.loc[g.index,'p_holm_three']=holm(g.p)
    t.to_csv(ROOT/'action_tests.csv',index=False)
    pd.DataFrame(commons).to_csv(ROOT/'common_component.csv',index=False)
    pd.DataFrame(common_rows).to_csv(ROOT/'common_proxy_points.csv',index=False)
    pd.DataFrame(standardized).to_csv(ROOT/'standardized_noise.csv',index=False)
    pd.DataFrame(lag).to_csv(ROOT/'serial_correlation.csv',index=False)
    te[['res_'+k for k in METRICS]].corr().to_csv(ROOT/'residual_correlations.csv')
    d.to_csv(ROOT/'epochs.csv',index=False)
    # Physical constraints, exact one-step scenario means, and held-out interval coverage.
    sim=Simulator(seed=SEED);coverage=[];rollouts=[]
    for mode in ('independent','pooled'):
        covered=np.zeros(3);width=np.zeros(3)
        for r in te.itertuples():
            cases=sim.scenarios(r.J,r.phi,r.action,r.epoch-1,mode)
            means=np.array(list(sim.means(r.J,r.phi,r.action).values()))
            np.testing.assert_allclose(cases[:,:3].mean(axis=0),means,rtol=1e-9,atol=1e-8)
            assert np.all(cases[:,:2]>0) and np.all((cases[:,3]>0)&(cases[:,3]<=r.J))
            lo,hi=np.quantile(cases[:,:3],[.05,.95],axis=0);obs=[r.T,r.dem,r.epsilon]
            covered+=(obs>=lo)&(obs<=hi);width+=hi-lo
        for i,k in enumerate(METRICS):coverage.append(dict(mode=mode,metric=k,coverage90=covered[i]/len(te),width=width[i]/len(te),n=len(te)))
    pd.DataFrame(coverage).to_csv(ROOT/'interval_coverage.csv',index=False)
    for (task,a),g in te.groupby(['task','action']):
        sim.reset();path=[sim.step(a,noise='none') for _ in range(4)]
        for r,p in zip(g.itertuples(),path):rollouts.append(dict(task=task,action=a,epoch=r.epoch,observed_J_next=r.J_next,simulated_J_next=p['J_next'],observed_T=r.T,simulated_T=p['T_s'],observed_dem=r.dem,simulated_dem=p['demand_J']))
    pd.DataFrame(rollouts).to_csv(ROOT/'heldout_rollouts.csv',index=False)
    # Full task stochastic examples, preserving the measured three-outcome coupling.
    examples=[]
    for mode in ('none','independent','trajectory','pooled'):
        for a in ACTIONS:
            for run in range(100):
                sim.reset();path=[sim.step(a,noise=mode) for _ in range(4)]
                examples.append(dict(mode=mode,action=a,replicate=run,total_T=sum(v['T_s'] for v in path),total_dem=sum(v['demand_J'] for v in path),J_end=path[-1]['J_next']))
    pd.DataFrame(examples).to_csv(ROOT/'simulation_samples.csv',index=False)
    print(n[n.group=='all'].to_string(index=False));print(t.to_string(index=False));print(pd.DataFrame(commons).to_string(index=False));print('SIMULATOR CHECKS PASSED')
if __name__=='__main__':main()
