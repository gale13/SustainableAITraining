"""State recursion and analytic MLP objective frozen from the feasibility model."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from simulator import EPS,phase_after

def rebuild_states(d, initial_losses, beta_internal):
    """Do not use epoch e's result in its own decision-state features.

    r=e-1 within a task. At r=0 the only saved pretraining measurement is the
    fixed training probe. At r>=1 the latest observed loss is train_loss[e-1].
    All post-epoch values, including the terminal one, belong to the same task.
    """
    d=d.sort_values(['task','action','epoch']).reset_index(drop=True).copy()
    d['r']=d.epoch-1
    d['L0']=[initial_losses[t,a] for t,a in zip(d.task,d.action)]
    d['observed_loss_before']=d.groupby(['task','action']).train_loss.shift().fillna(d.L0)
    d['observed_loss_after']=d.train_loss
    d['J_next']=np.minimum(d.groupby(['task','action']).train_loss.cummin(),d.L0)/(d.L0+EPS)
    d['J']=d.groupby(['task','action']).J_next.shift().fillna(d.L0/(d.L0+EPS))
    d['T']=d.train_seconds
    d['dem']=d.train_energy_j
    d['epsilon']=d['T']*d.J_next
    d['delta']=(d.J_next-d.J)/(d.J+EPS)
    d['phi']=1
    d['phi_next_within_task']=1
    cfg={'K':3,'W_ma':1,'beta_internal':list(beta_internal)}
    for _,g in d.groupby(['task','action']):
        p=1
        for row in g.itertuples():
            d.loc[row.Index,'phi']=p
            p=phase_after(p,row.J,row.J_next,cfg)
            d.loc[row.Index,'phi_next_within_task']=p
    return d

def fit_one(x,y,bound,width,penalty,seed,maxiter=6000):
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
    res=minimize(objective,q0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-7})
    W,b,v,bias=unpack(res.x)
    return dict(W=W.tolist(),b=b.tolist(),v=v.tolist(),bias=float(bias),scale=scale,width=width,
                penalty=penalty,initialization_seed=seed,optimization_loss=float(res.fun),target_sd=sd,
                iterations=int(res.nit),converged=bool(res.success),message=str(res.message),parameters=len(q0))

