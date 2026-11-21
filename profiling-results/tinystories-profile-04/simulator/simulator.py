"""Four-epoch, online-loss variant; inference requires NumPy only.

The conditional means use the paper's J/phase recursions. The empirical noise
bank remains a validation-calibrated diagnostic model, not a proof of state
sufficiency or of action-independent noise. Action switching was not measured.
Trajectory donors correlate with history: uniform scenario means are not a
proof of history-conditional state sufficiency for trajectory sampling.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
ACTIONS=(32,64,128,256,512)
METRICS=('T','dem','epsilon')
EPS=1e-8

def features(j,phase,action,config):
    j,p,a=np.broadcast_arrays(j,phase,action);j,p,a=j.ravel(),p.ravel(),a.ravel()
    if not (np.all((j>0)&(j<=1)) and np.isin(p,[1,2,3]).all() and np.isin(a,ACTIONS).all()):
        raise ValueError('Require 0<J<=1, phase in 1..3, and an observed action.')
    return np.column_stack([(j-config['j_mean'])/config['j_sd']]+[p==k for k in [1,2,3]]+[a==k for k in ACTIONS])

def phase_after(previous,j,j_next,config):
    if config.get('W_ma',1)!=1:
        raise ValueError('This four-epoch calibration uses W_ma=1; a longer window needs explicit history.')
    delta=(j_next-j)/(j+EPS)
    return max(previous,1+int(np.searchsorted(config['beta_internal'],delta,side='left')))

def predict(models,x,j):
    out=[]
    for k in METRICS:
        m=models[k];z=np.tanh(x@np.array(m['W'])+m['b'])@np.array(m['v'])+m['bias']
        out.append(out[0]*j/(1+np.exp(-np.clip(z,-50,50))) if k=='epsilon' else m['scale']*np.logaddexp(0,z))
    return np.column_stack(out)

class Simulator:
    def __init__(self,seed=42):
        self.config=json.loads((ROOT/'model_config.json').read_text())
        self.models=json.loads((ROOT/'models.json').read_text())
        self.bank=np.load(ROOT/'noise_bank.npz')['bank'] # donor task, action, epoch, correlated (T ratio,D ratio,retention error)
        self.rng=np.random.default_rng(seed);self.reset()
    def reset(self,initial_loss=None):
        # Supplying a task's measured initial loss exactly reproduces its J(0).
        # The default uses only the training-split mean (no held-out leakage).
        l0=float(self.config['initial_loss_mean_train'] if initial_loss is None else initial_loss)
        if not np.isfinite(l0) or l0<=0:
            raise ValueError('initial_loss must be finite and positive')
        self.J=l0/(l0+EPS);self.phase=1;self.epoch=0;self.donor=int(self.rng.integers(len(self.bank)))
    def means(self,J,phase,action):
        return dict(zip(METRICS,predict(self.models,features(J,phase,action,self.config),np.atleast_1d(J))[0]))
    def scenarios(self,J,phase,action,epoch,noise='trajectory'):
        if epoch not in range(4):raise ValueError('epoch must be zero-based 0..3')
        t,d,e=self.means(J,phase,action).values();ratio=e/(t*J)
        if noise=='none':return np.array([[t,d,e,e/t]])
        if noise in ('trajectory','independent'):
            b=self.bank[:,ACTIONS.index(action),epoch,:]
        elif noise=='pooled':
            b=self.bank[:,:,epoch,:].reshape(-1,3)
        else:raise ValueError('noise: trajectory, independent, pooled, none')
        times=b[:,0]/b[:,0].mean();energies=b[:,1]/b[:,1].mean();raw=ratio+b[:,2]
        lo,hi=-raw.max(),1-raw.min()
        for _ in range(60):
            shift=(lo+hi)/2
            if np.mean(times*np.clip(raw+shift,1e-12,1.))<ratio:lo=shift
            else:hi=shift
        jp=J*np.clip(raw+(lo+hi)/2,1e-12,1.)
        return np.column_stack([t*times,d*energies,t*times*jp,jp])
    def step(self,action,noise='trajectory'):
        if self.epoch>=4:raise ValueError('Task complete. Call reset() to begin the next task.')
        cases=self.scenarios(self.J,self.phase,action,self.epoch,noise)
        idx=self.donor if noise=='trajectory' else int(self.rng.integers(len(cases)))
        t,d,e,jp=cases[idx];j,p=self.J,self.phase
        self.phase=phase_after(p,j,jp,self.config);self.J=jp;self.epoch+=1
        return dict(epoch=self.epoch,action=action,J=j,phase=p,T_s=t,demand_J=d,demand_kWh=d/3.6e6,
                    epsilon_s=e,J_next=jp,phase_next=self.phase)
