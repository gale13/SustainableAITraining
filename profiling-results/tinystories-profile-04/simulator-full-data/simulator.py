"""Full-data Profiling 04 simulator; inference requires NumPy only.

All 600 observations calibrate the MLPs and empirical residual bank. There is
no held-out set in this version. Independent validation evidence lives in the
preserved feasibility simulator. Fresh joint residual draws are the default;
optional trajectory sampling is a correlated-noise sensitivity mode.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
ACTIONS=(32,64,128,256,512)
METRICS=('T','dem','epsilon')
EPS=1e-8
DEFAULT_NOISE='empirical'

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
        with np.load(ROOT/'noise_bank.npz') as saved:
            self.bank=saved['bank'] # donor task, action, epoch, correlated (T ratio,D ratio,retention error)
            self.donor_tasks=saved['tasks']
        if self.donor_tasks.tolist()!=self.config['noise_tasks']:
            raise ValueError('Noise donors must be all calibration tasks, in the configured order.')
        self.rng=np.random.default_rng(seed);self.reset()
    def reset(self,initial_loss=None):
        # Supplying a task's measured initial loss exactly reproduces its J(0).
        # The default uses the mean of all 150 measured initial losses.
        l0=float(self.config['initial_loss_mean_calibration'] if initial_loss is None else initial_loss)
        if not np.isfinite(l0) or l0<=0:
            raise ValueError('initial_loss must be finite and positive')
        self.J=l0/(l0+EPS);self.phase=1;self.epoch=0;self.donor=int(self.rng.integers(len(self.bank)))
    def means(self,J,phase,action):
        return dict(zip(METRICS,predict(self.models,features(J,phase,action,self.config),np.atleast_1d(J))[0]))
    def scenarios(self,J,phase,action,epoch,noise=DEFAULT_NOISE):
        if epoch not in range(4):raise ValueError('epoch must be zero-based 0..3')
        t,d,e=self.means(J,phase,action).values();ratio=e/(t*J)
        if noise=='none':return np.array([[t,d,e,e/t]])
        if noise in ('empirical','trajectory','independent'):
            b=self.bank[:,ACTIONS.index(action),epoch,:]
        elif noise=='pooled':
            b=self.bank[:,:,epoch,:].reshape(-1,3)
        else:raise ValueError('noise: empirical (default; alias independent), trajectory, pooled, none')
        times=b[:,0]/b[:,0].mean();energies=b[:,1]/b[:,1].mean();raw=ratio+b[:,2]
        lo,hi=-raw.max(),1-raw.min()
        for _ in range(60):
            shift=(lo+hi)/2
            if np.mean(times*np.clip(raw+shift,1e-12,1.))<ratio:lo=shift
            else:hi=shift
        jp=J*np.clip(raw+(lo+hi)/2,1e-12,1.)
        return np.column_stack([t*times,d*energies,t*times*jp,jp])
    def step(self,action,noise=DEFAULT_NOISE):
        if self.epoch>=4:raise ValueError('Task complete. Call reset() to begin the next task.')
        cases=self.scenarios(self.J,self.phase,action,self.epoch,noise)
        means=self.means(self.J,self.phase,action)
        idx=self.donor if noise=='trajectory' else int(self.rng.integers(len(cases)))
        donor_task=None if noise=='none' else int(self.donor_tasks[idx//len(ACTIONS) if noise=='pooled' else idx])
        donor_action=None if noise=='none' else (ACTIONS[idx%len(ACTIONS)] if noise=='pooled' else action)
        t,d,e,jp=cases[idx];j,p=self.J,self.phase
        self.phase=phase_after(p,j,jp,self.config);self.J=jp;self.epoch+=1
        return dict(epoch=self.epoch,action=action,J=j,phase=p,T_s=t,demand_J=d,demand_kWh=d/3.6e6,
                    epsilon_s=e,J_next=jp,phase_next=self.phase,
                    mean_T_s=means['T'],mean_demand_J=means['dem'],mean_epsilon_s=means['epsilon'],
                    noise_T_s=t-means['T'],noise_demand_J=d-means['dem'],noise_epsilon_s=e-means['epsilon'],
                    noise_mode='empirical' if noise=='independent' else noise,
                    noise_source='none' if noise=='none' else 'full_data_calibration',
                    noise_donor_task=donor_task,noise_donor_action=donor_action,
                    noise_donor_epoch=None if noise=='none' else self.epoch)
