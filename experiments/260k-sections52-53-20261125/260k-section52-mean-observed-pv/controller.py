"""Standalone LYA controller: model JSONs only, no simulator/noise/PV trajectory."""
import json
from pathlib import Path
import numpy as np

ACTIONS=np.array([32,64,128,256,512])


class MeanOnlyLYA:
    def __init__(self,model_directory,V=.1,weights=(100.,10.,1.),sigma_e=.24):
        directory=Path(model_directory)
        self.models=json.loads((directory/'models.json').read_text())
        self.config=json.loads((directory/'model_config.json').read_text())
        self.V=float(V);self.weights=np.array(weights,dtype=float);self.sigma_e=float(sigma_e)
        if not np.isfinite(self.V) or self.V<=0 or self.weights.shape!=(3,) or not np.all(self.weights>0):
            raise ValueError('Require V>0 and three positive phase weights')

    def means(self,J,phase):
        if not 0<J<=1 or phase not in (1,2,3):raise ValueError('Invalid simulator state')
        x=np.column_stack([np.full(5,(J-self.config['j_mean'])/self.config['j_sd'])]+
            [np.full(5,phase==k) for k in (1,2,3)]+[ACTIONS==a for a in ACTIONS])
        values=[]
        for target in ['T','dem','epsilon']:
            m=self.models[target]
            z=np.tanh(x@np.array(m['W'])+m['b'])@np.array(m['v'])+m['bias']
            values.append(values[0]*J/(1+np.exp(-np.clip(z,-50,50))) if target=='epsilon' else m['scale']*np.logaddexp(0,z))
        return np.column_stack(values)

    def select(self,J,phase,Q,observed_power_W):
        if not np.isfinite(Q) or Q<0 or not np.isfinite(observed_power_W) or observed_power_W<0:
            raise ValueError('Q and the already-observed power must be finite and nonnegative')
        means=self.means(J,phase)
        pv_estimate=float(observed_power_W)*means[:,0]
        shortfall=np.maximum(means[:,1]-pv_estimate,0)
        scores=(self.V*shortfall+self.weights[phase-1]*Q*(means[:,2]-self.sigma_e*means[:,0]))/means[:,0]
        return dict(action=int(ACTIONS[np.argmin(scores)]),means=means,
            estimated_PV_J=pv_estimate,estimated_shortfall_J=shortfall,scores=scores)
