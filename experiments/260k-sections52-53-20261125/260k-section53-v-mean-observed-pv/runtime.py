"""260K simulator: frozen model, empirical joint noise, and PVDatabase adapter."""
import ctypes,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SIM=ROOT.parent/'tinystories-04-simulator-full-data'
PV_PACKAGE=ROOT.parent/'opencem-pv-final'
sys.path.insert(0,str(PV_PACKAGE/'src'))
from opencem_pv import PVDatabase,DataGapError
from pv_observations import observation_stream
POLICIES=['LYA','B32','B128','B512','STA','INC','DEC']
ACTIONS=[32,64,128,256,512]
BASELINES={'B32':[32]*4,'B128':[128]*4,'B512':[512]*4,'STA':[32,32,512,512],
           'INC':[32,64,256,512],'DEC':[512,256,64,32]}
TRACE_COLUMNS=['global_epoch','task','epoch','clock_start_s','phase','phase_next','action','donor_task',
 'J','J_next','T_s','demand_J','renewable_J','shortfall_J','epsilon_s','Q_before_s','Q_after_s',
 'phase_weight','mean_T_s','mean_demand_J','mean_epsilon_s','estimated_shortfall_J','theta','is_LYA',
 'observed_power_W','observed_time_s','observation_age_s','estimated_PV_J']
CONFIG=json.loads((SIM/'model_config.json').read_text())
MODELS=json.loads((SIM/'models.json').read_text())
with np.load(SIM/'noise_bank.npz') as f:BANK=np.ascontiguousarray(f['bank'])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(bank_override=None):
    cpp=ROOT/'engine.cpp';binary=ROOT/('engine.dylib' if sys.platform=='darwin' else 'engine.so')
    controller=ROOT/'controller.cpp'
    if not binary.exists() or max(cpp.stat().st_mtime,controller.stat().st_mtime)>binary.stat().st_mtime:
        compiler=shutil.which('clang++') or shutil.which('g++')
        shared=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
        subprocess.run([compiler,'-std=c++17','-O3',*shared,str(cpp),str(controller),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.configure.argtypes=[arr]*7+[ctypes.c_int];lib.configure.restype=None
    lib.set_pv.argtypes=[arr,arr,ctypes.c_int];lib.set_pv.restype=None
    lib.set_observations.argtypes=[arr,arr,ctypes.c_int];lib.set_observations.restype=None
    lib.latest_observation.argtypes=[ctypes.c_double,ctypes.c_double,arr];lib.latest_observation.restype=None
    lib.predict_means.argtypes=[ctypes.c_double,ctypes.c_int,ctypes.c_int,arr];lib.predict_means.restype=None
    lib.policy_decision.argtypes=[ctypes.c_double,ctypes.c_int]+[ctypes.c_double]*6+[arr];lib.policy_decision.restype=None
    lib.pv_energy.argtypes=[ctypes.c_double]*3;lib.pv_energy.restype=ctypes.c_double
    lib.pv_invalid.argtypes=[];lib.pv_invalid.restype=ctypes.c_int
    lib.inspect.argtypes=[ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,arr];lib.inspect.restype=None
    lib.run.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,arr,ctypes.c_double,ctypes.c_double,ctypes.c_double,byte,arr,ctypes.c_void_p];lib.run.restype=None
    w=np.zeros((3,9,16));b=np.zeros((3,16));v=np.zeros((3,16));meta=np.zeros((3,3))
    for i,k in enumerate(['T','dem','epsilon']):
        m=MODELS[k];h=m['width'];w[i,:,:h]=m['W'];b[i,:h]=m['b'];v[i,:h]=m['v'];meta[i]=[h,m['bias'],m['scale']]
    l0=CONFIG['initial_loss_mean_calibration']
    spec=np.array([CONFIG['j_mean'],CONFIG['j_sd'],l0/(l0+1e-8),*CONFIG['beta_internal']])
    lib.configure(spec,w,b,v,meta,BANK if bank_override is None else np.ascontiguousarray(bank_override),np.array([0.]),1)
    return lib

LIB=load()

def set_day(day,max_gap_s=300):
    global OBSERVATIONS
    # Public API alone supplies both observations and midnight boundary values.
    # Complete coverage is required BEFORE constructing the fast representation.
    with PVDatabase() as db:pv=db.load_day(day,max_gap_s=max_gap_s)
    report=pv.energy_report(0,86400)
    if not report.complete:raise DataGapError(f'{day}: missing {report.missing_seconds} seconds')
    observed=pv.observations();valid=observed[observed.pv_power_w.notna()]
    times=valid.read_ts.to_numpy(dtype=float)-pv.start_timestamp
    inside=(times>0)&(times<86400)
    knots=np.ascontiguousarray(np.r_[0.,times[inside],86400.])
    power=np.ascontiguousarray(np.r_[pv.power_at(0),valid.pv_power_w.to_numpy(dtype=float)[inside],pv.power_at(86400)])
    assert np.all(np.diff(knots)>0) and np.isfinite(power).all()
    LIB.set_pv(knots,power,len(knots))
    OBSERVATIONS=observation_stream(pv)
    LIB.set_observations(*OBSERVATIONS,len(OBSERVATIONS[0]))
    assert abs(LIB.pv_energy(0,86400,1)-pv.energy(0,86400))<1e-5
    return pv

def decision(J,phase,Q,time_s,V,weight,alpha=1.,sigma=.24):
    out=np.zeros(34)
    LIB.policy_decision(J,phase,Q,time_s,V,weight,sigma,alpha,out)
    if LIB.pv_invalid():raise ValueError('Invalid observation request')
    return ACTIONS[int(out[0])],out[1:4],out[4:].reshape(5,6)

def means(J,phase,action):
    out=np.zeros(3);LIB.predict_means(J,phase,ACTIONS.index(action),out)
    return out

def inspect(J,phase,action,epoch,time_s,alpha=1.):
    out=np.zeros(124);LIB.inspect(J,phase,ACTIONS.index(action),epoch,time_s,alpha,out)
    if LIB.pv_invalid():raise ValueError('Candidate PV query outside this day')
    return out[:4],out[4:].reshape(30,4)

def run(policy,alpha,seed,V=.1,weights=(100.,10.,1.),tasks=30,trace=False,sigma=.24):
    if not np.isfinite(sigma) or not 0<sigma<1:raise ValueError('sigma must lie in (0,1)')
    donors=np.random.default_rng(seed).integers(0,30,size=4*tasks,dtype=np.uint8)
    summary=np.zeros(24);details=np.zeros((4*tasks,28)) if trace else None
    LIB.run(tasks,POLICIES.index(policy),V,np.asarray(weights,dtype=float),sigma,alpha,0.,donors,summary,
      details.ctypes.data_as(ctypes.c_void_p) if trace else None)
    assert not LIB.pv_invalid(),'An actual epoch crossed midnight or no past observation was available'
    t,d,pv,es,h,qs,qe,qm,js=summary[:9]
    assert np.isfinite(summary).all() and t<=86400 and int(summary[17])==tasks
    row=dict(policy=policy,alpha=alpha,seed=seed,V=V,sigma_e=sigma,w1=weights[0],w2=weights[1],w3=weights[2],
      tasks=tasks,epochs=4*tasks,T_s=t,demand_J=d,pv_J=pv,shortfall_J=es,epsilon_s=h,TARS_W=es/t,DWAE=h/t,
      mean_Q_s=qs/(4*tasks),terminal_Q_s=qe,max_Q_s=qm,mean_terminal_J=js/tasks,hours=t/3600,
      donor_sha256=hashlib.sha256(donors.tobytes()).hexdigest())
    for a,n in zip(ACTIONS,summary[9:14]):row[f'action_{a}_count']=int(n)
    for phase,n in zip([1,2,3],summary[14:17]):row[f'phase_{phase}_count']=int(n)
    row.update(violated=bool(h/t>sigma+1e-12),queue_bound_slack=sigma+qe/t-h/t)
    row.update(mean_observation_age_s=summary[18]/(4*tasks),max_observation_age_s=summary[19],
      epochs_observation_age_over_300s=int(summary[20]),epochs_observation_age_over_600s=int(summary[21]),
      mean_observed_power_W=summary[22]/(4*tasks),estimated_PV_total_J=summary[23])
    return row,pd.DataFrame(details,columns=TRACE_COLUMNS) if trace else None
