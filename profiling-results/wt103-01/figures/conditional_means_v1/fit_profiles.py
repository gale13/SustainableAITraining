"""WT103 fixed-action moment calibration. Run in the wt103-01 results directory."""
from pathlib import Path
import csv, json
import numpy as np

OUT = Path("figures/conditional_means_v1")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
BATCH = [4,8,10,16,24,32,48,64]
def read(path):
    return json.loads(Path(path).read_text())
def save(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
audit=read("training/audit.json")
assert audit["integrity_checks_passed"] and audit["common_completed_blocks"]==19
assert read("recovery_c16.json")["first_patched_block_id"]==16
data=np.full((19,8,229,3),np.nan)
with open("training/all_epochs.csv") as f:
    for r in csv.DictReader(f):
        t=int(r["block_id"]); b=BATCH.index(int(r["batch_sequences"])); e=int(r["epoch_in_block"])-1
        assert np.isnan(data[t,b,e,0]) and int(r["global_epoch"])==t*229+e+1
        assert r["train_energy_valid"]=="True"
        data[t,b,e]=[float(r[k]) for k in ("train_loss","train_seconds","train_energy_j")]
assert np.isfinite(data).all() and np.all(data>0) and data.size//3==audit["row_count"]

def derive(h):
    vals=data[:,:,:h]
    loss=vals[:,:,:,0]
    j=np.minimum.accumulate(loss,axis=2)/(loss[:,:,:1]+EPS)
    x,y=j[:,:,:-1],j[:,:,1:]  # start/end of epochs 2..h
    rel=(y-x)/(x+EPS)
    # W=2. With no pre-task loss, there is no known epoch-1 delta.
    # The first usable average (epochs 2 and 3) first enters decision 4.
    avg=(rel[:,:,:-1]+rel[:,:,1:])/2
    return x,y,rel,avg

def phases(rel,avg,betas):
    ph=np.ones(rel.shape,dtype=int)
    if len(betas):
        ph[:,:,2:]=np.searchsorted(betas,avg[:,:,:-1],side="left")+1
        ph=np.maximum.accumulate(ph,axis=2)
    return ph

def phase_selection(h,rel,avg):
    # Calibrate candidate thresholds on blocks 1..10, compare on blocks 11..16.
    grid=[]
    for k in (1,2,3,4):
        beta=np.quantile(avg[:10,:,:-1],np.arange(1,k)/k) if k>1 else np.array([])
        ph=phases(rel,avg,beta)
        residual=[]; min_tasks=20; min_count=10**6
        for b in range(8):
            for z in range(1,k+1):
                fit=ph[:10,b]==z
                min_tasks=min(min_tasks,int(fit.any(axis=1).sum()))
                min_count=min(min_count,int(fit.sum()))
                test=ph[10:16,b]==z
                if test.any() and fit.any():
                    residual.extend(rel[10:16,b][test]-rel[:10,b][fit].mean())
        rmse=float(np.sqrt(np.mean(np.square(residual)))) if len(residual)==rel[10:16].size else np.inf
        grid.append(dict(horizon=h,K=k,beta=";".join(f"{v:.10g}" for v in beta),min_calibration_blocks=min_tasks,
                         min_rows=min_count,gain_rmse=rmse,eligible=int(min_tasks>=5 and min_count>=10)))
    best=min(row["gain_rmse"] for row in grid if row["eligible"])
    # Simplest supported classifier within 5% relative RMSE of the best.
    sel=next(row for row in grid if row["eligible"] and row["gain_rmse"]<=1.05*best)
    beta=np.array([float(v) for v in sel["beta"].split(";") if v])
    return sel,grid,phases(rel,avg,beta)

def fit_predict(x,p,y,u,q,k,form):
    # Center J at 1, pooled slope across phases for this B.
    def design(x,p):
        cols=[np.ones(x.size)]
        if form in ("B_J","B_J_phi"): cols.append(x.ravel()-1)
        if form=="B_J_phi": cols.extend((p.ravel()==z).astype(float) for z in range(2,k+1))
        return np.stack(cols,axis=1)
    a=design(x,p)
    coef=np.linalg.lstsq(a,y.ravel(),rcond=None)[0]
    return design(u,q)@coef,coef,int(np.linalg.matrix_rank(a))

gridrows=[]; states=[]; diagnostics=[]; coeff=[]; tables=[]; reports=[]
for horizon in [10,229]:
    x,y,rel,avg=derive(horizon)
    sel,grid,ph=phase_selection(horizon,rel,avg)
    k=sel["K"]; gridrows+=grid
    reports.append(f"H={horizon} K={k} betas={sel['beta']} gridRMSE="+str([round(r["gain_rmse"],7) for r in grid]))
    for regime,train,test,full in [("original",slice(0,10),slice(10,16),slice(0,16)),
                                  ("math",slice(16,18),slice(18,19),slice(16,19))]:
        totals={}
        for bi,b in enumerate(BATCH):
            td=data[full,bi,:horizon,1:3]
            meanT,meanD=td.mean(axis=(0,1))
            tables.append(dict(horizon=horizon,backend=regime,batch=b,blocks=td.shape[0],rows=td.shape[0]*horizon,
                               m_T_seconds=meanT,m_D_joules=meanD,power_ratio_w=meanD/meanT))
            v=data[:,bi,1:horizon,1:3]
            targets={"T_s":v[:,:,0],"D_J":v[:,:,1],"epsilon_s":v[:,:,0]*y[:,bi]}
            xx=x[:,bi]; pp=ph[:,bi]
            for z in range(1,k+1):
                m=pp[full]==z
                states.append(dict(horizon=horizon,backend=regime,batch=b,phase=z,rows=m.sum(),
                               blocks=m.any(axis=1).sum(),J_min=xx[full][m].min() if m.any() else "",
                               J_max=xx[full][m].max() if m.any() else "",
                               mean_rel_change=rel[full,bi][m].mean() if m.any() else ""))
            for name,arr in targets.items():
                for form in ("B","B_J","B_J_phi"):
                    pred,_,rank=fit_predict(xx[train],pp[train],arr[train],xx[test],pp[test],k,form)
                    yy=arr[test].ravel()
                    nbad=int(((xx[test]<xx[train].min()) | (xx[test]>xx[train].max())).sum())
                    # Normalize per action, so large batches/slow actions do not dominate.
                    rmse=float(np.sqrt(np.mean((pred-yy)**2))/yy.mean())
                    key=name,form
                    totals.setdefault(key,[]).append(rmse)
                    diagnostics.append(dict(horizon=horizon,backend=regime,batch=b,target=name,form=form,
                           nrmse=rmse,test_rows=len(yy),outside_action_J_range=nbad,train_rank=rank))
                    _,c,rank=fit_predict(xx[full],pp[full],arr[full],xx[full],pp[full],k,form)
                    coeff.append(dict(horizon=horizon,backend=regime,batch=b,target=name,form=form,
                           rows=arr[full].size,rank=rank,intercept=c[0],coef_J_minus_1=c[1] if form!="B" else "",
                           phase_offsets=";".join(f"{i:.11g}" for i in c[2:]),J_min=xx[full].min(),J_max=xx[full].max()))
        for target in ("T_s","D_J","epsilon_s"):
            rr=[100*np.mean(totals[target,f]) for f in ("B","B_J","B_J_phi")]
            reports.append(f"{regime} {target} macroNRMSE(%), B/BJ/BJphi "+" ".join(f"{v:.2f}" for v in rr))
    counts=[int((ph[16:]==z).sum()) for z in range(1,k+1)]
    reports.append("math phase n="+str(counts))
save("phase_selection.csv",gridrows);save("phase_support.csv",states)
save("heldout_errors.csv",diagnostics);save("moment_coefficients.csv",coeff);save("action_costs.csv",tables)
(OUT/"diagnostics.txt").write_text("\n".join(reports)+"\n")
print("\n".join(reports))

# A physically compatible predictor for (m_T, m_epsilon), using a shared m_T.
# Estimate the T-weighted relative progress directly within each (B, phase) cell:
# E[T J_next|s,a] = J * E[T|s,a] * E[T (J_next/J)|s,a] / E[T|s,a].
# Pool the bounded ratio either by action or by action and observable phase.
pair_scores=[]; retained=[]; checks=[]
for h in (10,229):
    x,y,rel,avg=derive(h)
    sel,_,p=phase_selection(h,rel,avg)
    for regime,fit,test,full in [("original",slice(0,10),slice(10,16),slice(0,16)),
                                 ("math",slice(16,18),slice(18,19),slice(16,19))]:
        for bi,b in enumerate(BATCH):
            T=data[:,bi,1:h,1]; xx=x[:,bi]; yy=y[:,bi]; ph=p[:,bi]
            for byphase in [False,True]:
                groups=range(1,sel["K"]+1) if byphase else [0]
                pred=np.full_like(T[test],np.nan)
                for z in groups:
                    mask=(ph[fit]==z) if byphase else np.ones_like(ph[fit],dtype=bool)
                    take=(ph[test]==z) if byphase else np.ones_like(ph[test],dtype=bool)
                    allmask=(ph[full]==z) if byphase else np.ones_like(ph[full],dtype=bool)
                    for source,m,mode in [(fit,mask,"holdout"),(full,allmask,"refit")]:
                        if not m.any(): continue
                        t=T[source][m]
                        rho=float(np.average(yy[source][m]/xx[source][m],weights=t))
                        assert 0<=rho<=1
                        tau=data[source,bi,:h,1].mean()
                        if mode=="holdout": pred[take]=tau*xx[test][take]*rho
                        else:
                            retained.append(dict(horizon=h,backend=regime,batch=b,phase=z,mean_T_s=tau,
                                time_weighted_retention=rho,count=m.sum(),blocks=m.any(axis=1).sum(),
                                supported_J_min=xx[source][m].min(),supported_J_max=xx[source][m].max()))
                obs=T[test]*yy[test]
                if np.isfinite(pred).all():
                    pair_scores.append(dict(horizon=h,backend=regime,batch=b,phase_specific=int(byphase),
                                       nrmse=float(np.sqrt(np.mean((obs-pred)**2))/obs.mean())))
            if h==10 and regime=="math":
                # Independent state-dependent demand vs constant duration does not imply constant power.
                checks.append((b,round(float(xx[full].min()),4),round(float(xx[full].max()),4)))
save("bounded_epsilon.csv",retained);save("bounded_heldout.csv",pair_scores)
with (OUT/"diagnostics.txt").open("a") as f:
    for h in (10,229):
        for reg in ["original","math"]:
            vals=[]
            for z in [0,1]:
                sub=[r["nrmse"] for r in pair_scores if (r["horizon"],r["backend"],r["phase_specific"])==(h,reg,z)]
                vals.append(round(100*np.mean(sub),3) if len(sub)==8 else None)
            print("bounded eps",h,reg,"pooled/phase-specific NRMSE %",vals,file=f)
    print("math10 observed J ranges",checks,file=f)
