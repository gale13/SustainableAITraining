"""Causal measurement stream assembled exclusively with the public PV API."""
import numpy as np
from opencem_pv import PVDatabase,DataGapError


def observation_stream(pv):
    current=pv.observations()
    valid=current[current.pv_power_w.notna()].sort_values('read_ts')
    times=valid.read_ts.to_numpy(dtype=float)-pv.start_timestamp
    powers=valid.pv_power_w.to_numpy(dtype=float)
    # Midnight's interpolated power can involve a future observation. Fetch the
    # genuine last earlier measurement using public preceding-day observations.
    if len(times)==0 or times[0]>0:
        previous=None
        with PVDatabase() as db:
            dates=sorted(d for d in db.days().date.astype(str) if d<pv.date)
            for date in reversed(dates):
                history=db.load_day(date,max_gap_s=None).observations()
                history=history[history.pv_power_w.notna() & (history.read_ts<=pv.start_timestamp)]
                if len(history):
                    previous=history.sort_values('read_ts').iloc[-1]
                    break
        if previous is None:
            raise DataGapError(f'{pv.date}: no valid PV measurement observed by midnight')
        times=np.r_[float(previous.read_ts)-pv.start_timestamp,times]
        powers=np.r_[float(previous.pv_power_w),powers]
    assert len(times)>0 and times[0]<=0 and np.all(np.diff(times)>0)
    assert np.isfinite(powers).all() and np.all(powers>=0)
    return np.ascontiguousarray(times),np.ascontiguousarray(powers)


def latest(times,powers,time_s,scale=1.):
    if not np.isfinite(time_s) or not 0<=time_s<=86400:
        raise ValueError('Decision clock must be inside the calendar day')
    i=int(np.searchsorted(times,time_s,side='right')-1)
    if i<0:raise DataGapError('No past PV observation available')
    return float(scale*powers[i]),float(times[i]),float(time_s-times[i])
