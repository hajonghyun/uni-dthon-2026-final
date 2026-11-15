"""Causal relative measurements and train-calibrated event-memory feature groups.

Events are sensor-derived proxies, not recovered AI-Hub annotation flags.
No test-set fitting, future observations, or semantic experiment IDs are used.
"""
import numpy as np
import pandas as pd
from base_features import GASES

CUE_FLOORS={'voltage_rate':.002,'warming':.02,'rapid_heat':.08,'terminal_warming':.02,
            'pressure_rise':.0001,'gas_burst':.5,'cooling':.25,'heater_cross':.01,
            'near_zero_voltage':.95,'thermal_clip':.5}
BUNDLES={
 'base':[], 'relative':['relative'], 'relative_gas':['relative','gas'],
 'voltage_events':['voltage'], 'events':['voltage','events'],
 'relative_voltage':['relative','voltage'],
 'relative_events':['relative','voltage','events'],
 'full':['relative','gas','voltage','events']}


def _initial(a,t):
    """Median of arrived observations in the first30s; freeze only after they arrive."""
    out=np.full(len(a),np.nan);early=np.flatnonzero(t-t[0]<=30.)
    for i in range(min(len(a),len(early))):
        valid=a[:i+1][np.isfinite(a[:i+1])]
        if len(valid):out[i]=np.median(valid)
    if len(early)<len(a):out[len(early):]=out[len(early)-1]
    return out


def _slope(a,t,seconds=30.):
    lag=np.maximum(np.searchsorted(t,t-seconds,side='right')-1,0)
    dt=t-t[lag];out=np.full(len(a),np.nan);valid=dt>0
    out[valid]=(a[valid]-a[lag[valid]])/dt[valid]
    return out


def prepare(frame):
    """Fit-free quantities; restore arbitrary input row order after sorting groups."""
    frame=frame.reset_index(drop=True);n=len(frame)
    relative={};gas={};cues={k:np.full(n,np.nan) for k in CUE_FLOORS}
    state={'time':frame.relative_time_s.to_numpy(float),'dt':np.full(n,np.nan),
           'voltage':frame.meas_v.to_numpy(float)}
    for _,indices in frame.groupby('experiment_id',sort=False).indices.items():
        order=indices[np.argsort(state['time'][indices],kind='stable')];t=state['time'][order]
        if not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('Finite unique group timestamps required.')
        dt=np.r_[np.nan,np.diff(t)];state['dt'][order]=dt
        def put(target,name,values):
            if name not in target:target[name]=np.full(n,np.nan)
            target[name][order]=values
        v=frame.meas_v.to_numpy(float)[order];p=frame.meas_p.to_numpy(float)[order]
        heater=frame.indu_t.to_numpy(float)[order];ambient=frame.atmp_t.to_numpy(float)[order]
        previous=np.r_[np.nan,v[:-1]]
        cues['voltage_rate'][order]=np.maximum((previous-v)/dt,0)/np.maximum(np.abs(previous),.1)
        v0=_initial(v,t);p0=_initial(p,t)
        put(relative,'relative_voltage_initial',v/np.maximum(np.abs(v0),.1))
        put(relative,'relative_pressure_initial',p-p0)
        put(relative,'relative_pressure_fraction',(p-p0)/np.maximum(np.abs(p0),1.))
        cues['pressure_rise'][order]=np.maximum(_slope(p,t),0)/np.maximum(np.abs(p0),1.)
        cues['near_zero_voltage'][order]=1-np.minimum(np.abs(v)/np.maximum(np.abs(v0),.1),1.)
        for name in ['meas_t1','meas_t2','thermal_t']:
            a=frame[name].to_numpy(float)[order];a0=_initial(a,t);rate=_slope(a,t)
            peak=np.maximum.accumulate(np.where(np.isfinite(a),a,-np.inf))
            rate_max=np.maximum.accumulate(np.nan_to_num(np.abs(rate),nan=0.))
            put(relative,name+'_relative_rise',a-a0)
            put(relative,name+'_rate_relative_peak',rate/np.maximum(rate_max,.02))
            put(relative,name+'_cooling_fraction',np.maximum(peak-a,0)/np.maximum(peak-a0,10.))
            denominator=heater-a0
            valid=np.isfinite(denominator)&(denominator>5.)
            ratio=np.full(len(t),np.nan);ratio[valid]=(a[valid]-a0[valid])/denominator[valid]
            put(relative,name+'_heater_rise_ratio',np.clip(ratio,-5,10))
            put(relative,name+'_heater_ratio_available',valid.astype(float))
            if name=='meas_t1':
                cues['warming'][order]=np.maximum(rate,0);cues['rapid_heat'][order]=np.maximum(rate,0)
                cues['cooling'][order]=np.maximum(peak-a,0)/np.maximum(peak-a0,10.)
                cues['heater_cross'][order]=np.maximum(a-heater,0)/np.maximum(np.abs(heater),20.)
            if name=='meas_t2':cues['terminal_warming'][order]=np.maximum(rate,0)
        cues['thermal_clip'][order]=(frame.thermal_t.to_numpy(float)[order]>=509.9).astype(float)
        gaslog=np.log1p(np.maximum(frame[GASES].to_numpy(float)[order],0.));total=np.nansum(gaslog,axis=1)
        for k,name in enumerate(GASES):put(gas,name+'_relative_initial_log',gaslog[:,k]-_initial(gaslog[:,k],t))
        delta=total-_initial(total,t);put(gas,'gas_sum_relative_initial_log',delta)
        cues['gas_burst'][order]=np.maximum(delta,0)
    return {'relative':pd.DataFrame(relative).astype(np.float32),'gas':pd.DataFrame(gas).astype(np.float32),
            'cues':pd.DataFrame(cues).replace([np.inf,-np.inf],np.nan).astype(np.float32),'state':state}


def calibrate(views,groups):
    """Median of per-experiment early-phase99th percentiles; group-balanced noise."""
    groups=set(groups);thresholds={};rows_used={}
    for cue,floor in CUE_FLOORS.items():
        values=[];count=0
        for v in views:
            f=v['frame'];labels=[1,2] if cue=='rapid_heat' else [1]
            mask=f.experiment_id.isin(groups)&f.label.isin(labels)
            for _,g in f.loc[mask].groupby('experiment_id',sort=False):
                a=v['prepared']['cues'][cue].iloc[g.index].to_numpy(float);a=a[np.isfinite(a)]
                if len(a):values.append(float(np.quantile(a,.99)));count+=len(a)
        thresholds[cue]=max(floor,float(np.median(values)) if values else floor);rows_used[cue]=count
    thresholds['rapid_heat']=max(thresholds['rapid_heat'],2*thresholds['warming'])
    return {'thresholds':thresholds,'training_groups':sorted(groups),'early_rows_by_cue':rows_used,
            'quantile':.99,'normal_rule':'label1, rapid heat labels1/2','voltage_episode_fraction':.08,
            'voltage_quiet_seconds':15.,'sustained_seconds':10.}


def _memory(cue,t,threshold,hold):
    active=np.isfinite(cue)&(cue>threshold);start=np.maximum.accumulate(np.where(active&~np.r_[False,active[:-1]],t,t[0]))
    duration=np.where(active,t-start,0.);confirmed=active&(duration>=hold)
    first=np.minimum.accumulate(np.where(confirmed,t,np.inf));last=np.maximum.accumulate(np.where(confirmed,t,-np.inf))
    ever=np.isfinite(first);onset=confirmed&~np.r_[False,confirmed[:-1]]
    score=np.clip(np.nan_to_num(cue/threshold,nan=0.),0,20)
    return {'ever':ever.astype(float),'age':np.where(ever,np.log1p(np.maximum(t-first,0)),np.nan),
            'last_age':np.where(np.isfinite(last),np.log1p(np.maximum(t-last,0)),np.nan),
            'count':np.minimum(np.cumsum(onset),3).astype(float),'active_duration':np.log1p(duration),
            'strength':score,'peak_strength':np.maximum.accumulate(score)},first


def event_features(frame,prepared,calibration):
    n=len(frame);voltage={};other={};threshold=calibration['thresholds'];times=prepared['state']['time']
    for _,indices in frame.groupby('experiment_id',sort=False).indices.items():
        order=indices[np.argsort(times[indices],kind='stable')];t=times[order]
        def put(target,name,values):
            if name not in target:target[name]=np.full(n,np.nan)
            target[name][order]=values
        first={}
        for cue in CUE_FLOORS:
            if cue=='voltage_rate':continue
            target=voltage if cue=='near_zero_voltage' else other
            memory,first[cue]=_memory(prepared['cues'][cue].to_numpy(float)[order],t,threshold[cue],calibration['sustained_seconds'])
            for name,value in memory.items():put(target,'event_'+cue+'_'+name,value)
        v=prepared['state']['voltage'][order];dt=prepared['state']['dt'][order]
        rate=prepared['cues']['voltage_rate'].to_numpy(float)[order]
        counts=np.zeros(len(t));ages1=np.full(len(t),np.nan);ages2=np.full(len(t),np.nan)
        firsttime=np.inf;secondtime=np.inf;lasttime=np.inf;count=0;quiet=1e9;dose=0.;denominator=.1;counted=False
        drops=np.zeros(len(t));between=np.full(len(t),np.nan);since_last=np.full(len(t),np.nan)
        for i in range(1,len(t)):
            active=np.isfinite(rate[i]) and rate[i]>threshold['voltage_rate']
            if active:
                if quiet>=calibration['voltage_quiet_seconds']:
                    denominator=max(abs(v[i-1]),.1);dose=0.;counted=False
                quiet=0.;dose+=max(v[i-1]-v[i],0)/denominator
                if dose>=calibration['voltage_episode_fraction'] and not counted:
                    count=min(count+1,3);counted=True;lasttime=t[i]
                    if count==1:firsttime=t[i]
                    if count==2:secondtime=t[i]
            else:quiet+=dt[i]
            counts[i]=count;drops[i]=min(dose,5)
            if np.isfinite(firsttime):ages1[i]=np.log1p(t[i]-firsttime)
            if np.isfinite(secondtime):ages2[i]=np.log1p(t[i]-secondtime);between[i]=np.log1p(secondtime-firsttime)
            if np.isfinite(lasttime):since_last[i]=np.log1p(t[i]-lasttime)
        for name,value in [('count',counts),('first_age',ages1),('second_age',ages2),('between_first_second',between),
                            ('last_age',since_last),('episode_drop_fraction',drops),('first_seen',(counts>=1).astype(float)),
                            ('second_seen',(counts>=2).astype(float))]:put(voltage,'voltage_event_'+name,value)
        first_drop=np.where(np.isfinite(ages1),t-(np.expm1(np.nan_to_num(ages1))),np.inf)
        for name in ['warming','pressure_rise','gas_burst','cooling']:
            known=np.isfinite(first_drop)&np.isfinite(first[name]);difference=np.full(len(t),np.nan)
            difference[known]=np.clip((first[name][known]-first_drop[known])/120.,-10,10)
            put(other,'event_'+name+'_time_minus_first_voltage',difference)
    return {'voltage':pd.DataFrame(voltage).replace([np.inf,-np.inf],np.nan).astype(np.float32),
            'events':pd.DataFrame(other).replace([np.inf,-np.inf],np.nan).astype(np.float32)}


def blocks(frame,prepared,calibration):
    return {'relative':prepared['relative'],'gas':prepared['gas'],**event_features(frame,prepared,calibration)}


def assemble(base,feature_blocks,bundle):
    return pd.concat([base]+[feature_blocks[k] for k in BUNDLES[bundle]],axis=1)
