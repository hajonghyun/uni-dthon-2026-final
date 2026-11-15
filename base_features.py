"""Fold-fitted sensor features; all temporal calculations use observed current/past."""
import warnings

import numpy as np
import pandas as pd

SENSORS = ['meas_v','meas_t1','meas_t2','meas_p','meas_ch4','meas_co','meas_co2',
           'meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2',
           'atmp_t','thermal_t','indu_t']
GASES = SENSORS[4:14]
META = ['bat_class','bat_cc','soc_status','tr_method']


def build_features(frame):
    frame = frame.reset_index(drop=True)
    x = frame[SENSORS+META].apply(pd.to_numeric, errors='coerce').copy()
    x[GASES] = np.log1p(x[GASES].clip(lower=0))
    x['gas_log_sum'] = x[GASES].sum(axis=1, min_count=1)
    x['t1_minus_t2'] = x.meas_t1-x.meas_t2
    x['t1_minus_indu'] = x.meas_t1-x.indu_t
    x['thermal_minus_t1'] = x.thermal_t-x.meas_t1
    x['t1_minus_ambient'] = x.meas_t1-x.atmp_t
    x['thermal_clipped'] = (x.thermal_t >= 509.9).astype(float)
    extra = {}
    times = frame.relative_time_s.to_numpy(float)
    signals = ['meas_v','meas_t1','meas_t2','meas_p','thermal_t','t1_minus_indu','gas_log_sum']
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        order = idx[np.argsort(times[idx], kind='stable')]
        t = times[order]
        if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
            raise ValueError('Finite unique timestamps required within each experiment.')
        def put(name, values):
            if name not in extra: extra[name] = np.full(len(frame), np.nan)
            extra[name][order] = values
        v = x.meas_v.to_numpy()[order]
        maximum = np.maximum.accumulate(np.nan_to_num(v, nan=0))
        put('voltage_drop_cummax', v-maximum)
        put('voltage_fraction_cummax', v/np.maximum(maximum, .1))
        for name in signals:
            a = x[name].to_numpy(float)[order]
            put(name+'_instant_rate', np.r_[np.nan, np.diff(a)]/np.r_[np.nan, np.diff(t)])
            series = pd.Series(a, index=pd.to_timedelta(t, unit='s'))
            for seconds in (30, 120):
                rolling = series.rolling(f'{seconds}s', closed='both', min_periods=1)
                for suffix, value in [('mean', rolling.mean()),('max', rolling.max()),('std', rolling.std(ddof=0))]:
                    put(f'{name}_{suffix}_{seconds}s', value.to_numpy())
                lag = np.searchsorted(t, t-seconds, side='right')-1
                valid = lag >= 0
                rate = np.full(len(t), np.nan)
                rate[valid] = (a[valid]-a[lag[valid]])/(t[valid]-t[lag[valid]])
                put(f'{name}_rate_{seconds}s', rate)
    return pd.concat([x,pd.DataFrame(extra)],axis=1).replace([np.inf,-np.inf],np.nan).astype(np.float32)


def fit_preprocessing(features):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q = np.nanpercentile(features.to_numpy(), [25,50,75], axis=0)
    median = np.nan_to_num(q[1])
    scale = np.nan_to_num(q[2]-q[0], nan=1.)
    scale[scale < 1e-5] = 1.
    return {'columns':features.columns.tolist(),'median':median,'scale':scale}


def transform(features, prep):
    a = features[prep['columns']].to_numpy(np.float32)
    missing = ~np.isfinite(a)
    z = (np.where(missing, prep['median'], a)-prep['median'])/prep['scale']
    return np.concatenate([np.clip(z,-12,12),missing.astype(np.float32)],axis=1).astype(np.float32)


def route(anchor, conditional):
    p = np.array(anchor, copy=True)
    q = np.asarray(conditional)
    q = q/np.maximum(q.sum(-1,keepdims=True),1e-12)
    p[...,3:5] = p[...,3:5].sum(-1,keepdims=True)*q
    return p
