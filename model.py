"""Unchanged HGB and boundary expert with a single fixed voltage-event block."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import boundary_features as BF
from boundary_network import Expert
import hgb_model as H
import memory_features as MF


def voltage(frame, prepared, calibration, index=None):
    result = MF.event_features(frame, prepared, calibration)['voltage']
    if result.shape != (len(frame), 15):
        raise ValueError('Expected the unchanged fifteen voltage-event columns')
    if index is not None:
        result.index = index
    return result


def attach(base, extra):
    if not base.index.equals(extra.index) or set(base).intersection(extra):
        raise ValueError('Feature alignment or duplicate-column error')
    return pd.concat([base, extra], axis=1)


def make_model(columns, iterations=300):
    return H.build_model(model_kind='hist', feature_columns=list(columns),
                         categorical_columns=H.CATEGORIES, seed=42,
                         iterations=iterations, depth=4, learning_rate=.05,
                         add_missing_indicators=True)


def route(p, q):
    result = p.copy()
    q = q / q.sum(-1, keepdims=True)
    result[..., 3:5] = p[..., 3:5].sum(-1, keepdims=True) * q
    if not np.allclose(result.sum(-1), 1, atol=1e-6):
        raise ValueError('Invalid routed probabilities')
    return result


def predict_matrix(bundle, frame, base):
    extra = voltage(frame, MF.prepare(frame), bundle['calibration'], base.index)
    x = attach(base[bundle['original_features']], extra)
    if list(x) != bundle['features']:
        raise ValueError('Unexpected feature schema')
    with threadpool_limits(limits=2):
        return bundle['pipeline'].predict_proba(x)


def conditional(frame, weights, batch=512):
    torch.set_num_threads(2)
    prep = joblib.load(Path(weights) / 'expert_preprocessing.joblib')
    x = BF.transform(BF.build_features(frame), prep)
    expert = Expert(x.shape[1], 'sensor_only').eval()
    expert.load_state_dict(torch.load(Path(weights) / 'expert.pt',
                                      map_location='cpu', weights_only=True))
    values = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            logits, _, _ = expert(torch.from_numpy(x[start:start + batch]))
            values.append(logits.softmax(-1).numpy())
    return np.concatenate(values)


def predict_frame(frame, image_root, weights, *, image_features=None, batch=512):
    weights = Path(weights)
    bundle = joblib.load(weights / 'hgb.joblib')
    config = H.FeatureConfig.from_dict(bundle['feature_config'])
    H.validate_frame(frame, config)
    if image_features is None:
        image_root = Path(image_root)
        paths = H.resolve_image_paths(frame, [image_root, image_root.parent])
        image_features = H.extract_image_features(frame, paths, n_jobs=2, config=config)
    base = H.build_features(frame, image_features, config)[bundle['original_features']]
    p = predict_matrix(bundle, frame, base)
    return route(p, conditional(frame, weights, batch=batch))
