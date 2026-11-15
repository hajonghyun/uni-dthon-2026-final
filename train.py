"""Train the fixed E1 recipe from a labeled CSV and actual PNG images.

Default output is a NEW retrained_weights directory. The submitted weights/
and CSV are never overwritten. --smoke reduces training only to check the
execution path; its outputs are not submission candidates.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import boundary_features as BF
from boundary_network import Expert
import hgb_model as H
from learner import StageXGB
import memory_features as MF
import model as S20

PACKAGE = Path(__file__).resolve().parent


def package_versions():
    packages = {p: version(p) for p in
                ['numpy', 'pandas', 'scikit-learn', 'torch', 'torchvision']}
    import xgboost
    packages['xgboost_library'] = xgboost.__version__
    return packages


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def feature_config():
    return H.FeatureConfig.from_dict(json.loads((PACKAGE/'training_config.json').read_text())['feature_config'])


def view_indices(frame, stride, offset=0):
    """Return original row order after selecting each time-sorted experiment."""
    times = frame.relative_time_s.to_numpy(float)
    selected = []
    for rows in frame.groupby('experiment_id', sort=False).indices.values():
        order = rows[np.argsort(times[rows], kind='stable')]
        selected.extend(order[offset::stride])
    return np.sort(np.asarray(selected, dtype=int))


def build_views(frame, images, config):
    """Recompute causal features for six observation views, without caches."""
    views = []
    for stride in (1, 2, 3):
        for offset in range(stride):
            indices = view_indices(frame, stride, offset)
            sub = frame.iloc[indices].reset_index(drop=True)
            x = H.build_features(sub, images, config)
            if x.shape != (len(sub), 788):
                raise ValueError('Expected the original 788-feature schema.')
            views.append({'stride': stride, 'offset': offset, 'indices': indices,
                          'frame': sub, 'x': x})
    coverage = np.bincount(np.concatenate([v['indices'] for v in views]), minlength=len(frame))
    if not np.array_equal(coverage, np.full(len(frame), 3)):
        raise ValueError('Each physical row must occur exactly three times.')
    return views


def train_hgb(views, config, output, iterations):
    # Match voltage_v11: calibration uses stride1/2/3 OFFSET ZERO only.
    calibration_views = [{'frame': v['frame'], 'prepared': MF.prepare(v['frame'])}
                         for v in views if v['offset'] == 0]
    frame = views[0]['frame']
    groups = set(frame.experiment_id)
    calibration = MF.calibrate(calibration_views, groups)
    base = views[0]['x']
    extra = S20.voltage(frame, calibration_views[0]['prepared'], calibration, base.index)
    x = S20.attach(base, extra)
    if x.shape[1] != 803 or set(calibration['training_groups']) != groups:
        raise ValueError('HGB feature schema or calibration scope differs.')
    pipeline = S20.make_model(x.columns, iterations=iterations)
    pipeline.fit(x, frame.label.to_numpy(int))
    bundle = {'pipeline': pipeline, 'features': list(x), 'original_features': list(base),
              'feature_config': config.to_dict(), 'calibration': calibration,
              'training_groups': sorted(groups),
              'candidate': 'fixed voltage803 plus sensor_only 4/5 expert'}
    joblib.dump(bundle, output/'s20/hgb.joblib')
    return {'features': len(x.columns), 'training_rows': len(frame),
            'iterations': iterations, 'calibration_views': ['stride1_offset0', 'stride2_offset0', 'stride3_offset0']}


def sampling_pools(views):
    pools = []
    for view in views:
        frame = view['frame']
        by_class = []
        for label in range(1, 7):
            rows = np.flatnonzero(frame.label.eq(label).to_numpy())
            by_class.append([rows[a] for a in frame.iloc[rows].groupby('experiment_id', sort=False).indices.values()])
        if not all(by_class):
            raise ValueError('Each offset-zero expert view needs all six training classes.')
        pools.append(by_class)
    return pools


def train_expert(views, output, device, smoke):
    """Original boundary_v6 sensor-only recipe, including sampling and loss."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(42)
    rng = np.random.default_rng(42)
    expert_views = [v for v in views if v['offset'] == 0]
    features = [BF.build_features(v['frame']) for v in expert_views]
    prep = BF.fit_preprocessing(features[0])
    xs = [BF.transform(x, prep) for x in features]
    pools = sampling_pools(expert_views)
    model = Expert(xs[0].shape[1], 'sensor_only').to(device)
    optimizer = torch.optim.AdamW([{'params': list(model.parameters()), 'lr': 1e-3}], weight_decay=1e-3)
    epochs = 1 if smoke else 20
    steps = 2 if smoke else 6144 // 64
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        history = []
        for _ in range(steps):
            vi = int(rng.choice(3, p=[.6, .2, .2]))
            labels = rng.choice(6, size=64, p=[.10, .15, .15, .25, .25, .10])
            rows = []
            for label in labels:
                groups = pools[vi][label]
                group = groups[int(rng.integers(len(groups)))]
                rows.append(group[int(rng.integers(len(group)))])
            sensor = torch.from_numpy(xs[vi][np.asarray(rows)]).to(device)
            sensor = sensor + .01 * torch.randn_like(sensor)
            y = torch.as_tensor(labels, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
            with context:
                pair, aux, _ = model(sensor)
                boundary = (y == 3) | (y == 4)
                loss = F.cross_entropy(pair[boundary], y[boundary]-3, label_smoothing=.02)
                loss = loss + .2 * F.cross_entropy(aux, y, label_smoothing=.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            history.append(float(loss.detach()))
        scheduler.step()
        losses.append(float(np.mean(history)))
        print(json.dumps({'stage': 'expert_epoch', 'epoch': epoch, 'epochs': epochs,
                          'loss': losses[-1]}), flush=True)
    joblib.dump(prep, output/'s20/expert_preprocessing.joblib')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output/'s20/expert.pt')
    return {'kind': 'sensor_only', 'input_features': len(prep['columns']),
            'input_dimension_with_missing_flags': xs[0].shape[1],
            'epochs': epochs, 'steps_per_epoch': steps, 'batch': 64,
            'seed': 42, 'device': device, 'losses': losses,
            'cuda_training_autocast': 'bfloat16' if device == 'cuda' else None}


def train_multirate(views, config, output, iterations, device):
    frame = views[0]['frame']
    columns = list(views[0]['x'])
    pre = H.build_model(model_kind='hist', feature_columns=columns,
                        categorical_columns=H.CATEGORIES).named_steps['preprocessor']
    # Match the submitted recipe: fit preprocessing only on dense training rows.
    pre.fit(views[0]['x'])
    x = np.concatenate([pre.transform(v['x']) for v in views])
    y = np.concatenate([v['frame'].label.to_numpy(int) for v in views])
    origin = np.concatenate([v['indices'] for v in views])
    weight = np.full(len(y), 1/3, dtype=np.float64)
    physical_weight = np.bincount(origin, weights=weight, minlength=len(frame))
    if not np.allclose(physical_weight, 1., atol=1e-12):
        raise ValueError('Multirate physical-row weight differs from one.')
    for label in range(1, 7):
        if abs(float(weight[y == label].sum()) - float(frame.label.eq(label).sum())) > 1e-8:
            raise ValueError('Multirate weighted class prior changed.')
    model = StageXGB(iterations=iterations, device=device).fit(x, y, weight)
    pipeline = Pipeline([('preprocessor', pre), ('classifier', model)])
    bundle = {'pipeline': pipeline, 'features': columns, 'feature_config': config.to_dict(),
              'arm': 'MULTIRATE', 'training_groups': sorted(set(frame.experiment_id)),
              'final_train78_candidate': frame.experiment_id.nunique() == 78,
              'audit_only_fold_model': False}
    joblib.dump(bundle, output/'mrxgb/model.joblib', compress=3)
    return {'features': len(columns), 'physical_rows': len(frame), 'augmented_rows': len(y),
            'total_sample_weight': float(weight.sum()), 'per_physical_row_weight_one': True,
            'weighted_class_prior_preserved': True, 'iterations': iterations,
            'device_at_training': device, 'device_at_inference': 'cpu'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Labeled training CSV only; exclude validation and Test for E1 #25.')
    parser.add_argument('--image_root', required=True, type=Path)
    parser.add_argument('--weights_dir', type=Path, default=PACKAGE/'retrained_weights',
                        help='New directory; existing directories are rejected.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='Original training used CUDA. CPU is portable, not bit-identical to CUDA training.')
    parser.add_argument('--smoke', action='store_true', help='3 tree rounds and 1 short expert epoch; not a submission model.')
    args = parser.parse_args()
    destination = args.weights_dir.resolve()
    frozen = PACKAGE/'weights'
    if destination == frozen or frozen in destination.parents:
        parser.error('The preserved submitted weights/ directory is protected.')
    if destination.exists():
        parser.error('weights_dir already exists; choose a new directory.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Use --device cpu or a CUDA-enabled environment.')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = feature_config()
    frame = pd.read_csv(args.input).reset_index(drop=True)
    H.validate_frame(frame, config)
    if 'label' not in frame:
        parser.error('Training CSV must contain labels; never use Test for fitting.')
    y = pd.to_numeric(frame.label, errors='raise')
    if not np.isfinite(y).all() or not y.eq(y.round()).all() or set(y.astype(int)) != set(range(1, 7)):
        parser.error('Training labels must be integers and include all classes 1..6.')
    frame['label'] = y.astype(int)
    # Detect invalid times before fitting any preprocessing or model.
    if frame.duplicated(['experiment_id', 'relative_time_s']).any():
        parser.error('Timestamps must be unique within each experiment.')
    destination.mkdir(parents=True)
    (destination/'s20').mkdir()
    (destination/'mrxgb').mkdir()
    metadata = {'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'training_rows': len(frame), 'training_experiments': int(frame.experiment_id.nunique()),
                'training_scope': 'Only the explicitly provided labeled CSV; no validation/Test is read automatically.',
                'smoke_only': args.smoke, 'submitted_weights_replaced': False,
                'seed': 42, 'blend_weights': [.5, .5], 'feature_config': config.to_dict(),
                'original_training_device': 'cuda', 'requested_training_device': args.device,
                'python': sys.version, 'packages': package_versions()}
    write_json(destination/'training.json', metadata)
    started = time.monotonic()
    try:
        paths = H.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
        images = H.extract_image_features(frame, paths, n_jobs=2, config=config, progress=True)
        with threadpool_limits(limits=2):
            views = build_views(frame, images, config)
            metadata['views'] = [{'stride': v['stride'], 'offset': v['offset'],
                                  'rows': len(v['frame'])} for v in views]
            iterations = 3 if args.smoke else 300
            metadata['hgb'] = train_hgb(views, config, destination, iterations)
            metadata['expert'] = train_expert(views, destination, args.device, args.smoke)
            metadata['multirate'] = train_multirate(views, config, destination, iterations, args.device)
        metadata.update(status='complete', elapsed_seconds=time.monotonic()-started)
        write_json(destination/'training.json', metadata)
    except BaseException as error:
        metadata.update(status='failed', error=str(error), elapsed_seconds=time.monotonic()-started)
        write_json(destination/'training.json', metadata)
        raise
    print(json.dumps({'stage': 'training_complete', 'weights_dir': str(destination),
                      'smoke_only': args.smoke, 'seconds': metadata['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
