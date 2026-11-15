"""공유 피처·모델 코드. 실제 대회 데이터는 서버에서 사용자가 실행할 때만 읽는다."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LABELS = list(range(1, 7))
SENSORS = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']
EXTRA_SENSORS = [
    'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
    'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2', 'atmp_t', 'thermal_t',
    'indu_t', 'indu_t.1', 'meas_t1_plus_t2',
]
CATEGORIES = ['bat_class', 'soc_status', 'tr_method']
IMAGE_STATS = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio']
WINDOW_SIGNALS = SENSORS + ['img_max', 'img_p99']
WINDOW_S = 30.0
HOT_THRESHOLD = 0.8  # 회색조 밝기 임계값이며 실제 온도 임계값이 아니다.
FEATURE_VERSION = 2


@dataclass(frozen=True)
class FeatureConfig:
    feature_set: str = 'baseline'
    sensors: tuple[str, ...] = tuple(SENSORS)
    categories: tuple[str, ...] = ()
    static_numeric: tuple[str, ...] = ()
    windows: tuple[int, ...] = (30,)
    image_mode: str = 'basic'
    grid: int = 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key in ('sensors', 'categories', 'static_numeric', 'windows'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def make_feature_config(frame, feature_set='expanded', sensor_set='all',
                        image_mode='spatial', windows=(10, 30, 60, 120)):
    """Train의 컬럼 존재 여부만 확인한다. Test 분포로 피처를 선택하지 않는다."""
    if feature_set == 'baseline':
        return FeatureConfig()
    if feature_set != 'expanded' or sensor_set not in ('basic', 'all'):
        raise ValueError('지원하지 않는 피처 설정입니다.')
    windows = tuple(sorted(set(windows)))
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError('윈도우는 양의 정수 초 단위로 지정하십시오.')
    if image_mode not in ('basic', 'spatial'):
        raise ValueError('image_mode는 basic 또는 spatial이어야 합니다.')
    candidates = SENSORS if sensor_set == 'basic' else SENSORS + EXTRA_SENSORS
    sensors = tuple(c for c in candidates if c in frame.columns)
    if not sensors:
        raise ValueError('사용 가능한 센서 컬럼이 없습니다.')
    return FeatureConfig(
        feature_set='expanded', sensors=sensors,
        categories=tuple(c for c in CATEGORIES if sensor_set == 'all' and c in frame),
        static_numeric=tuple(c for c in ['bat_cc'] if sensor_set == 'all' and c in frame),
        windows=windows, image_mode=image_mode,
    )


def validate_frame(frame, config=None):
    required = ['sample_id', 'experiment_id', 'relative_time_s', 'image_path']
    if config is not None:
        required += list(config.sensors + config.categories + config.static_numeric)
    if any(c not in frame.columns for c in required):
        raise ValueError('입력에 필수 컬럼이 없습니다. 저장된 피처 설정과 CSV 스키마를 확인하십시오.')
    if frame.empty:
        raise ValueError('입력 CSV가 비어 있습니다.')
    if frame['sample_id'].isna().any() or not frame['sample_id'].is_unique:
        raise ValueError('sample_id는 결측 없이 고유해야 합니다.')
    if frame['experiment_id'].isna().any() or frame['image_path'].isna().any():
        raise ValueError('experiment_id와 image_path에는 결측을 허용하지 않습니다.')
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('relative_time_s는 유한한 0 이상 숫자여야 합니다.')


def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    """행마다 경로를 연결한다. 파일명 fallback의 중복은 임의 선택하지 않는다."""
    roots = list(dict.fromkeys(Path(root).resolve() for root in roots))
    resolved = []
    missing = []
    for i, raw in enumerate(frame['image_path'].astype(str)):
        p = Path(raw)
        candidates = [p] if p.is_absolute() else [root / p for root in roots]
        found = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        resolved.append(found)
        if found is None:
            missing.append((i, p.name))
    if missing:
        wanted = {name for _, name in missing}
        index = {name: set() for name in wanted}
        for root in roots:
            for path in root.rglob('*'):
                if path.name in wanted and path.is_file():
                    index[path.name].add(path.resolve())
        for i, name in missing:
            if len(index[name]) != 1:
                raise ValueError('이미지 경로가 없거나 같은 파일명이 여러 곳에 있습니다. image_root를 확인하십시오.')
            resolved[i] = next(iter(index[name]))
    return pd.Series(resolved, index=frame.index, dtype=object)


def image_feature_names(config):
    names = list(IMAGE_STATS)
    if config.image_mode == 'spatial':
        names += [f'img_{channel}_{stat}' for channel in 'rgb' for stat in ('mean', 'std', 'p95')]
        names += [f'img_hue_bin{i}' for i in range(8)]
        names += [f'img_{channel}_{stat}' for channel in ('sat', 'value') for stat in ('mean', 'std')]
        names += [f'img_tile{y}_{x}_{channel}_{stat}'
                  for y in range(config.grid) for x in range(config.grid)
                  for channel in 'rgb' for stat in ('mean', 'std')]
    return names


def _image_stats(path, config=None) -> list[float]:
    config = config or FeatureConfig()
    with Image.open(path) as image:
        image = image.convert('RGB')
        gray = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
        values = [float(gray.mean()), float(gray.std()), float(gray.max()),
                  float(np.percentile(gray, 95)), float(np.percentile(gray, 99)),
                  float((gray >= HOT_THRESHOLD).mean())]
        if config.image_mode == 'spatial':
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            for channel in range(3):
                a = rgb[:, :, channel]
                values += [float(a.mean()), float(a.std()), float(np.percentile(a, 95))]
            hsv = np.asarray(image.convert('HSV'), dtype=np.float32) / 255.0
            hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 1))
            values += (hist / hist.sum()).tolist()
            for channel in (1, 2):
                a = hsv[:, :, channel]
                values += [float(a.mean()), float(a.std())]
            if min(rgb.shape[:2]) < config.grid:
                raise ValueError('이미지가 영역 분할 격자보다 작습니다.')
            for row in np.array_split(rgb, config.grid, axis=0):
                for tile in np.array_split(row, config.grid, axis=1):
                    for channel in range(3):
                        a = tile[:, :, channel]
                        values += [float(a.mean()), float(a.std())]
    return values


def extract_image_features(frame, paths, n_jobs=2, config=None, progress=False):
    config = config or FeatureConfig()
    paths = list(paths)
    rows = []
    with Parallel(n_jobs=n_jobs, prefer='threads', batch_size=32) as parallel:
        for start in range(0, len(paths), 512):
            batch = paths[start:start + 512]
            rows.extend(parallel(delayed(_image_stats)(p, config) for p in batch))
            if progress:
                percent = int(100 * (start + len(batch)) / len(paths))
                print(f'[progress] 이미지 피처 생성 {percent}%', flush=True)
    return pd.DataFrame(rows, columns=image_feature_names(config),
                        index=pd.Index(frame['sample_id'].values, name='sample_id'))


def numeric(frame, cols):
    return frame[list(cols)].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def add_causal_features(base, signals, window_s=WINDOW_S, *, windows=None, expanded=False):
    """(t-window, t]만 집계한다. 같은 시각 관측도 순서에 무관하게 포함한다.

    lag는 t-window 이하 마지막 관측 시각의 센서 평균이다. 차이는 현재값 기준이며,
    변화율의 분모는 그 관측까지의 실제 시간 차이다. 실험 간 상태를 공유하지 않는다.
    """
    windows = tuple(windows or (int(window_s),))
    parts = {}
    n = len(base)
    stats = ('mean', 'max', 'delta', 'std', 'min', 'range', 'rate', 'count') if expanded else ('mean', 'max', 'delta')
    for window in windows:
        for signal in signals:
            for stat in stats:
                parts[f'{signal}_{stat}{window}s'] = np.full(n, np.nan)
        if expanded:
            parts[f'lag_age{window}s'] = np.full(n, np.nan)
    if expanded:
        parts['previous_gap_s'] = np.full(n, np.nan)
    times = base['relative_time_s'].to_numpy(dtype=float)
    matrix = base[signals].to_numpy(dtype=float)
    for idx in base.groupby('experiment_id', sort=False).indices.values():
        if 'sample_id' in base:
            tie_break = base['sample_id'].astype(str).to_numpy()[idx]
            idx = idx[np.lexsort((tie_break, times[idx]))]
        else:
            idx = idx[np.argsort(times[idx], kind='mergesort')]
        t, x = times[idx], matrix[idx]
        obs = pd.DataFrame(x, index=pd.to_timedelta(t, unit='s'), columns=signals)
        equal_time_mean = obs.groupby(level=0, sort=False).transform('mean').to_numpy()
        # 같은 시각의 마지막 rolling 결과를 써서 모든 동시 관측을 포함한다.
        end = np.searchsorted(t, t, side='right') - 1
        if expanded:
            previous = np.searchsorted(t, t, side='left') - 1
            ok = previous >= 0
            gap = np.full(len(t), np.nan)
            gap[ok] = t[ok] - t[previous[ok]]
            parts['previous_gap_s'][idx] = gap
        for window in windows:
            rolling = obs.rolling(pd.Timedelta(seconds=window), closed='right', min_periods=1)
            mean = rolling.mean().to_numpy()[end]
            maximum = rolling.max().to_numpy()[end]
            lag = np.searchsorted(t, t - window, side='right') - 1
            valid = lag >= 0
            delta = np.full_like(x, np.nan)
            age = np.full(len(t), np.nan)
            delta[valid] = x[valid] - equal_time_mean[lag[valid]]
            age[valid] = t[valid] - t[lag[valid]]
            values = {'mean': mean, 'max': maximum, 'delta': delta}
            if expanded:
                minimum = rolling.min().to_numpy()[end]
                values.update(std=rolling.std(ddof=0).to_numpy()[end], min=minimum,
                              range=maximum - minimum, rate=delta / age[:, None],
                              count=rolling.count().to_numpy()[end])
                parts[f'lag_age{window}s'][idx] = age
            for j, signal in enumerate(signals):
                for stat in stats:
                    parts[f'{signal}_{stat}{window}s'][idx] = values[stat][:, j]
    return pd.DataFrame(parts, index=base.index)


def build_features(frame, image_features, config=None):
    config = config or FeatureConfig()
    MISSING = '__missing__'
    validate_frame(frame, config)
    work = frame.reset_index(drop=True)
    base = work[['sample_id', 'experiment_id']].copy()
    base['relative_time_s'] = pd.to_numeric(work['relative_time_s'])
    sensor_cols = list(config.sensors + config.static_numeric)
    base[sensor_cols] = numeric(work, sensor_cols)
    image_cols = image_feature_names(config)
    img = image_features.reindex(work['sample_id'].values)[image_cols]
    if not np.isfinite(img.to_numpy(dtype=float)).all():
        raise ValueError('이미지 피처가 누락되었거나 유한하지 않습니다.')
    base[image_cols] = img.to_numpy()
    expanded = config.feature_set == 'expanded'
    signals = list(config.sensors) + ['img_max', 'img_p99']
    causal = add_causal_features(base, signals, windows=config.windows, expanded=expanded)
    features = base[['relative_time_s'] + sensor_cols + image_cols].copy()
    if expanded:
        for left, right in [('meas_t1', 'meas_t2'), ('meas_t1', 'thermal_t'), ('meas_t2', 'thermal_t')]:
            if left in base and right in base:
                features[f'{left}_minus_{right}'] = base[left] - base[right]
    features = pd.concat([features, causal], axis=1)
    for col in config.categories:
        def category_value(value):
            if pd.isna(value):
                return MISSING
            try:
                return format(float(value), '.15g')
            except (ValueError, TypeError):
                return str(value)
        features[col] = work[col].map(category_value)
    features.index = frame.index
    return features


def class_weights_for(y, mode):
    """호출한 학습 fold의 라벨만 사용한다."""
    if mode == 'none':
        return None
    counts = pd.Series(y).value_counts()
    if mode not in ('balanced', 'sqrt_balanced'):
        raise ValueError('지원하지 않는 클래스 가중치입니다.')
    weights = len(y) / (len(counts) * counts)
    if mode == 'sqrt_balanced':
        weights = np.sqrt(weights)
    return {int(label): float(weight) for label, weight in weights.items()}


def build_model(class_weight=None, *, model_kind='logreg', feature_columns=None,
                categorical_columns=(), seed=SEED, iterations=300, depth=4,
                learning_rate=0.05, n_jobs=2, add_missing_indicators=True,
                catboost_task_type='CPU', catboost_devices='0',
                catboost_border_count=128, catboost_l2_reg=5.0, gpu_ram_part=0.25):
    columns = list(feature_columns or [])
    cats = list(categorical_columns)
    if model_kind == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError('CatBoost 선택 시 requirements-catboost.txt를 먼저 설치하십시오.') from exc
        options = dict(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            loss_function='MultiClass', class_weights=class_weight, cat_features=cats,
            random_seed=seed, thread_count=n_jobs, task_type=catboost_task_type,
            border_count=catboost_border_count, l2_leaf_reg=catboost_l2_reg,
            boosting_type='Plain', bootstrap_type='Bayesian', bagging_temperature=1.0,
            one_hot_max_size=2, nan_mode='Min',
            verbose=False, allow_writing_files=False,
        )
        if catboost_task_type == 'GPU':
            options.update(devices=catboost_devices, gpu_ram_part=gpu_ram_part)
        return CatBoostClassifier(**options)
    numeric_steps = [('imputer', SimpleImputer(strategy='median', add_indicator=add_missing_indicators,
                                             keep_empty_features=True))]
    if model_kind == 'logreg':
        numeric_steps.append(('scaler', StandardScaler()))
        classifier = LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000,
                                        class_weight=class_weight, random_state=seed)
    elif model_kind == 'hist':
        if isinstance(class_weight, dict):
            raise ValueError('hist의 수동 가중치는 fit의 classifier__sample_weight로 전달하십시오.')
        classifier = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=iterations, max_depth=depth,
            max_leaf_nodes=min(31, 2 ** depth), l2_regularization=5.0,
            early_stopping=False, class_weight=class_weight, random_state=seed,
        )
    else:
        raise ValueError('model_kind는 logreg, hist, catboost 중 하나여야 합니다.')
    if not columns:
        preprocess = Pipeline(numeric_steps)
    else:
        preprocess = ColumnTransformer([
            ('numeric', Pipeline(numeric_steps), [col for col in columns if col not in cats]),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
        ], sparse_threshold=0)
    return Pipeline([('preprocessor', preprocess), ('classifier', classifier)])
