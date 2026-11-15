"""Fixed E1 inference: original Train78 #20 and multirate XGB, equal probability weight."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import model as S20


class Predictor:
    def __init__(self, weights):
        self.weights = Path(weights)
        self.s20 = joblib.load(self.weights/'s20/hgb.joblib')
        self.mr = joblib.load(self.weights/'mrxgb/model.joblib')
        if json.loads(json.dumps(self.s20['feature_config'])) != json.loads(json.dumps(self.mr['feature_config'])):
            raise ValueError('The two preserved feature configurations differ')
        if self.s20['original_features'] != self.mr['features']:
            raise ValueError('The preserved base feature schemas differ')
        self.config = S20.H.FeatureConfig.from_dict(self.s20['feature_config'])
        for bundle in (self.s20,self.mr):
            if not np.array_equal(bundle['pipeline'].classes_,np.arange(1,7)):
                raise ValueError('Class order must be exactly1 through6')

    def predict(self, frame, image_root):
        frame = frame.reset_index(drop=True)
        S20.H.validate_frame(frame,self.config)
        image_root = Path(image_root)
        paths = S20.H.resolve_image_paths(frame,[image_root,image_root.parent])
        images = S20.H.extract_image_features(frame,paths,n_jobs=2,config=self.config)
        x = S20.H.build_features(frame,images,self.config)
        with threadpool_limits(limits=2):
            p20 = S20.route(S20.predict_matrix(self.s20,frame,x),
                            S20.conditional(frame,self.weights/'s20'))
            pmr = self.mr['pipeline'].predict_proba(x[self.mr['features']])
        p = .5*p20+.5*pmr
        if p.shape != (len(frame),6) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=1e-5):
            raise ValueError('Invalid blended probabilities')
        return p
