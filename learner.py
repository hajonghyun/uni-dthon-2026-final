"""Same XGBoost parameters in both arms; fit on CUDA, portable CPU prediction."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from xgboost import XGBClassifier

from models import XGB_PARAMS


class StageXGB(ClassifierMixin,BaseEstimator):
    def __init__(self,iterations=300,device='cuda'):
        self.iterations=iterations;self.device=device

    def fit(self,x,y,sample_weight=None):
        self.model_=XGBClassifier(**dict(XGB_PARAMS,n_estimators=self.iterations,device=self.device))
        self.model_.fit(x,np.asarray(y,int)-1,sample_weight=sample_weight)
        if not np.array_equal(self.model_.classes_,np.arange(6)):raise ValueError('Class alignment failed')
        self.classes_=np.arange(1,7)
        # The fitted trees are portable; use the CPU predictor for NumPy inputs.
        self.model_.set_params(device='cpu')
        return self

    def predict_proba(self,x):return self.model_.predict_proba(x)

    def predict(self,x):return self.predict_proba(x).argmax(-1)+1
