"""Prespecified independent algorithms with the original fold-fitted preprocessing."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

import baseline_model as M

XGB_PARAMS=dict(n_estimators=300,max_depth=4,max_leaves=16,learning_rate=.05,
                reg_lambda=5.,reg_alpha=0.,min_child_weight=1.,subsample=1.,colsample_bytree=1.,
                objective='multi:softprob',num_class=6,eval_metric='mlogloss',
                tree_method='hist',device='cpu',n_jobs=2,random_state=42,verbosity=0)
ET_PARAMS=dict(n_estimators=300,max_depth=18,min_samples_leaf=8,max_features='sqrt',
               random_state=42,n_jobs=2,class_weight=None)


class SixStageXGB(ClassifierMixin,BaseEstimator):
    def __init__(self,iterations=300,device='cpu'):
        self.iterations=iterations;self.device=device

    def fit(self,x,y):
        from xgboost import XGBClassifier
        params=dict(XGB_PARAMS,n_estimators=self.iterations,device=self.device)
        self.model_=XGBClassifier(**params).fit(x,np.asarray(y,dtype=int)-1)
        np.testing.assert_array_equal(self.model_.classes_,np.arange(6))
        self.classes_=np.arange(1,7)
        return self

    def predict_proba(self,x):
        return self.model_.predict_proba(x)

    def predict(self,x):
        return self.predict_proba(x).argmax(-1)+1


def build_model(columns,arm,iterations=300,device='cpu'):
    pipe=M.build_model(model_kind='hist',feature_columns=columns,categorical_columns=M.CATEGORIES,
                       seed=42,iterations=iterations,depth=4,learning_rate=.05,add_missing_indicators=True)
    if arm=='X1':pipe.set_params(classifier=SixStageXGB(iterations=iterations,device=device))
    elif arm=='T1':pipe.set_params(classifier=ExtraTreesClassifier(**dict(ET_PARAMS,n_estimators=iterations)))
    elif arm not in ('H0','I1','I2'):raise ValueError(arm)
    return pipe
