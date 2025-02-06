
from collections import OrderedDict
from vertex_voyage.model import BaseModel
import numpy as np 
from hashlib import sha256
import json 
from vertex_voyage.data import NewColumns

class LinearRegressionModel(BaseModel):
    def __init__(self, sources=None, target=None):
        super().__init__()
        self.sources = sources
        self.target = target
        if sources is None:
            self.sources = []

    def valid(self):
        return len(self.sources) > 0 and self.target is not None

    def fit(self, data):
        X = data[self.sources]
        y = data[self.target]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        

    def ready(self):
        return hasattr(self, 'coef_')

    def run(self, data):
        X = data[self.sources]
        data[self.target] = X @ self.coef_
        return data
    
    def produces(self):
        return NewColumns(columns=[self.target])

    def key(self):
        str_repr = OrderedDict({
            'type': 'LinearRegressionModel',
            'sources': self.sources,
            'target': self.target
        })
        str_repr = json.dumps(str_repr, sort_keys=True)
        return sha256(str(str_repr).encode()).hexdigest()
    
    def get_data(self):
        return {
            'sources': self.sources,
            'target': self.target
        }

    def add_source(self, source: str):
        return LinearRegressionModel(sources=self.sources + [source], target=self.target)
    
    def set_target(self, target: str):
        return LinearRegressionModel(sources=self.sources, target=target)