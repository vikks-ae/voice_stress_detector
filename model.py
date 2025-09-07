from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np

class StressMoodModel:
    def __init__(self, clustering_method='kmeans'):
        self.method = clustering_method
        self.model = None

    def fit(self, features: np.ndarray):
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=2).fit(features)
        elif self.method == 'isolation_forest':
            self.model = IsolationForest().fit(features)
    
    def predict(self, new_feature: np.ndarray):
        if self.method == 'kmeans':
            label = self.model.predict([new_feature])
            return label
        elif self.method == 'isolation_forest':
            label = self.model.predict([new_feature])
            return label  # -1: anomaly (possibly stressed), 1: normal
