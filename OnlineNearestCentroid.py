import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures

class OnlineNearestCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.centroids = {}
        self.class_counts = {}
        self.transform = lambda x: x#PolynomialFeatures(degree=2, include_bias=True)

    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) if self.num_classes is None else self.num_classes
        X = self.transform.fit_transform(X)
        for label in np.unique(y):
            self.centroids[label] = np.mean(X[y == label], axis=0)
            self.class_counts[label] = np.sum(y == label)
        
        return self

    def predict(self, X):
        return np.array([self._nearest_centroid(x) for x in X])

    @property
    def is_fitted(self):
        return bool(self.centroids)

    def predict_proba(self, X):
        # if not self.centroids:
        #     raise ValueError("Model not trained. Call fit() before making predictions.")
        X = self.transform.fit_transform(X)
        
        if self.centroids:
            n_datapoints = X.shape[0]
            proba = np.full((n_datapoints, self.num_classes), -1000.0)

            for label, centroid in self.centroids.items():
                distances = np.linalg.norm(X - centroid, axis=1)
                proba[:, label] = -distances
        else:
            proba = np.ones((X.shape[0], self.num_classes)) # uniform distribution

        proba -= np.max(proba, axis=1, keepdims=True)  # For numerical stability
        proba = np.exp(proba)
        proba /= np.sum(proba, axis=1, keepdims=True)

        return proba

    def _nearest_centroid(self, x):
        distances = {label: np.linalg.norm(x - centroid) for label, centroid in self.centroids.items()}
        return min(distances, key=distances.get)

    def partial_fit(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        X = self.transform.fit_transform(X)
        for label in classes:
            if label in self.centroids:
                alpha = 1.0 / (self.class_counts[label] + 1.0)
                self.centroids[label] = (1 - alpha) * self.centroids[label] + alpha * np.mean(X[y == label], axis=0)
                self.class_counts[label] += 1
                
            else:
                
                self.centroids[label] = np.mean(X[y == label], axis=0)
                self.class_counts[label] = np.sum(y == label)

        self.num_classes = max(self.num_classes, np.max(y) + 1)
        
        return self

    @classmethod
    def merge(cls, model1, model2):
        merged_model = cls(num_classes=max(model1.num_classes, model2.num_classes))

        all_classes = set(model1.centroids.keys()) | set(model2.centroids.keys())
        for label in all_classes:
            if label in model1.centroids and label in model2.centroids:
                merged_model.centroids[label] = (model1.centroids[label]*model1.class_counts.get(label, 0) + model2.centroids[label]*model2.class_counts.get(label, 0)) /(model1.class_counts.get(label, 0)+model2.class_counts.get(label, 0))
            elif label in model1.centroids:
                merged_model.centroids[label] = model1.centroids[label]
            elif label in model2.centroids:
                merged_model.centroids[label] = model2.centroids[label]
            merged_model.class_counts[label] = model1.class_counts.get(label, 0) + model2.class_counts.get(label, 0)
        
        return merged_model

    def __repr__(self):

        c = [0]*self.num_classes
        for label in self.class_counts:
            c[label] = self.class_counts[label]

        return f"OnlineNearestCentroid(class_counts={c})"

    def __format__(self, __format_spec: str) -> str:
        return self.__repr__()