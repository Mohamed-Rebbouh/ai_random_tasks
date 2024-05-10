import numpy as np 
from collections import Counter

class KNN:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, x, y):
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        if self.metric not in ['euclidean', 'manhattan']:
            raise ValueError(f"Metric {self.metric} does not exist")
    
    def _distance(self, x1, x2):
      if self.metric == 'euclidean':
        distance = np.sqrt(np.sum((x1 - x2)**2))
      else:
        distance = np.sum(np.abs(x1 - x2))
      return distance
    
    def _predict(self, x):
      
        distances = [self._distance(x, data) for data in self.x_data]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_data[i] for i in k_indices]

        
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
    def predict(self, X):
      X=np.array(X)
      pred = [self._predict(x) for x in X]
      return pred
