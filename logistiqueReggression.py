import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, Max_iterations=1000):
        self.learning_rate = learning_rate
        self.Max_iterations = Max_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.Max_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        return np.round(self.sigmoid(z))
