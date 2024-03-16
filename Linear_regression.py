import numpy as np 

class Linear_regression():
    def __init__(self, x, y, lr, max_epochs=400):
        self.x = x
        self.y = y
        self.lr = lr
        self.max_epochs = max_epochs
        self.w = 0
        self.b = 0
        self.best_loss = float('inf')  # Initialize best_loss to infinity
        self.best_params = None 

    def gradient_descent(self):
        dldw = 0.0
        dldb = 0.0
        N = len(self.x)
        for i, j in zip(self.x, self.y):
            dldw += -2 * i * (j - (self.w * i + self.b))
            dldb += -2 * (j - self.w * i + self.b)
        self.w -= self.lr * (1 / N) * dldw
        self.b -= self.lr * (1 / N) * dldb

    def fit(self):
        for epoch in range(self.max_epochs):
            self.gradient_descent()
            yhat = self.w * self.x + self.b
            loss = np.divide(np.sum((yhat - self.y) ** 2, axis=0), self.x.shape[0])
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = (self.w, self.b)
            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        self.w, self.b = self.best_params  # Update parameters with the best found parameters
        print(f'Best params - w = {self.w}, b = {self.b}, Loss: {self.best_loss}')
    def pridect(self,x):
        return self.w*x+self.b
            
