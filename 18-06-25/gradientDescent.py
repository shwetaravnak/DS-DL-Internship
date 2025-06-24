import numpy as np
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, input_data, input_label, \
                 b0=0, b1=0):
        self.b0, self.b1 = 0, 0
        self.X = input_data
        self.y = input_label

    def fit (self, X, y):
        X_mean = np.mean (X)
        y_mean = np.mean (y)
        ssxy, ssx = 0, 0
        for _ in range (len (self.X)):
            ssxy += (self.X[_]-X_mean)*(self.y[_]-y_mean)
            ssx += (self.X[_]-X_mean)**2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - (self.b1 * X_mean)
        return self.b0, self.b1
    
    def predict (self, Xi):
        self.y_hat = self.b0 + (self.b1 * Xi)
        self.y_hat = np.squeeze (self.y_hat)
        return self.y_hat
    
    def mean_squared_error(self):
        error = self.y - self.y_hat
        squared_error = error ** 2
        return np.mean (squared_error)

    def gradientDescent (self, alpha=0.00005, epochs=1):
        error = self.y - self.y_hat
        n = len (self.X)
        for _ in range (epochs):
            del_b1 = (-2/n) * np.sum(self.X * error)
            del_b0 = (-2/n) * np.sum(error)
            self.b1 -= alpha * del_b1
            self.b0 -= alpha * del_b0
            print (f'Epoch : {_+1}st | B1 : {self.b1} \
                  | B0 : {self.b0}')
            return self.b0, self.b1


if __name__ == '__main__':
    heights = np.array ([
        [160], [171], [182], [180], [154]
    ])

    weights = np.array ([
        72, 76, 77, 83, 76
    ])

    # print (f'The shape of X : {heights.shape} \
        #    and the shape of y : {weights.shape}')

    lr = LinearRegression (input_data=heights, input_label=weights)
    b0, b1 = lr.fit (X=heights, y=weights)
    print (f'The value of intercept : {b0} \
           The value of slope : {b1}')
    
    # Xi = [[176]]
    y_hat = lr.predict (Xi=heights)
    print (f'True Label : {weights}')
    print (f'Predicted Label : {y_hat}')
    
    loss = lr.mean_squared_error ()
    print(f'Earlier loss : {loss}')

    newb0, newb1 = lr.gradientDescent (epochs=10)
    print(newb0, newb1)

    new_y_hat = lr.predict (heights)
    print (f'New Predictions : {new_y_hat}')
    




    # Sklearn Model
    # model = LinearRegression ()
    # model.fit (heights, weights)
    # y_pred = model.predict (Xi)
    # print (y_pred)