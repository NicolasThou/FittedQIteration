import numpy as np
from sklearn.linear_model import LinearRegression


class SkLinearRegression(LinearRegression):
    def __call__(self, X):
        return super().predict(X)


if __name__ == '__main__':
    X = [[1, 2, 3], [3, 4, 5], [2, 4, 5]]
    Y = [6, 12, 11]

    reg = SkLinearRegression()
    reg.fit(X, Y)

    print(reg([[2, 7, 9]]))

