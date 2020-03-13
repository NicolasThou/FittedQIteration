from sklearn.linear_model import LinearRegression
import numpy as np


if __name__ == '__main__':
    X = [[1, 2, 3], [3, 4, 5], [2, 4, 5]]
    Y = [6, 12, 11]

    reg = LinearRegression().fit(X, Y)

    print(reg.predict([[2, 7, 9]]))

