from sklearn.ensemble import ExtraTreesRegressor
import numpy as np


if __name__ == '__main__':
    X = [[2, 3], [4, 5], [2, 5]]
    Y = [2, 5, 4]

    t = ExtraTreesRegressor(n_estimators=50, min_samples_split=2)
    t.fit(X, Y, 2)

    x = [[4, 6], [2, 7]]

    print(t.predict(x))


