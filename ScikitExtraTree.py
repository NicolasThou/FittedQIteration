import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class SkExtraTree(ExtraTreesRegressor):
    def __init__(self, n_trees, n_min):
        super().__init__(n_estimators=n_trees, min_samples_split=n_min)

    def __call__(self, X):
        return super().predict(X)


if __name__ == '__main__':
    X = [[2, 3], [4, 5], [2, 5]]
    Y = [2, 5, 4]

    t = SkExtraTree(50, 2)
    t.fit(X, Y)

    x = [[4, 6], [2, 7]]

    print(t(x))
