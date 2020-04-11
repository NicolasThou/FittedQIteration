import numpy as np
import torch.nn
import math


k_clusters = 10
sigma = 0.5
beta = 1/(2*(sigma**2))


class RBFLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RBFLayer, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        # apply weight
        q = self.linear(x)

        # normalisation
        q = q/torch.sum(x)

        return q


class RBFNet():
    def __init__(self, k_centers):
        self.centers = k_centers
        self.nb_in = len(k_centers)
        self.layer = RBFLayer(self.nb_in, 1)

    def fit(self, X, Y, batch_size=None, epochs=None):
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
        pass

    def predict(self, X):
        # go through the centers
        X = preprocess(X, self.centers)

        # apply the weights with normalisation
        return self.layer(X)


def euclidian_distance(a, b):
    """
    Compute the Euclidian distance between two points
    """
    # vectors must be equal size !
    assert len(a) == len(b)
    res = 0
    for i, j in list(zip(a, b)):
        res += math.sqrt((i-j)**2)

    return res


def kernel(x, mu):
    """
    Apply the Gaussian kernel to an input
    """
    diff = euclidian_distance(x, mu)
    return math.exp(-beta*(diff**2))


def preprocess(X, clusters_centers):
    """
    Apply the radial basis function to all inputs
    """
    inputs = []
    for x in X:
        processed_input = []
        for center in clusters_centers:
            # computes the Gaussian kernel for one center
            processed_input.append((kernel(x, center)))

        # add the processed input
        inputs.append(processed_input)

    return torch.tensor(inputs, dtype=torch.float32)


if __name__ == '__main__':
    # model = RBFNet(10, 1)
    # criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #
    # X = [[1.4, 3.4, 4, 6, 2.8, 1.4, 2.4, 4.7, 56.8, 7],
    #      [3, 8.4, 6.7, 6.8, 5.6, 1.4, 45.3, 23.4, 9, 6.3],
    #      [9, 3.4, 3.2, 6, 5.2, 1.7, 3.4, 4, 2, 7]]
    # Y = [[3], [5], [6]]
    #
    # for x, y in zip(X, Y):
    #     x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    #     y = torch.tensor(y, dtype=torch.float32)
    #
    #     y_pred = model(x)
    #     loss = criterion(y_pred, y)
    #     print(loss.item())
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    k_centers = [[1, 2], [1, 4], [3, 4], [4, 1], [2, 1], [5, 2]]
    X_train = [[1, 2], [1, 4], [3, 4], [4, 1], [2, 1], [5, 2]]
    Y_train = [0, 0, 0, 1, 1, 1]
    model = RBFNet(k_centers)

    x = [[2, 7]]
    print(model.predict(x))


