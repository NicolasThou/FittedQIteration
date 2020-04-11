import numpy as np
import torch.nn
import math
import random
import matplotlib.pyplot as plt


class RBFLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RBFLayer, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        # apply weight
        q = self.linear(x)

        # add constant bias
        q += 1

        # normalisation
        q = q/torch.sum(x)

        return q


class RBFNet():
    def __init__(self, kMeans):
        self.nb_in = kMeans
        self.centers = None
        self.layer = RBFLayer(self.nb_in, 1)
        self.beta = None

    def fit(self, X, Y, epochs=50):
        """
        Train the RBFNet model
        :return : the loss list
        """
        self.centers = extract_k_centers(X, self.nb_in)


        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.01, momentum=0.9, weight_decay=0.9)

        losses = []
        for e in range(epochs):
            l = 0
            for x, y in zip(X, Y):
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor([y], dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # model make rpediction
                y_pred = self.predict(x.tolist())

                # computes the loss function
                loss = criterion(y_pred, y)
                l += loss.item()

                # computes the backpropagation
                loss.backward()
                optimizer.step()
                print('y_true = {}  |  y_pred = {}'.format(y, y_pred))
                print(self.layer.linear.weight)

            l /= len(X)
            losses.append(l)
            print('epoch {}  |  loss : {}'.format(e, l))

        return losses

    def predict(self, X):
        """
        Make prediction for a set of samples
        """
        # go through the centers
        X = preprocess(X, self.centers, self.beta)

        # apply the weights with normalisation
        out = self.layer(X)

        return out


def extract_k_centers(X, k):
    pass


def euclidian_distance(u, v):
    """
    Compute the Euclidian distance between two points
    """
    # vectors must be equal size !
    assert len(u) == len(v)

    res = 0
    for i in range(len(u)):
        res += (u[i]-v[i])**2

    return math.sqrt(res)


def kernel(x, mu, beta):
    """
    Apply the Gaussian kernel to an input
    """
    d = euclidian_distance(x, mu)
    return math.exp(-beta*(d**2))


def preprocess(X, clusters_centers, beta):
    """
    Apply the radial basis function to all inputs
    """

    processed_input = []
    for center in clusters_centers:
        # computes the Gaussian kernel for one center (e.g. one node of the RBFNet)
        phi = kernel(X, center, beta)
        processed_input.append(phi)

    return torch.tensor(processed_input, dtype=torch.float32)


def compute_sigma(k_centers):
    """
    Compute the sigma using the centers
        (see 'Radial basis functions: normalised or lln-norn1alised?' by M.R. Cowper)
    """
    d_max = 0
    for center in k_centers:
        for center2 in k_centers:
            d = euclidian_distance(center, center2)
            d_max = d if d > d_max else d_max

    return d_max/math.sqrt(2*len(k_centers))


if __name__ == '__main__':
    k_clusters = 10
    epochs = 50

    # X_train = []
    # y_train = []
    # for onsenfou in range(50):
    #     a, b = random.randint(0, 10), random.randint(0, 10)
    #     X_train.append([a, b])
    #     if a < b:
    #         y_train.append(-1)
    #     else:
    #         y_train.append(1)
    #
    # print(X_train)
    # print(y_train)

    # Dataset
    NUM_SAMPLES = 100
    X_train = np.random.uniform(0., 1., NUM_SAMPLES)
    X_train = np.sort(X_train, axis=0)
    noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
    y_train = np.sin(2 * np.pi * X_train) + noise


    sigma = compute_sigma(k_centers)
    beta = 1 / (2 * (sigma ** 2))

    model = RBFNet(k_centers, beta)
    loss = model.fit(X_train, y_train, epochs=epochs)

    plt.plot(range(epochs), loss)
    plt.show()

