import torch.nn
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import euclidean
import Section6 as s6


class RBFLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RBFLayer, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x, normalisation=False):
        # apply weight
        q = self.linear(x)

        # add constant bias
        q += 1

        # normalise output
        if normalisation:
            q = q/torch.sum(x)

        return q


class RBFNet:
    def __init__(self, k_centers):
        self.layer = RBFLayer(k_centers, 1)

        # store the centers
        self.centers = None

        # beta = 1 / (2 * (sigma ** 2))
        self.beta = None

    def fit(self, X, Y, epochs=50):
        """
        Train the RBFNet model
        :return : the loss list
        """
        # compute the centers
        self.centers = extract_centers(X, self.layer.linear.in_features)

        # compute beta
        sigma = compute_sigma(self.centers)
        self.beta = 1 / (2 * (sigma ** 2))

        # loss function and optimizer
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.01)

        error = []

        # train the network
        for e in range(epochs):
            l = []

            for x, y in zip(X, Y):
                # convert the input/output to tensors
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor([y], dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # model make rpediction
                y_pred = self.predict(x, normalisation=False)

                # computes the loss function
                loss = criterion(y_pred, y)
                l.append(loss.item())

                # computes the backpropagation
                loss.backward()
                optimizer.step()

            error.append(np.mean(l))
            print('epoch {}  |  loss : {}'.format(e, np.mean(l)))

        return error

    def predict(self, X, normalisation=False):
        """
        Make prediction for a set of samples
        """
        # go through the centers
        X = preprocess(X, self.centers, self.beta)

        # apply the weights with normalisation
        out = self.layer(X, normalisation=normalisation)

        return out


def kernel(x, mu, beta):
    """
    Apply the Gaussian kernel to an input
    """
    d = euclidean(np.array(x), np.array(mu))

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
    Compute the sigma using the centers : sigma = d_max(sqrt(2M)
                                            where   dmax = larger distance between two centers
                                                    M = number of cluster
        (see 'Radial basis functions: normalised or lln-norn1alised?' by M.R. Cowper)
    """
    distances = []
    for idx, center in enumerate(k_centers[:-1]):
        # compare this center with all the next centers in the list
        for center2 in k_centers[idx+1:]:
            distances.append(euclidean(np.array(center), np.array(center2)))

    d_max = np.max(distances)

    return d_max/math.sqrt(2*len(k_centers))


def extract_centers(X, k):
    # run k-mean algorithm on dataset
    centers = kmeans2(np.array(X).astype(float), k, minit='points')[0].tolist()

    # reshape the list to be 2d array
    centers = np.reshape(centers, (-1, 1)).tolist()

    return centers


if __name__ == '__main__':
    torch.device("cuda:0")

    # TRAIN
    NUM_SAMPLES = 500
    X_train = np.random.uniform(0., 1., NUM_SAMPLES)
    X_train = np.sort(X_train, axis=0)
    noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
    y_train = np.sin(2 * np.pi * X_train) + noise
    X_train = np.reshape(X_train, (-1, 1))

    # FIT
    model = RBFNet(k_centers=5)
    loss = model.fit(X_train, y_train, epochs=100)
    plt.plot(range(len(loss)), loss)
    plt.show()

    # TEST
    X_test = np.random.uniform(0., 1., 300)
    X_test = np.sort(X_test, axis=0)
    X_test = np.reshape(X_test, (-1, 1))
    y_test = []
    y = []
    for x in X_test:
        y_test.append(model.predict(x))
        y.append(np.sin(2 * np.pi * x))

    plt.subplots()
    plt.plot(X_test, y, color='red')
    plt.plot(X_test, y_test, color='blue', linewidth=3)
    plt.show()
