import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import domain
from joblib import load
import section5 as s5


def compare_algorithms(models_list, models_name, error_threshold=0.1):
    """
    Plot the expected return for the differents algorithms model
    """
    assert len(models_list) == len(models_name)

    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # set of states X used to have an approximation of J
    p_values = [random.uniform(-1, 1) for i in range(5)]
    s_values = [random.uniform(-3, 3) for i in range(5)]

    # values of J for each model
    j = []

    for idx, model in enumerate(models_list):
        print('Computing {}'.format(models_name[idx]))
        expected_return_over_X = []

        # we compute the expected return for a certain number of states, and take the average
        # see 'Tree-Based Batch Mode Reinforcement Learning' by D. Ernst (figure 4. p519)
        for p in p_values:
            for s in s_values:
                expected_return = s5.compute_J((p, s), model, N)
                expected_return_over_X.append(expected_return)

        j.append(np.mean(expected_return_over_X))

    # plot the expected returns
    plt.bar(range(len(models_list)), j)
    plt.xticks(range(len(models_list)), models_name)
    plt.ylabel('$J^{\hat{\mu_{N}^{*}}}$', rotation=0)
    plt.show()


if __name__ == '__main__':
    torch.device('cuda:0')

    models_name = ['Neural Network', 'Linear Regression', 'Extra Tree']
    models_path = ['models/neural_net_first_2.joblib', 'models/regression_first_2.joblib', 'models/tree_first_2.joblib']

    models = []
    for name in models_path:
        if name[:6] == 'models':
            models.append(load(name)[-1])
        else:
            models.append(load(name))

    compare_algorithms(models, models_name)
