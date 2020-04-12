import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import domain
from joblib import load
import time
import section5 as s5


def compare_algorithms(models_list, models_name, error_threshold=0.1):
    """
    Plot the expected return for the differents algorithms model
    """
    assert len(models_list) == len(models_name)

    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # set of states X used to have an approximation of J
    p_values = [random.uniform(-1, 1) for i in range(1)]
    s_values = [random.uniform(-3, 3) for i in range(1)]

    # values of J for each model
    j = []

    # computation time for each model
    times = []

    for idx, model in enumerate(models_list):
        print('Computing {}'.format(models_name[idx]))
        expected_return_over_X = []
        model_time = []

        # we compute the expected return for a certain number of states, and take the average
        # see 'Tree-Based Batch Mode Reinforcement Learning' by D. Ernst (figure 4. p519)
        for p in p_values:
            for s in s_values:
                start = time.time()
                expected_return = s5.compute_J((p, s), model, N)
                end = time.time()

                expected_return_over_X.append(expected_return)
                model_time.append(end-start)

        j.append(np.mean(expected_return_over_X))
        times.append(np.mean(model_time))

    # plot the expected returns
    plt.subplot(2, 1, 1)
    plt.bar(range(len(models_list)), j)
    plt.xticks(range(len(models_list)), models_name)
    plt.ylabel('$J^{\hat{\mu_{N}^{*}}}$', rotation=0)
    plt.title('Expected return over infinite time horizon for each model')

    plt.subplot(2, 1, 2)
    plt.bar(range(len(models_list)), times)
    plt.xticks(range(len(models_list)), models_name)
    plt.ylabel('t', rotation=0)
    plt.title('Computation time for each model')

    plt.show()

    return j, times


if __name__ == '__main__':
    models_name = ['Neural Network', 'Linear Regression', 'Extra Tree']
    models_path = ['models/neural_net_first_2.joblib', 'models/regression_first_2.joblib', 'models/tree_first_2.joblib']

    models = []
    for name in models_path:
        if name[:6] == 'models':
            models.append(load(name)[-1])
        else:
            models.append(load(name))

    j, times = compare_algorithms(models, models_name)
    print(times)
