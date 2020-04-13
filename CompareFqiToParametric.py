import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import load
import time
import domain
import Section5 as s5


# ---------------- Compare method FQI and Q-Learning with Function Approximators ---------------------
def compare_algorithms(models_list, models_name, colors, error_threshold=0.1):
    """
    Plot the expected return for the differents algorithms model and the time consumed
    """
    assert len(models_list) == len(models_name)

    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # set of states X used to have an approximation of J
    p_values = [random.uniform(-1, 1) for i in range(15)]
    s_values = [random.uniform(-3, 3) for i in range(15)]

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

                # add the expected return and value of J
                expected_return_over_X.append(expected_return)
                model_time.append(end-start)

        # we store the average of the values for all the states
        j.append(np.mean(expected_return_over_X))
        times.append(np.mean(model_time))

    # plot the expected returns
    plt.subplot(2, 1, 1)
    plt.bar(range(len(models_list)), j, color=colors)
    plt.xticks(range(len(models_list)), models_name)
    plt.ylabel('$J^{\hat{\mu_{N}^{*}}}$', rotation=0)
    plt.title('Expected return over infinite time horizon for each model')

    # Plot the computation times
    plt.subplot(2, 1, 2)
    plt.bar(range(len(models_list)), times, color=colors)
    plt.xticks(range(len(models_list)), models_name)
    plt.ylabel('t', rotation=0)
    plt.title('Computation time for each model')
    plt.show()

    return j, times


if __name__ == '__main__':
    models_name = ['NN', 'LR', 'ET', 'NN', 'RBF']
    models_path = ['fqi_models/neural_net_second_1.joblib', 'fqi_models/regression_second_1.joblib',
                   'fqi_models/tree_second_1.joblib', 'parametric_models/NeuralNet.joblib', 'parametric_models/RBFN.joblib']
    colors = ['blue', 'blue', 'blue', 'red', 'red']  # FQI = blue  |  Parametric Q-learning = red

    models = []
    for name in models_path:
        if name[:3] == 'fqi':
            models.append(load(name)[-1])
        else:
            models.append(load(name))

    j, times = compare_algorithms(models, models_name, colors)
    print(times)
