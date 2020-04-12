import numpy as np
import matplotlib.pyplot as plt
import random
import domain
import section5 as s5


def comapre_algorithms(models_list, models_name, error_threshold=0.1):
    """
    Plot the expected return for the differents algorithms model
    """
    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # set of states X used to have an approximation of J
    p_values = [round(random.uniform(-1, 1), 2) for i in range(5)]
    s_values = [round(random.uniform(-3, 3), 2) for i in range(5)]

    # values of J for each model
    j = []

    for model in models_list:
        expected_return_over_X = []

        # we compute the expected return for a certain number of states, and take the average
        # see 'Tree-Based Batch Mode Reinforcement Learning' by D. Ernst (figure 4. p519)
        for p in p_values:
            for s in s_values:
                expected_return = s5.compute_J((p, s), model, N)

                expected_return_over_X.append(expected_return)

        j.append(np.mean(expected_return_over_X))

    # plot the expected returns
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(models_name, j)
    plt.xlabel('models')
    plt.ylabel('$J^{\hat{\mu_{N}^{*}}}$', rotation=0)
    plt.show()


