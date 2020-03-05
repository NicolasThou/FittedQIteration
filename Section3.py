from matplotlib import pyplot as plt
import numpy as np
import Section2 as s2
import Bilel


def j_n(initial_state, policy, N):  # policy is a function approximator
    """
    recurrence state-value function
    """
    if N < 0:
        print("N has to be greater or equal than 0 !")
    elif N == 0:
        return 0
    else:
        action = policy(initial_state)
        next_state = s2.f(initial_state, action)
        return s2.r(initial_state, action) + s2.gamma*j_n(next_state, policy, N-1)


def monte_carlo_simulation(policy, number_of_sample, error_threshold):
    """
    Compute simulation, more faster than the function expected_return
    """
    # N threshold computation using J function bound
    a = (error_threshold * ((1 - s2.gamma) ** 2)) / 2
    
    # for N >= n, J_N is a good approximation of J
    n = int(np.ceil(np.log(a) / np.log(s2.gamma)))

    result = []
    for i in range(number_of_sample):
        state = s2.initial_state()
        result.append(j_n(state, policy, n))

    return np.mean(result)


def multiple_simulations(policy, min, max, error_threshold):
    """
    Computes and display the difference of the state-value function between two simulations
    using different number of samples from min to max
    """
    # store the values of j
    j = []

    # store the difference between two consecutive simulations
    diff = []

    # first simulation
    previous_j = monte_carlo_simulation(policy, min, error_threshold)

    sample = range(min+10, max, 10)
    for n in sample:
        print("Monte-Carlo simulation using " + str(n) + " samples")

        # computes Monte-carlo simulation
        new_j = monte_carlo_simulation(policy, n, error_threshold)
        j.append(new_j)

        # computes difference with previous simulation
        diff.append(abs(previous_j - new_j))
        previous_j = new_j

    # display the evolution of the difference along with the number of simulations
    plt.xlabel('number of simulations')
    plt.title(str(max))
    plt.plot(sample, diff)
    plt.show()


if __name__ == '__main__':
    multiple_simulations(s2.random_policy, 100, 250, 0.01)
