from matplotlib import pyplot as plt
import numpy as np
import Section2 as s2


def j_n(initial_state, policy, N):
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
        print('simulation ' + str(i+1))
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

    previous_j = 0

    # save the numbers of sample to plot a figure
    sample = range(min, max+10, 10)

    for n in sample:
        print("Monte-Carlo simulation using " + str(n) + " samples")

        # computes Monte-carlo simulation
        new_j = monte_carlo_simulation(policy, n, error_threshold)
        j.append(new_j)

        # computes difference with previous simulation
        diff.append(abs(previous_j - new_j))
        previous_j = new_j

    # delete the first elements
    diff = diff[1:]
    sample = sample[1:]

    # display the evolution of the difference along with the number of simulations
    plt.xlabel('number of simulations')
    plt.title("Difference of consecutive simulations expected return")
    plt.plot(sample, diff)
    plt.show()


if __name__ == '__main__':
    expected = monte_carlo_simulation(s2.random_policy, 90, 0.01)
    print(expected)
