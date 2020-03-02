from section1 import *
import numpy as np


def j_n(initial_state, policy, N):  # policy is a function approximator
    """
    state-value function, iterative version
    """
    state = initial_state
    j = 0
    for i in range(N):
        action = policy(state)
        j += (gamma**i)*r(state, action)
        state = f(state, action)

    return j


def expected_return(policy, nb_simulation, error_threshold):
    """
    Return an approximation of the expected return of a policy
    over the infinite time horizon using Monte-Carlo simulation
    """
    # N threshold computation using J function bound
    a = (error_threshold * ((1 - gamma)**2))/2  # TODO make a comment here
    n = int(np.ceil(np.log(a)/np.log(gamma)))  # TODO make a comment here
    print(n)

    j_list = []
    for i in range(nb_simulation):
        x_0 = initial_state()
        j = j_n(x_0, policy, n)
        j_list.append(j)

    return np.mean(j_list)

""" =================================  """

def expected_return_policy(state, policy, N):
    """
    expected return of the policy, value of the value
    function J(x). Exactly the same function as j_n, and return
    the same result
    """

    iteration = N

    if iteration == 0:
        return 0

    else:
        action = policy(state)
        reward = r(state, action)
        next_state = f(state, action)
        return reward + 0.95 * expected_return_policy(next_state, policy, N-1)


def monte_carlo_simulation(policy, N, N_Monte_Carlo):
    """
    Compute simulation, more faster than the function expected_return
    """
    result = []
    initial_state_x = []
    for i in range(N_Monte_Carlo):
        state = initial_state()
        result.append(expected_return_policy(state, policy, N))
        initial_state_x.append(state)

    result = np.array(result)
    initial_state_x = np.array(initial_state_x)
    """
    print(np.shape(result))
    print(np.shape(initial_state_x))
    print(initial_state_x)
    #plt.scatter(initial_state_x, result)
    #plt.show()
    """
    return np.mean(result)


if __name__ == '__main__':

    print("Simulation j_n and expected return")
    x0 = initial_state()
    print(j_n(x0, random_policy, 200))
    #print(expected_return(random_policy, 10, 0.05))

    print()
    print("==================================")
    print()

    print("Monte Carlo Simulation")
    print(monte_carlo_simulation(random_policy, 200, 10))
