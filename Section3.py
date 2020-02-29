import section1 as s1
import numpy as np


def j_n(initial_state, policy, N):  # policy is a function approximator
    """
    state-value function (recurrence version)
    """
    state = initial_state
    j = 0
    for i in range(N):
        action = policy(state)
        j += (s1.gamma**i)*s1.r(state, action)
        state = s1.f(state, action)

    return j


def expected_return(policy, nb_simulation, error_threshold):
    """
    Return an approximation of the expected return of a policy
    over the infinite time horizon using Monte-Carlo simulation
    """
    # N threshold computation using J function bound
    a = (error_threshold * ((1 - s1.gamma)**2))/2
    n = int(np.ceil(np.log(a)/np.log(s1.gamma)))

    j_list = []
    for i in range(nb_simulation):
        initial_state = s1.initial_state()
        j = j_n(initial_state, policy, n)
        j_list.append(j)

    return np.mean(j_list)
