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
    a = (error_threshold * ((1 - gamma) ** 2)) / 2
    # for N >= n, J_N is a good approximation of J
    n = int(np.ceil(np.log(a) / np.log(gamma)))

    result = []
    for i in range(number_of_sample):
        state = initial_state()
        result.append(j_n(state, policy, n))

    return np.mean(result)


if __name__ == '__main__':
    j = []
    dj = []
    pj = expected_return(Bilel.forward_policy, 10, 0.01)
    sample = range(20, 160, 10)
    for n in sample:
        print(n)
        nj = expected_return(Bilel.forward_policy, n, 0.01)
        j.append(nj)
        dj.append(abs(pj - nj))
        pj = nj

    plt.xlabel('number of simulations')
    plt.ylabel('$|\ J^{\mu}_{N}\ -\ J^{\mu}_{N-1}\ |$')
    plt.plot(sample, dj)
    plt.show()
