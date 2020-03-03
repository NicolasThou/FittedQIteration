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


def expected_return(policy, nb_simulation, error_threshold):
    """
    Return an approximation of the expected return of a policy
    over the infinite time horizon using Monte-Carlo simulation
    """
    # N threshold computation using J function bound
    a = (error_threshold * ((1 - s2.gamma)**2))/2
    # for N >= n, J_N is a good approximation of J
    n = int(np.ceil(np.log(a)/np.log(s2.gamma)))

    j_list = []
    for i in range(nb_simulation):
        x_0 = s2.initial_state()

        # compute the infinite time horizon state value function for this initial state
        j = j_n(x_0, policy, n)

        # add it to the list of the Monte-Carlo simulation
        j_list.append(j)

    return np.mean(j_list)


def monte_carlo_simulation(policy, N_Monte_Carlo):
    """
    Compute simulation, more faster than the function expected_return
    """
    error_threshold = 0.01
    # N threshold computation using J function bound
    a = (error_threshold * ((1 - gamma) ** 2)) / 2
    # for N >= n, J_N is a good approximation of J
    n = int(np.ceil(np.log(a) / np.log(gamma)))

    result = []
    initial_state_x = []
    for i in range(N_Monte_Carlo):
        state = initial_state()
        result.append(j_n(state, policy, n))
        initial_state_x.append(state)

    result = np.array(result)
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
