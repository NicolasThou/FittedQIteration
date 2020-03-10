import numpy as np
from matplotlib import pyplot as plt
import random
import domain


def forward_policy(x):
    return 4


def backward_policy(x):
    return -4


def random_policy(x):
    """
    return, in a randomly way, 4 or -4

    Return:
    ======
    return an action
    """
    power = np.random.randint(0, 2)
    return ((-1) ** power) * 4


def hill(p):
    if p < 0:
        return (p**2) + p
    else:
        return p/(np.sqrt(1 + 5*(p**2)))


def disp_hill():
    """
        Plot the hill
    """
    x = []
    y = []
    dy = []
    ddy = []
    for p in range(-100, 100, 1):
        pos = (p*1.0)/100
        x.append(pos)
        y.append(hill(pos))

        k = 1 + 5 * (pos ** 2)
        dH = 1 / (np.sqrt(k) ** 3)
        ddH = -(15*pos*np.sqrt(k))/(k**3)

        dy.append(dH)
        ddy.append(ddH)

    fig, axis = plt.subplots(3, 1)
    axis[0].plot(x, y, '+')
    axis[1].plot(x, dy, '+')
    axis[2].plot(x, ddy, '+')
    axis[0].set(xlim=(-1, 1))
    axis[1].set(xlim=(-1, 1))
    axis[2].set(xlim=(-1, 1))
    plt.show()


def plot_hill_trajectory(p):
    """
        Plot the trajectory along the hill
    """
    h = []
    for x in p:
        h.append(hill(x))
    fig, axis = plt.subplots()
    axis.plot(p, h, '.')
    axis.set(xlim=(-1, 1), ylim=(-0.5, 0.5))
    plt.show()


def create_transitions1(N):
    """
        Create a one-step transition system of size N
        We start from an initial state and use a random policy.
        Each time, we check if a final state is reached, and if it's the case
        we compute a new initial state
    """
    count = 0
    transitions = []
    x = domain.initial_state()
    while count < N:
        u = random_policy(x)
        r = domain.r(x, u)
        next_x = domain.f(x, u)

        # add the transition to the set
        transitions.append([x, u, r, next_x])

        # increment the counter
        count += 1

        if domain.is_final_state(next_x):
            # if we reached a final state, we start a new trajectory
            x = domain.initial_state()
        else:
            # otherwise, we continue
            x = next_x

    # shuffle the set in so it doesn't seems as a trajectory
    random.shuffle(transitions)

    return transitions


def create_transitions2(N):
    """
        Create a one-step transition system of size N
        We take a random state in the dynamic, apply a random action and observe a reward and a new state.
        Each time we save the four-tuple
    """
    trajectory = []
    count = 0

    while count < N:
        p = round(np.random.uniform(-1, 1), 2)
        s = round(np.random.uniform(-3, 3), 2)

        x = np.array([p, s])
        u = random_policy(x)
        r = domain.r(x, u)
        next_x = domain.f(x, u)

        trajectory.append([x, u, r, next_x])
        count += 1

    return trajectory


if __name__ == '__main__':
    create_transitions2(50)
