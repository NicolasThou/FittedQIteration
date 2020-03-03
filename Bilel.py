import numpy as np
from matplotlib import pyplot as plt
import Section2 as s2


def forward_policy(x):
    return 4


def backward_policy(x):
    return -4


def hill(p):
    if p < 0:
        return (p**2) + p
    else:
        return p/(np.sqrt(1 + 5*(p**2)))


def simulation_section2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    print(state[0])
    p = []
    s = []
    for i in range(50):
        action = forward_policy(state)  # use a random policy
        print(action)
        state = f(state, action)  # use the dynamic of the domain
        print(state[0])
        p.append(state[0])
        s.append(state[1])
        if is_final_state(state):
            print('Nous avons atteint un état finale')
            break
    fig, axis = plt.subplots()
    plt.xlabel('p')
    plt.ylabel('s', rotation=0)
    axis.plot(p, s, '+')
    axis.set(xlim=(-1, 1), ylim=(-3, 3))
    plt.show()
    return p, s


def disp_hill():
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
    h = []
    for x in p:
        h.append(hill(x))
    fig, axis = plt.subplots()
    axis.plot(p, h, '.')
    axis.set(xlim=(-1, 1), ylim=(-0.5, 0.5))
    plt.show()


if __name__ == '__main__':
    assert is_final_state(np.array([-2, 0]))
    assert is_final_state(np.array([0, 5]))

    p, _ = simulation_section2()
