import numpy as np
from matplotlib import pyplot as plt
import trajectory
import domain


def simulation():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = domain.initial_state()
    print(state[0])
    p = []
    s = []
    for i in range(50):
        action = forward_policy(state)  # use a random policy
        print(action)
        state = domain.f(state, action)  # use the dynamic of the domain
        print(state[0])
        p.append(state[0])
        s.append(state[1])
        if domain.is_final_state(state):
            print('Nous avons atteint un état finale')
            break
    fig, axis = plt.subplots()
    plt.xlabel('p')
    plt.ylabel('s', rotation=0)
    axis.plot(p, s, '+')
    axis.set(xlim=(-1, 1), ylim=(-3, 3))
    plt.show()
    return p, s


if __name__ == '__main__':
    pass
