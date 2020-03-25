import os
from matplotlib import pyplot as plt
from joblib import load
import trajectory
import domain
import section5


def simulation():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    # we start with an inital state
    state = domain.initial_state()
    print(state[0])
    p = []
    s = []

    # until we reach a final state or more than 50 trnsitions, we apply a random policy
    for i in range(50):
        action = trajectory.random_policy(state)  # use a random policy
        print(action)
        state = domain.f(state, action)  # observe the next state
        print(state)
        p.append(state[0])
        s.append(state[1])

        # if we've reached a final state, we stop the simulation
        if domain.is_final_state(state):
            print('Nous avons atteint un état finale')
            break

    # display the simulation
    fig, axis = plt.subplots()
    plt.xlabel('p')
    plt.ylabel('s', rotation=0)
    axis.plot(p, s, '+')
    axis.set(xlim=(-1, 1), ylim=(-3, 3))
    plt.show()
    return p, s


if __name__ == '__main__':
    models_path = 'models/'

    for name in os.listdir('models/'):
        print(name)
        model_list = load(models_path + name)
        section5.visualize_Q(name + '.png', model_list[-1])

