import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
import trajectory
import domain
import section5 as s5
import ScikitLinearRegression as SLR
import ScikitExtraTree as SET


def simulation():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = domain.initial_state()
    print(state[0])
    p = []
    s = []
    for i in range(50):
        action = trajectory.forward_policy(state)  # use a random policy
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
    reg_models = load('models/regression_300_first_1.joblib')
    tree_models = load('models/tree_300_first_1.joblib')

    n = range(min(len(reg_models)-1, len(tree_models)-1))
    print(n)

    fig, ax = plt.subplots(2)
    tree_errors = []
    reg_errors = []
    for i in n:
        test_trajectory = s5.first_generation_set_one_step_system_transition(50)
        X_test, Y_test = s5.build_training_set(test_trajectory, None, 1)

        d_tree = np.mean(abs(tree_models[i](X_test) - tree_models[i+1](X_test)))
        d_reg = np.mean(abs(reg_models[i](X_test) - reg_models[i+1](X_test)))

        tree_errors.append(d_tree)
        reg_errors.append(d_reg)

    ax[0].plot(n, tree_errors)
    ax[1].plot(n, reg_errors)

    plt.show()


