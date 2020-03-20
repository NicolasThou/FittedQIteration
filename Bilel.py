import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
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


def visualize_Q(model):
    """
    Plot the Q values (for both actions)
    """

    # used to have an approximation of Q along the state space
    p_space = [i/10 for i in range(-10, 11, 1)]
    s_space = [i/10 for i in range(-30, 30, 3)]

    # store the values of the q-functions
    q_functions = []

    for u in [4, -4]:
        q = []
        for s in s_space:
            q_s = []
            for p in p_space:
                # apply the model on both action for this state
                q_s.append(round(model([[p, s, u]]).item(), 2))

            q.append(q_s)

        q_functions.append(q)

    fig, ax = plt.subplots(1, 2)

    for i, q in enumerate(q_functions):
        # plot the q-function as a heatmap
        heatmap = ax[i].imshow(q)

        # annotate the axes
        ax[i].set_xlabel('p')
        ax[i].set_ylabel('s', rotation=0)
        ax[i].set_xticks((np.arange(5)/4)*(np.shape(q)[1] - 1))
        ax[i].set_yticks((np.arange(7)/6)*(np.shape(q)[0] - 1))
        ax[i].set_xticklabels([-1, 0.5, 0, 0.5, 1])
        ax[i].set_yticklabels([-3, -2, -1, 0, 1, 2, 3])

        # display the color map above this heatmap
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("top", size="7%", pad="2%")
        colorbar = fig.colorbar(heatmap, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position("top")

    fig.suptitle('Q-functions : $\hat{Q}_{4}$ in the left and $\hat{Q}_{-4}$ in the right.', fontsize=14)
    plt.show()


def compute_J(state, model, N):
    if N < 0:
        print('N must be positive !')
    elif N == 0:
        return 0
    else:
        p, s = state[0], state[1]
        q = [model([[p, s, 4]]), model([[p, s, -4]])]
        if np.argmax(q) == 0:
            mu = 4
        else:
            mu = -4
        return domain.r(state, mu) + domain.gamma*compute_J(domain.f(state, mu), model, N-1)


def visualize_expected_return_policy(models, error_threshold=0.1):
    """
    Plot the expected return of a policy (Q-function)
    """
    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # set of states X used to have an approximation of J
    values = np.arange(-10, 15, 5)/10

    # values of J along N
    j = []

    for n in range(len(models)):
        print(n)

        # expected return of all states in X
        expected_return_over_X = []

        # for p in values:
        #     for s in 3*values:
        #         expected_return = compute_J((p, s), model[n], N)
        #
        #         expected_return_over_X.append(expected_return)
        #
        # j.append(np.mean(expected_return_over_X))


        j.append(compute_J(domain.initial_state(), models[n], N))

    plt.plot(len(models), j)
    plt.show()


if __name__ == '__main__':
    model = load('models/regression_300_first_1.joblib')

    visualize_expected_return_policy(model, error_threshold=1)


