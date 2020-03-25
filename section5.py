from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import colors, pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import copy as cp
from joblib import load
import random
import domain
import trajectory


Br = 1  # bound value for the reward


def baseline_model():
    """
    Define base model for artificial neural network
    """
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))  # avoid overfitting
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    return model


def first_generation_set_one_step_system_transition(N):
    """
    First strategies for generating sets of one-step system transitions that will be
    used in your experiments.
    We start from an initial state and use a random policy.
    Each time, we check if a final state is reached, and if it's the case we compute a new initial state.

    :argument
        N : length of the system
    """
    count = 0
    transitions = []
    x = domain.initial_state()
    while count < N:
        u = trajectory.random_policy(x)
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

    # shuffle the set in so the samples are not correlated
    random.shuffle(transitions)

    return transitions


def second_generation_set_one_step_system_transition(N):
    """
    Second strategies for generating sets of one-step system transitions that will be
    used in your experiments.
    We take a random state in the dynamic, apply a random action and observe a reward and a new state.
    Each time we save the four-tuple.

    :argument
        N : length of the system
    """
    # history of four-tuples
    ht = []
    count = 0

    while count < N:
        # draw randomly a state in the domain
        p = round(np.random.uniform(-1, 1), 2)
        s = round(np.random.uniform(-3, 3), 2)

        x = np.array([p, s])

        # apply a random policy
        u = trajectory.random_policy(x)

        # observe reward and next state
        r = domain.r(x, u)
        next_x = domain.f(x, u)

        ht.append([x, u, r, next_x])
        count += 1

    return ht


def dist(function1, function2, F):
    """
    Compute the distance between this 2 functions

    return:
    ======
    return an integer, the distance between this 2 functions
    """
    sum = 0
    l = len(F)
    for sample in F:
        X = np.array([[sample[0][0], sample[0][1], sample[1]]])
        difference = (function1.predict(X) - function2.predict(X))**2
        sum += difference.item()
    return sum/l


def build_training_set(F, Q_N_1):
    """
    Build the training set in the fitted-Q iteration from F for the
    supervised learning algorithm

    Argument:
    =======
    F : is the four-tuples set
    Q : is the Q_(N-1) functions

    Return:
    ======
    return inputs and outputs
    """
    inputs = []  # input set
    outputs = []  # output set
    for tuple in F:
        i = [tuple[0][0], tuple[0][1], tuple[1]]

        if Q_N_1 is None:  # First Iteration
            o = tuple[2]
        else:  # Iteration N > 1
            # action are 4 or -4
            x0 = np.array([[tuple[3][0], tuple[3][1], 4]])
            x1 = np.array([[tuple[3][0], tuple[3][1], -4]])
            maximum = np.max([Q_N_1.predict(x0).item(), Q_N_1.predict(x1).item()])

            o = tuple[2] + domain.gamma * maximum

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    # methhod 'fit' needs an ndarray, we have to convert the lists to ndarray
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


def fitted_Q_iteration_first_stopping_rule(F, algorithm, tolerance_fixed=0.001, batch_size=None, epoch=None):
    """
    Implement the fitted-Q Iteration Algorithm with as stopping rule a maximum N
    computed by fixing a tolerance

    Argument:
    ========
    F : is the four-tuples set
    algorithm : is the initializer of a supervised learning algorithm
    tolerance_fixed : is the threshold under which fall the infinite norm of the difference of the expected return function

    Return:
    ======
    Return the sequence of approximation of Q_N function
    """

    # sequence of approximation of Q_N functions
    sequence_Q_N = []

    max = int(np.log(tolerance_fixed * ((1 - domain.gamma) ** 2) / (2 * Br)) / (np.log(domain.gamma))) + 1
    N = 0

    while N < max:
        # we create a new empty model
        model = cp.deepcopy(algorithm)

        # check if we are in the first step or not
        previous_Q = sequence_Q_N[N-1] if len(sequence_Q_N) > 0 else None

        # get the training step built with the trajectory
        X, Y = build_training_set(F, previous_Q)

        if batch_size is not None and epoch is not None:  # means that we use a neural network as a supervised learning algorithm
            model.fit(X, Y, batch_size=batch_size, epochs=epoch, verbose=0)
        else:  # means that we use extra-trees or linear regression
            model.fit(X, Y)

        # add of the Q_N function in the sequence of Q_N functions
        sequence_Q_N.append(model)

        N = N + 1
    return sequence_Q_N


def fitted_Q_iteration_second_stopping_rule(F, algorithm, tolerance_fixed=0.01, batch_size=None, epoch=None):
    """
    Implement the fitted-Q Iteration Algorithm with the stopping rule as
    the distance bewteen Q_N and Q_N-1

    Argument:
    ========
    F : is the four-tuples set
    algorithm : is the initializer of a supervised learning algorithm
    tolerance_fixed : is the threshold under which fall the infinite norm of the difference of the expected return function

    Return:
    ======
    Return the sequence of approximation of Q_N function
    """

    # sequence of approximation of Q_N functions
    sequence_Q_N = []

    """
        First iteration
    """
    X, y = build_training_set(F, None)
    model = cp.deepcopy(algorithm)
    if batch_size is not None and epoch is not None:
        model.fit(X, y, batch_size=batch_size, epochs=epoch, verbose=0)
    else:
        model.fit(X, y)

    sequence_Q_N.append(model)  # add of the Q_N function in the sequence of Q_N functions

    """
        Iteration for N > 1
    """
    # we want to do at least one iteration
    distance = tolerance_fixed + 1

    N = 1
    while distance > tolerance_fixed and N <= 50:   # TODO : somme supervised learning algorithm don't provides the convergence !!!
        # we create a new empty model
        model = cp.deepcopy(algorithm)

        # build the training set
        X, y = build_training_set(F, sequence_Q_N[N - 1])

        if batch_size is not None and epoch is not None:
            model.fit(X, y, batch_size=batch_size, epochs=epoch, verbose=0)
        else:
            model.fit(X, y)

        # add of the Q_N function in the sequence of Q_N functions
        sequence_Q_N.append(model)
        distance = dist(sequence_Q_N[N], sequence_Q_N[N - 1], F)

        N = N + 1
    return sequence_Q_N


def visualize_Q(name, model):
    """
    Plot the Q values (for both actions)
    """

    # approximation of the state space
    p_space = [i/10 for i in range(-10, 11, 1)]
    s_space = [i/10 for i in range(-30, 30, 3)]

    # store the values of the q-functions
    q_functions = []

    for u in [4, -4]:
        q = []
        for s in s_space:
            q_s = []
            for p in p_space:
                # apply the model for this state/action pair
                input = np.array([[p, s, u]])
                q_s.append(round(model.predict(input).item(), 2))

            q.append(q_s)

        q_functions.append(q)

    # we want the heatmaps be based on the same heat scale
    vmin, vmax = np.amin(q_functions), np.amax(q_functions)

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

        # make the values of both image fall into the same range
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heatmap.set_norm(norm)

        # display the color map above this heatmap
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("top", size="7%", pad="2%")
        colorbar = fig.colorbar(heatmap, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position("top")

    fig.suptitle('Q-functions : $\hat{Q}_{4}$ in the left and $\hat{Q}_{-4}$ in the right.', fontsize=14)
    plt.savefig('plots/Q/' + name)
    plt.show()


def compute_J(state, model, N):
    """
    Compute the state value function for a particular state
    """
    if N < 0:
        print('N must be positive !')
    elif N == 0:
        return 0
    else:
        p, s = state[0], state[1]

        # compute which action leads to best Q value (optimal policy)
        first_action_input = np.array([[p, s, 4]])
        second_action_input = np.array([[p, s, -4]])
        q = [model.predict(first_action_input), model.predict(second_action_input)]
        if q[0] > q[1]:
            mu = 4
        else:
            mu = -4

        return domain.r(state, mu) + domain.gamma*compute_J(domain.f(state, mu), model, N-1)


def visualize_expected_return_policy(name, models, error_threshold=0.1):
    """
    Plot the expected return of a policy (Q-function)
    """
    # compute the n minimum for which Jn is a good approximation of J
    N = int(np.ceil(np.log(error_threshold*(1 - domain.gamma))/np.log(domain.gamma)))

    # we compute the expected return for a maximum of 50 models
    n = min(50, len(models))

    # set of states X used to have an approximation of J
    p_values = [round(random.uniform(-1, 1), 2) for i in range(5)]
    s_values = [round(random.uniform(-3, 3), 2) for i in range(5)]

    # values of J along N
    j = []

    for i in range(n):
        expected_return_over_X = []

        # we compute the expected return for a certain number of states, and take the average
        for p in p_values:
            for s in s_values:
                expected_return = compute_J((p, s), models[i], N)

                expected_return_over_X.append(expected_return)

        j.append(np.mean(expected_return_over_X))

    # plot the expected return along N
    plt.plot(range(n), j)
    plt.xlabel('N')
    plt.ylabel('$J^{\hat{\mu_{N}^{*}}}$', rotation=0)
    plt.savefig('plots/J/' + name)
    plt.show()


if __name__ == '__main__':

    print("======================= TEST OF FIRST ITERATION WITH Q_0 ==============================")
    F_test = first_generation_set_one_step_system_transition(200)
    X_train1, Y_train1 = build_training_set(F_test, None)
    regressor_test = baseline_model()
    regressor_test.fit(X_train1, Y_train1, batch_size=10, epochs=20, verbose=0)

    print("=======================================================================================")
    print("============================== TEST NEXT ITERATION WITH ANN ===========================")
    print("=======================================================================================")
    X_train2, Y_train2 = build_training_set(F_test, regressor_test)
    regressor_test_2 = baseline_model()
    regressor_test_2.fit(X_train2, Y_train2, batch_size=10, epochs=20, verbose=0)

    # print("=======================================================================================")
    # print("======================== Test of the Algorithm for 220 Iteration ======================")
    # print("=======================================================================================")
    # comment or uncomment this line of code to test this stopping rule
    # list_of_regressor = fitted_Q_iteration_first_stoppin_rule(F_test, 10, 20)
    # list_of_regressor[-1].save("model1.h5")
    # print("Saved model to disk")

    print("=======================================================================================")
    print("============================= TEST distance ===========================================")
    print("=======================================================================================")
    distance_test = dist(regressor_test_2, regressor_test, F_test)
    print('distance = {}'.format(distance_test))

    # print("=======================================================================================")
    # print("============= Test for the fitted Q iteration for the distance stopping rule ==========")
    # print("=======================================================================================")
    # comment or uncomment this line of code to test this stopping rule
    # list_of_regressor2 = fitted_Q_iteration_second_stopping_rule(F_test, 10, 100)
    # list_of_regressor2[-1].save("model2.h5")
    # print("Saved model to disk 2")

    print("=======================================================================================")
    print("================================== Test Visualize Q ===================================")
    print("=======================================================================================")
    model = load('models/neural_net_first_1.joblib')[-1]  # visualize Qn
    visualize_Q('ANN_Second_Stopping_rule', model)

    print("=======================================================================================")
    print("========================== Test Visualize Expected Return =============================")
    print("=======================================================================================")
    model = load('models/neural_net_first_2.joblib')
    visualize_expected_return_policy('ANN_Second_Stopping_rule', model)




