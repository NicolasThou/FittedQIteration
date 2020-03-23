from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import copy as cp
import random
import domain
import trajectory


Br = 1  # bound value for the reward


def baseline_model():
    """
    define base model for artificial neural network
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
        p = round(np.random.uniform(-1, 1), 2)
        s = round(np.random.uniform(-3, 3), 2)

        x = np.array([p, s])
        u = trajectory.random_policy(x)
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
    l = len(F)  # only F is a list, whereas use np.shape if it is an array
    for sample in F:
        X = [[sample[0][0], sample[0][1], sample[1]]]
        difference = (function1.predict(X) - function2.predict(X))**2
        sum += difference
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
            x0 = np.array([[tuple[3][0], tuple[3][1], 4]])
            x1 = np.array([[tuple[3][0], tuple[3][1], -4]])
            maximum = np.max([Q_N_1.predict(x0).item(), Q_N_1.predict(x1).item()])  # action are 4 or -4
            o = tuple[2] + domain.gamma * maximum  # reward + gamma * Max(Q_N-1) for every action

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

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

    """
        Initialization
    """
    # sequence of approximation of Q_N functions
    sequence_Q_N = []

    """
        Iteration
    """
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
        else:  # means that we use extra trees or linear regression
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

    sequence_Q_N = []  # sequence of approximation of Q_N functions

    """
        First iteration
    """
    N = 1
    X, y = build_training_set(F, sequence_Q_N[N - 1])
    model = cp.deepcopy(algorithm)
    if batch_size is not None and epoch is not None:
        model.fit(X, y, batch_size=batch_size, epoch=epoch)
    else:
        model.fit(X, y)

    sequence_Q_N.append(model)  # add of the Q_N function in the sequence of Q_N functions

    """
        Iteration for N > 1
    """
    distance = dist(sequence_Q_N[N], sequence_Q_N[N-1], F)

    while distance > tolerance_fixed and N <= 50:   # TODO : somme supervised learning algorithm don't provides the convergence !!!
        # we create a new empty model
        model = cp.deepcopy(algorithm)

        # build the training set
        X, y = build_training_set(F, sequence_Q_N[N - 1])

        if batch_size is not None and epoch is not None:
            model.fit(X, y, batch_size=batch_size, epochs=epoch)
        else:
            model.fit(X, y)

        # add of the Q_N function in the sequence of Q_N functions
        sequence_Q_N.append(model)
        distance = dist(sequence_Q_N[N], sequence_Q_N[N - 1], F)

        N = N + 1

    return sequence_Q_N


if __name__ == '__main__':
    pass

