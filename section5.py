import numpy as np
from Section2 import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import math
import copy as cp
import random
import domain

Br = 1  # bound value for the reward


def neural_network():
    """
    Build the baseline for an artificial neural network with keras
    """
    regressor = Sequential()
    regressor.add(Dense(units=4, input_dim=3, kernel_initializer='random_uniform', activation='relu'))
    regressor.add(Dropout(rate=0.1))  # avoid overfitting
    regressor.add(Dense(units=4, kernel_initializer='random_uniform', activation='relu'))
    regressor.add(Dropout(rate=0.1))  # avoid overfitting
    regressor.add(Dense(units=1, kernel_initializer='random_uniform', activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return regressor


def accuracy_neural_network(X_train, y_train, X_test, y_test):
    """
    test the regressor Ann, with a K-folds cross validation method

    Return:
    ======
    print the average accuracy, confusion matrix, the best score obtain by the best parameters

    """

    # use of confusion matrix

    regressor = neural_network()
    regressor.fit(X_train, y_train, batch_size=2, epochs=2)
    y_pred = regressor.predict(X_test)


    # use of k-fold cross validation

    regressor = KerasRegressor(build_fn=neural_network, batch_size=10, nb_epoch=100)
    accuracies = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=3, n_jobs=-1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print('The mean of accuracy is : {}' .format(mean))
    print('The variance is : {}' .format(variance))



    # use of GridSearchCV method to find the best hyperparameters epoch, batch_size and the best optimizer

    regressor2 = KerasRegressor(build_fn=neural_network)
    parameters = {'batch_size' : [10, 20, 30],
                  'nb_epoch': [100, 200, 300],
                  'optimizer': ['adam', 'rmsprop']}  # we test different parameters
    grid_search = GridSearchCV(estimator=regressor2, param_grid=parameters, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)
    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_param)
    print(best_score)


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


def second_generation_set_one_step_system_transition(N):
    """
    Second strategies for generating sets of one-step system transitions that will be
    used in your experiments.
    We take a random state in the dynamic, apply a random action and observe a reward and a new state.
    Each time we save the four-tuple.

    :argument
        N : length of the system
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


def dist(function1, function2, F):
    """
    Compute the distance between this 2 functions

    return:
    ======
    return an integer, the distance between this 2 functions
    """
    sum = 0
    l = len(F)  # only F is a list, whereas use np.shape if it is an array
    for tuple in F:
        difference = (function1(tuple[0], tuple[1]) - function2(tuple[0], tuple[1]))**2
        sum += difference
    return sum/l


def build_training_set(F, Q, N):
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
        i = [tuple[0], tuple[1]]

        if N == 0:  # First Iteration

            o = tuple[2]
        else:  # Iteration N>1
            maximum = np.max(Q_N_1(tuple[3], 4), Q_N_1(tuple[3], -4))  # action are 4 or -4
            o = tuple[2] + gamma * maximum  # reward + gamma * Max(Q_N-1) for every action

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


def fitted_Q_iteration_first_stoppin_rule(F, regressor, batch_size=0, epoch=0):
    """
    Implement the fitted-Q Iteration Algorithm with as stopping rule a maximum N
    computed by fixing a tolerance

    Argument:
    ========
    F : is the four-tuples set
    Regressor : is the supervised learning algorithm

    Return:
    ======
    Return the sequence of approximation of Q_N function
    """

    # Initialization

    sequence_Q_N = []  # sequence of approximation of Q_N functions
    sequence_Q_N.add(Q_0)  # we add the Q_0 function which return 0 everywhere
    N = 0

    # Iteration

    tolerance_fixed = 0.01
    max = int(math.log(tolerance_fixed * ((1 - gamma) ** 2) / (2 * Br)) / (math.log(gamma)))  # equal to 220

    while N < max:
        N = N + 1
        X, y = build_training_set(F, sequence_Q_N[N-1])
        if batch_size != 0 and epoch != 0:  # means that we use a neural network as a supervised learning algorithm
            regressor.fit(X, y, batch_size=batch_size, epochs=epoch)  # regressor is an argument, might be copy before fitting and add in the sequence ?
        else:  # means that we use extra trees or logistic regression
            regressor.fit(X, y)
        sequence_Q_N.add(regressor)  # add of the Q_N function in the sequence of Q_N functions
    return sequence_Q_N


def fitted_Q_iteration_second_stopping_rule(F, regressor, batch_size=0, epoch=0):
    """
    Implement the fitted-Q Iteration Algorithm with the stopping rule as
    the distance bewteen Q_N and Q_N-1

    Argument:
    ========
    F : is the four-tuples set
    Regressor : is the supervised learning algorithm

    Return:
    ======
    Return the sequence of approximation of Q_N function
    """

    # Initialization

    sequence_Q_N = []  # sequence of approximation of Q_N functions
    sequence_Q_N.add(Q_0)  # we add the Q_0 function which return 0 everywhere

    # First iteration
    N = 1
    X, y = build_training_set(F, sequence_Q_N[N - 1])
    if batch_size != 0 and epoch != 0:
        regressor.fit(X, y, batch_size=batch_size,
                      epoch=epoch)  # regressor is an argument, might be copy before fitting and add in the sequence ?
    else:
        regressor.fit(X, y)
    sequence_Q_N.add(regressor)  # add of the Q_N function in the sequence of Q_N functions

    # Iteration for N > 1
    distance = dist(sequence_Q_N[N], sequence_Q_N[N-1], F)
    print("the first distance is : ")
    print(distance)
    tolerance_fixed = 0.01
    while distance > tolerance_fixed:
        N = N + 1
        X, y = build_training_set(F, sequence_Q_N[N - 1])
        if batch_size != 0 and epoch != 0:
            regressor.fit(X, y, batch_size=batch_size,
                          epochs=epoch)  # regressor is an argument, might be copy before fitting and add in the sequence ?
        else:
            regressor.fit(X, y)
        sequence_Q_N.add(regressor)  # add of the Q_N function in the sequence of Q_N functions
        distance = dist(sequence_Q_N[N], sequence_Q_N[N - 1], F)
    return sequence_Q_N


if __name__ == '__main__':
    print()
    print('My own test')
    tolerance_fixed = 0.01
    max = int(math.log(tolerance_fixed * ((1 - gamma) ** 2) / (2 * Br)) / (math.log(gamma)))  # equal to 220
    print(max)


    ann = neural_network()
    X_train = np.array([[0.5, 2, 4], [-0.5, 2.2, 4], [0.4, 2, -4], [0.6, 2, -4]])
    print(np.shape(X_train))
    X_test = np.array([[0.4, 2, -4], [0.5, 1, -4], [0.5, 1, 4], [0.98, 2, 4]])
    y_train = [1, 0, -1, 1]
    y_test = [-1, 0, 0, -1]
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)



    accuracy_neural_network(X_train, y_train, X_test, y_test)

    print()
    print('------ First strategies for generating sets of one-step system transitions --------')
    print()

    print()
    print('------- Second strategies for generating sets of one-step system transitions ---------')
    print()

    print()
    print('------------ Q_N with the first stopping rules --------------')
    print()

    print()
    print('------------- Q_N with the second stopping rules ----------------')
    print()

    print()
    print('----------- Display of Q_N using Linear Logistic Regression -------------')
    print()

    print()
    print('------------- Display of Q_N using Extremely Randomized Tree -------------- ')
    print()

    print()
    print('------------- Display of Q_N using Neural Network ----------------- ')
    print()

    print()
    print('------------------ J_N Expected return of µ^*_N ------------------- ')
    print()


