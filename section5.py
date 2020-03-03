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



"""
The algorithm of extremely_randomized_trees is coding thanks to the article : Extremely randomized trees
Pierre Geurts · Damien Ernst · Louis Wehenkel (2 March 2006).

"
The term attribute denotes a particular input variable used in a
supervised learning problem. The candidate attributes denote all input variables that are
available for a given problem. We use the term output to refer to the target variable that
defines the supervised learning problem. 

The term learning sample denotes the observations used to build a model, and the term test
sample the observations used to compute its accuracy (error-rate, or mean square-error).
N refers to the size of the learning sample, i.e., its number of observations, and n refers to
the number of candidate attributes, i.e., the dimensionality of the input space. "

"""

def extremely_randomized_trees():
    """
    implement the extra trees algorithm
    """

def Split_a_node(S):
    """
    Input: the local learning subset S corresponding to the node we want to split
    Output: a split [a < ac] or nothing
    """

def Pick_a_random_split(S,a):
    """
    Inputs: a subset S and an attribute a
    Output: a split
    """

def Stop_split(S):
    """
    Input: a subset S
    Output: a boolean
    """




def first_generation_set_one_step_system_transition():
    """
    Fisrt strategies for generating sets of one-step system transitions that will be
    used in your experiments.
    """


def second_generation_set_one_step_system_transition():
    """
    Second strategies for generating sets of one-step system transitions that will be
    used in your experiments.
    """


def Q_0(state, action):
    """
    initialization of the fitted Q-iteration where 0 is initialize everywhere
    """
    return 0



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


def build_training_set(F, Q_N_1):
    """
    Build the training set in the fitted-Q iteration from F for the
    supervised learning algorithm

    Argument:
    =======
    F : is the four-tuples set
    Q_N_1 : is the Q_(N-1) functions

    Return:
    ======
    return X (input), and y (output)
    """
    X = []  # input set
    y = []  # output set
    a = []  # create empty input
    b = []  # create empty output
    for tuple in F:
        input = cp.deepcopy(a)
        output = cp.deepcopy(b)
        input.append(tuple[0])  # add x
        input.append(tuple[1])  # add u
        maximum = np.max(Q_N_1(tuple[3], 4), Q_N_1(tuple[3], -4))  # action are 4 or -4
        output.append(tuple[2] + gamma * maximum)  # reward + gamma * Max(Q_N-1) for every action
        X.append(input)  # add the new sample in the training set
        y.append(output)  # add the new sample in the training set
    X = np.array([X])
    y = np.array([y])
    return X, y


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


