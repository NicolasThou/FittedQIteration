from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import random
import domain
import numpy as np
from Section2 import *


Br = 1  # bound value for the reward


class Q_0():

    def predict(self, input):
        """
        return 0 everywhere
        """
        return np.array([[0]])

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

    # shuffle the set in so the samples are not correlated
    random.shuffle(transitions)

    return transitions


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
        i = [tuple[0][0], tuple[0][1], tuple[1]]  # p, s, u

        input1 = np.array([[tuple[3][0], tuple[3][1], 4]])  # input of the predictor (x,u)
        input2 = np.array([[tuple[3][0], tuple[3][1], -4]])  # input of the predictor (x,u)
        maximum = np.max([Q_N_1.predict(input1)[0][0], Q_N_1.predict(input2)[0][0]])
        o = tuple[2] + gamma * maximum  # reward + gamma * Max(Q_N-1) for every action
        #print("======= Here the output ======")
        #print(o)
        #print("=====================")

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    # reshape the vectors
    inputs = np.array(inputs)
    outputs = np.array([outputs])
    outputs = outputs.transpose()
    #print(inputs)
    #print(np.shape(inputs))
    #print(outputs)
    #print(np.shape(outputs))

    return inputs, outputs

print("========================TEST OF FIRST ITERATION WITH Q_0====================")

F_test = first_generation_set_one_step_system_transition(1000)
q0 = Q_0()
X_train1, Y_train1 = build_training_set(F_test, q0)

print("=======================================================================================")


# define base model for artificial neural network
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))  # avoid overfitting
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dropout(0.4))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model


def evaluate(model, X_test, Y_test):
    print("==================== EVALUATE ===================")
    score = model.evaluate(X_test, Y_test)
    print(score)
    print("===============================================")


def fitted_Q_iteration_first_stoppin_rule(F, batch_size, epoch):
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

    first_regressor = Q_0()
    sequence_Q_N = [first_regressor]  # sequence of approximation of Q_N functions
    N = 1

    # Iteration
    tolerance_fixed = 0.01
    max = int(math.log(tolerance_fixed * ((1 - gamma) ** 2) / (2 * Br)) / (math.log(gamma))) + 1  # equal to 220
    while N < max:
        print('======================= ITERATION =====================')
        print("Iteration numéro : {}" .format(N))
        model = baseline_model()  # Here we use an artificial network
        X, Y = build_training_set(F, sequence_Q_N[N-1])
        model.fit(X, Y, batch_size=batch_size, epochs=epoch)
        # add of the Q_N function in the sequence of Q_N functions
        sequence_Q_N.append(model)
        N = N + 1

    return sequence_Q_N


print("===============================TEST NEXT ITERATION WITH ANN =============================================")

regressor_test = baseline_model()
regressor_test.fit(X_train1, Y_train1, batch_size=10, epochs=20)
X_train2, Y_train2 = build_training_set(F_test, regressor_test)

"""
test = np.array([[2, 0.54, 4]])
print(type(regressor.predict(test)))
print(regressor.predict(test)[0])
print(regressor.predict(test)[0][0])
"""
print("=======================================================================================")



print("======================== Test of the Algorithm for 220 Iteration =================================")

#list_of_regressor = fitted_Q_iteration_first_stoppin_rule(F_test, 10, 20)

print("=======================================================================================")




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
        input = np.array([[tuple[0][0], tuple[0][1], tuple[1]]])
        difference = (function1.predict(input)[0][0] - function2.predict(input)[0][0])**2
        sum += difference
    return sum/l


distance_test = dist(regressor_test, q0, F_test)
print(distance_test)


def fitted_Q_iteration_second_stopping_rule(F, batch_size, epoch):

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
    first_regressor = Q_0()
    sequence_Q_N = [first_regressor]  # sequence of approximation of Q_N functions
    N = 1

    # First iteration
    X_train, y_train = build_training_set(F, sequence_Q_N[N - 1])
    regressor = baseline_model()
    regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
    sequence_Q_N.append(regressor)

    # Iteration for N > 1
    distance = dist(sequence_Q_N[N], sequence_Q_N[N - 1], F)
    tolerance_fixed = 0.001

    while distance > tolerance_fixed:

        print('======================= ITERATION =====================')
        print("Iteration numéro : {}".format(N))
        print()

        model = baseline_model()  # Here we use an artificial network
        X, Y = build_training_set(F, sequence_Q_N[N - 1])
        model.fit(X, Y, batch_size=batch_size, epochs=epoch)
        # add of the Q_N function in the sequence of Q_N functions
        sequence_Q_N.append(model)
        distance = dist(sequence_Q_N[N], sequence_Q_N[N - 1], F)
        N = N + 1

        print()
        print("**************** the distance is : ********************* ")
        print(distance)

    return sequence_Q_N


print("============= Test for the fitted Q iteration for the distance stopping rule ==========")


fitted_Q_iteration_second_stopping_rule(F_test, 10, 100)


print("=======================================================================================")




if __name__ == '__main__':
    print("hello world")

