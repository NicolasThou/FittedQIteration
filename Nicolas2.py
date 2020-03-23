from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras import metrics
import math
import random
import domain
import numpy as np
from Section2 import *
from Bilel import *


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


# define base model for artificial neural network
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))  # avoid overfitting
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
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


"""
=======================================================================
======================== BILEL PART MODIFIED ==========================
=======================================================================
"""


def visualize_Q(name, model, type_of_regressor=0):
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
                if type_of_regressor != 1:
                    q_s.append(round(model([[p, s, u]]).item(), 2))
                else:
                    input = np.array([[p, s, u]])
                    q_s.append(round(model.predict(input)[0][0], 2))

            q.append(q_s)

        q_functions.append(q)

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

    fig.suptitle('Q-functions : $\hat{Q}_{4}$ in the left and $\hat{Q}_{-4}$ in the right for '+ name)
    plt.savefig(name)
    plt.show()


"""
=======================================================================
=======================================================================
=======================================================================
"""

if __name__ == '__main__':

    print("========================TEST OF FIRST ITERATION WITH Q_0===============================")

    F_test = first_generation_set_one_step_system_transition(200)
    q0 = Q_0()
    X_train1, Y_train1 = build_training_set(F_test, q0)

    print("=======================================================================================")
    print("===============================TEST NEXT ITERATION WITH ANN ===========================")
    print("=======================================================================================")

    regressor_test = baseline_model()
    regressor_test.fit(X_train1, Y_train1, batch_size=10, epochs=20)
    X_train2, Y_train2 = build_training_set(F_test, regressor_test)
    print(X_train2)
    print(np.shape(X_train2))
    print(Y_train2)
    print(np.shape(Y_train2))

    """
    test = np.array([[2, 0.54, 4]])
    print(type(regressor.predict(test)))
    print(regressor.predict(test)[0])
    print(regressor.predict(test)[0][0])
    """
    print("=======================================================================================")
    print("======================== Test of the Algorithm for 220 Iteration ======================")
    print("=======================================================================================")

    # comment or uncomment this line of code to test this stopping rule
    #list_of_regressor = fitted_Q_iteration_first_stoppin_rule(F_test, 10, 20)
    #list_of_regressor[-1].save("model1.h5")
    #print("Saved model to disk")

    print("=======================================================================================")
    print("============================= TEST distance ===========================================")
    print("=======================================================================================")

    distance_test = dist(regressor_test, q0, F_test)
    print(distance_test)

    print("=======================================================================================")
    print("============= Test for the fitted Q iteration for the distance stopping rule ==========")
    print("=======================================================================================")

    # comment or uncomment this line of code to test this stopping rule
    #list_of_regressor2 = fitted_Q_iteration_second_stopping_rule(F_test, 10, 100)
    #list_of_regressor2[-1].save("model2.h5")
    #print("Saved model to disk 2")

    print("=======================================================================================")
    print("===================================Test Visualize Q ===================================")
    print("=======================================================================================")

    model1 = load_model('model1.h5')
    model2 = load_model('model2.h5')
    visualize_Q('ANN_First_Stopping_rule', model1, 1)
    visualize_Q('ANN_Second_Stopping_rule', model2, 1)

