from matplotlib import pyplot as plt
from joblib import load, dump
import math
import random
import domain
import numpy as np
from random import seed
from random import randrange
from random import random
from math import exp
import keras.backend as K
from keras import *
from keras.layers import LSTM
from keras.models import load_model
from Section2 import *
from section5 import *

gamma = 0.95
# alpha is the learning ration associated with the Q-learning, not the learning rate of SGD !!
alpha = 0.05


# ----------------------- Neural Network -------------------------


# function intermediate to compute the temporal difference delta(x, u) = r + gamma * max(Q_N_1(x_suivant, u)) - Q_N_1(x,u)

def delta(step, model_delta):
    """
    Compute the temporal difference state action

    Argument:
    =======
    step = one step system transition (x, u, r, x_next)
    model_delta = Function approximator Q function

    Return:
    ======
    return float, wich is the new temporal difference
    """

    # the model must be defined !
    #assert model_delta is not None
    if model_delta is None:
        result = step[2]
    else:
        x_suivant1 = np.array([[step[3][0], step[3][1], 4]])
        x_suivant2 = np.array([[step[3][0], step[3][1], -4]])
        x = np.array([[step[0][0], step[0][1], step[1]]])
        result = step[2] + gamma * np.max(
            [model_delta.predict(x_suivant1)[0][0], model_delta.predict(x_suivant2)[0][0]]) - model_delta.predict(x)[0][
                         0]
    return result


# Build training set input : (x,u)
# 					 output : Q_N_1 + alpha * delta(x,u)

def build_training_set_parametric_Q_Learning(F, model_build):
    """
    Build the training set for training the parametric
    approximation architecture for the Q-Learning

    Argument:
    =======
    F : is the four-steps set
    model_build : is the Q_(N-1) functions

    Return:
    ======
    return inputs and outputs data sets
    """
    inputs = []  # input set
    outputs = []  # output set
    for step in F:
        i = [step[0][0], step[0][1], step[1]]
        o = delta(step, model_build)

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


"""
===============================
We need to code our own neural network to customize the optimizer and the loss function,
because in the Q-Learning Algorithm, the weight are updated in order to MAXIMIZE the value of
Q(x, u) and the learning rate is NON CONSTANT because it depends on the temporal difference.
==========================
"""


# Define custom loss
def custom_loss(y_true, y_pred):
    """
    Create a loss function wich is L = y_pred * delta(x,u) such that Q(x, u) = y_pred, but the multiplication
    with delta(x, u) will be possible thanks to the parameter sample_weight in the fit method.
    """
    return -y_pred * y_true


def new_baseline_model():
    """
    Define base model for artificial neural network
    """
    # create model
    model_baseline = Sequential()
    model_baseline.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='relu'))
    model_baseline.add(Dropout(0.4))  # avoid overfitting
    model_baseline.add(Dense(5, kernel_initializer='normal', activation='relu'))
    # model_baseline.add(LSTM(units=64))
    model_baseline.add(Dropout(0.4))  # avoid overfitting
    model_baseline.add(Dense(1, kernel_initializer='normal'))

    # Compile model_baseline
    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, clipvalue=0.5, clipnorm=1.)

    # Choose loss parameter : 'mse' or custom_loss
    model_baseline.compile(loss=custom_loss, optimizer=sgd, metrics=['mse'])

    return model_baseline


"""
=========================================================
Q-Learning Algorithm with parametric approximator architecture
=========================================================

Pseudo-code Cannot work with Q-table because of the infinite number of state so:

    - Set Q_0 wich return 0 everywhere
    - Build a training set with Q_0 wich is : input : (x,u)
                                                output : Q_N_1 + alpha * delta(x,u)
    - Training our customize Artificial neural network on this set (loss customize)
    - Build a new training set with the previous Artificial Neural network already train
    - Training a new customize Artificial neural network on this NEW set
    - etc... iteration k times
    - then return the last Artificial Neural Network wich is the Q(x, u)

"""


def Q_learning_parametric_function(F, N):
    """
    new_baseline_model() use a customize loss which return only the y_pred.
    Plus, we have the parameter sample_weight on the fit() method which multiply each y_pred by the correct
    delta corresponding to the sample. So we obtain : y_pred * delta in the output layer

    Then we use the SGD on the Loss : y_pred * delta in order to optimize all the parameters of
    the neural network.
    """

    # store the delta along the training
    temporal_difference = []

    # test for plotting delta w.r.t one input and the model_Q_learning
    t = [np.array([-0.44, -1.43]), -4, 0, np.array([-0.92, 1.24])]  # one step system transition fixed

    model_Q_learning = new_baseline_model()
    for k in range(N):
        print()
        print("================================= iteration k = {} ====================================".format(k))
        print()

        # build batch trajectory with 100 random samples of F
        indexes = np.random.randint(len(F), size=100)
        f = []
        for i in indexes:
            f.append(F[i])

        if k == 0:
            X, y = build_training_set_parametric_Q_Learning(f, None)
        else:
            X, y = build_training_set_parametric_Q_Learning(f, model_Q_learning)

        model_Q_learning.fit(X, y, batch_size=32, epochs=50, verbose=0)

        # computes for each iteration the delta with the updated model
        d = delta(t, model_Q_learning)
        print('delta = {}'.format(d))
        temporal_difference.append(d)

    return model_Q_learning, temporal_difference


"""
=========================================================
"""

# ----------------------- Derive the policy -------------------------

# policy(x) ----> take the argmax(Q(x,u1), Q(x, u2))
# return the u


def policy(x, model_policy):
    """
    policy of the model Q
    """
    x_1 = np.array([[x[0], x[1], 4]])
    x_2 = np.array([[x[0], x[1], -4]])
    return np.argmax([model_policy.predict(x_1)[0][0], model_policy.predict(x_2)[0][0]])

# ----------------------- Expected Return of mu* ---------------------------------

# Calculate J with the policy mu*


# ----------------------- Compare method DNFQI and Q-Learning with Function Approximators -------------------------

# Design an experiment protocol to compare FQI and parametric Q-learning with
# both approximation architectures

# First, compute the speed between FQI and parametric Q-learning to compute
# Second, compare memory complexity, don't have to store every Q function
# Compare the result, the score of Q(x, u) ????


if __name__ == '__main__':
    print("=======================================================================================")
    print("================================= TEST delta ==========================================")
    print("=======================================================================================")

    F_test = first_generation_set_one_step_system_transition(400)
    model2 = load_model('model2.h5')
    print(delta(F_test[12], model2))
    print(model2.predict(np.array([[1, 2, 4]])))

    print("=======================================================================================")
    print("=================== Build Training Set for Parametric Q-Learning ======================")
    print("=======================================================================================")

    X, y = build_training_set_parametric_Q_Learning(F_test, model2)
    print(X, type(X), np.shape(X))
    print(y, type(y), np.shape(y))

    print("=======================================================================================")
    print("=================== Q-Learning Algorithm parametric function ==========================")
    print("=======================================================================================")

    N = 100  # number of iterations
    F = second_generation_set_one_step_system_transition(5000)
    model, delta_test = Q_learning_parametric_function(F, N)


    print("=======================================================================================")
    print("=========================== delta for each Q during Q-learning ========================")
    print("=======================================================================================")
    print(delta_test)

    """
    model.save('parametric_models/Q.h5')
    dump(delta_test, 'delta.joblib')
    """

    N = range(N)
    plt.plot(N, np.abs(delta_test))
    plt.show()


    # load a model with a custom object (e.g. loss function)
    # model = load_model('parametric_models/Q.h5', custom_objects={'custom_loss': custom_loss})


