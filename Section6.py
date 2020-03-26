import math
import random
import domain
import numpy as np
from random import seed
from random import randrange
from random import random
from math import exp
import keras.backend as K
from keras import optimizers
from keras.models import load_model
from Section2 import *
from section5 import *

gamma = 0.95
alpha = 0.05


# ----------------------- Neural Network -------------------------


# function intermediate to compute the temporal difference delta(x, u) = r + gamma * max(Q_N_1(x_suivant, u)) - Q_N_1(x,u)

def delta(tuple, model):
    """
    Compute the temporal difference state action

    Argument:
    =======
    tuple = one step system transition (x, u, r, x_next)
    input =

    Return:
    ======
    return float, wich is the new temporal difference
    """
    if model is None:
        return alpha * tuple[2]
    else:
        x_suivant1 = np.array([[tuple[3][0], tuple[3][1], 4]])
        x_suivant2 = np.array([[tuple[3][0], tuple[3][1], -4]])
        x = np.array([[tuple[0][0], tuple[0][1], tuple[1]]])
        result = tuple[2] + gamma * np.max([model.predict(x_suivant1)[0][0], model.predict(x_suivant2)[0][0]]) - model.predict(x)[0][0]
    return result


# Build training set input : (x,u)
# 					 output : Q_N_1 + alpha * delta(x,u)

def build_training_set_parametric_Q_Learning(F, model):
    """
    Build the training set for training the parametric
    approximation architecture for the Q-Learning

    Argument:
    =======
    F : is the four-tuples set
    model : is the Q_(N-1) functions

    Return:
    ======
    return inputs and outputs data sets
    """
    inputs = []  # input set
    outputs = []  # output set
    temporal_difference = []
    for tuple in F:
        i = [tuple[0][0], tuple[0][1], tuple[1]]

        if model is None:  # First Iteration
            o = tuple[2] * alpha
        else:  # Iteration N > 1
            x = np.array([[tuple[0][0], tuple[0][1], tuple[1]]])
            o = model.predict(x)[0][0] + alpha * delta(tuple, model)

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)
        temporal_difference.append(delta(tuple, model))

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    temporal_difference = np.array(temporal_difference)
    return inputs, outputs, temporal_difference



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
    return y_pred


def custom_loss2(F, model):

    d = delta() # HOW TO CALCULATE delta corresponding to the right input ?!!!!
    def loss(y_true, y_pred):
        return y_pred * d

    # Return a function
    return loss


def new_baseline_model():
    """
    Define base model for artificial neural network
    """
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))  # avoid overfitting
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    sgd = optimizers.SGD(learning_rate=alpha, momentum=0.0, nesterov=False)
    model.compile(loss=custom_loss, optimizer=sgd, metrics=['mse'])
    return model


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

    #Iteration k = 0
    X, y, temporal_difference = build_training_set_parametric_Q_Learning(F, None)
    model = new_baseline_model()
    model.fit(X, y, sample_weight=temporal_difference, batch_size=1, epochs=10)

    #Iteration k>0
    for k in range(1,N):
        X, y, temporal_difference = build_training_set_parametric_Q_Learning(F, model)
        model = new_baseline_model()
        model.fit(X, y, sample_weight=temporal_difference, batch_size=1, epochs=10)

    return model


"""
=========================================================
"""

# ----------------------- Derive the policy -------------------------

# policy(x) ----> take the argmax(Q(x,u1), Q(x, u2))
# return the u


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

    X, y, temporal = build_training_set_parametric_Q_Learning(F_test, model2)
    print(X, type(X), np.shape(X))
    print(y, type(y), np.shape(y))
    print(temporal, np.shape(temporal))

    print("=======================================================================================")
    print("=================== Q-Learning Algorithm parametric function ==========================")
    print("=======================================================================================")

    F = first_generation_set_one_step_system_transition(400)
    Q = Q_learning_parametric_function(F, 2)

    print("=======================================================================================")
    print("================================= X training Set ======================================")
    print("=======================================================================================")
    print(X, np.shape(X))
    print("=======================================================================================")
    print("================================= y Training Set ======================================")
    print("=======================================================================================")
    print(y, np.shape(y))
    print("=======================================================================================")
    print("================================= Temporal Difference =================================")
    print("=======================================================================================")
    print(temporal, np.shape(temporal))
