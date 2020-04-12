from keras import *
from keras.models import load_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from joblib import load, dump
import os
from Section2 import *
from section5 import *
from RadialBasisFunctionNet import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

gamma = 0.95
# alpha is the learning ration associated with the Q-learning, not the learning rate of SGD !!
alpha = 0.05


def delta(step, model, model_type):
    """
    Compute the temporal difference state action

    Argument:
    =======
    step = one step system transition (x, u, r, x_next)
    model = Function approximator Q function

    Return:
    ======
    return float, wich is the new temporal difference
    """

    # assert model_delta is not None  # the model must be defined !

    if model is None:
        result = step[2]
    else:
        x_suivant1 = np.array([[step[3][0], step[3][1], 4]])
        x_suivant2 = np.array([[step[3][0], step[3][1], -4]])
        x = np.array([[step[0][0], step[0][1], step[1]]])
        if model_type == 'NN':
            result = step[2] + gamma * np.max([model.predict(x_suivant1)[0][0], model.predict(x_suivant2)[0][0]]) - \
                     model.predict(x)[0][0]
        else:
            result = step[2] + gamma * np.max([model.predict(x_suivant1).item(), model.predict(x_suivant2).item()]) - \
                     model.predict(x).item()

    return result


def build_training_set_parametric_Q_Learning(F, model_build, model_type):
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

        if is_final_state(step[3]):
            o = step[2]
        else:
            if model_build is None:
                o = step[2]
            else:
                o = delta(step, model, model_type)

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


def NeuralNetwork():
    """
    Define base model for artificial neural network
    """
    # create model
    model_baseline = Sequential()
    model_baseline.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='relu'))
    model_baseline.add(Dropout(0.4))  # avoid overfitting
    model_baseline.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model_baseline.add(Dropout(0.4))  # avoid overfitting
    model_baseline.add(Dense(1, kernel_initializer='normal'))

    # Compile model_baseline
    sgd = optimizers.SGD(learning_rate=0.01)

    # Choose loss parameter : 'mse'
    model_baseline.compile(loss='MSE', optimizer=sgd, metrics=['mse'])

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


def Q_learning_parametric_function(F, N, model_type):
    """
    new_baseline_model() use a customize loss which return only the y_pred.
    Plus, we have the parameter sample_weight on the fit() method which multiply each y_pred by the correct
    delta corresponding to the sample. So we obtain : y_pred * delta in the output layer

    Then we use the SGD on the Loss : y_pred * delta in order to optimize all the parameters of
    the neural network.
    """
    assert (model_type == 'NN' or model_type == 'RBFN')

    # store the delta along the training
    temporal_difference = []

    # test for plotting delta w.r.t one input and the model_Q_learning
    t = [np.array([-0.44, -1.43]), -4, 0, np.array([-0.92, 1.24])]  # one step system transition fixed

    # Initialize the model
    if model_type == 'NN':
        model_Q_learning = NeuralNetwork()
    else:
        model_Q_learning = RBFNet(k_centers=4)

    for k in range(N):
        print()
        print("================================= iteration k = {} ====================================".format(k))
        print()

        # build batch trajectory with 100 random samples of F
        idx = np.random.choice(range(len(F)), size=500).tolist()
        f = np.array(F)[idx].tolist()

        if k == 0:
            X, y = build_training_set_parametric_Q_Learning(f, None, model_type)
        else:
            X, y = build_training_set_parametric_Q_Learning(f, model_Q_learning, model_type)

        if model_type == 'NN':
            model_Q_learning.fit(X, y, batch_size=32, epochs=50, verbose=0)
        else:
            model_Q_learning.fit(X, y, epochs=100, verbose=False)

        # computes for each iteration the delta with the updated model
        d = delta(t, model_Q_learning, model_type)
        print('delta = {}'.format(d))
        temporal_difference.append(d)

    return model_Q_learning, temporal_difference


# ------------------------ PLOT CURVE AND TENDANCY ---------------
def show(X, y, title, xlabel, ylabel):
    """
    Argument:
    ========
    result : variable type "list" and represent the value we obtain like delta, or J-value
    N_iteraion : Integer, Number of iterations we use to compute the results

    """
    # Fit
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    # Visualize
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue', linewidth=3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    torch.device("cuda:0")

    # print("=======================================================================================")
    # print("================================= TEST delta ==========================================")
    # print("=======================================================================================")
    #
    # F_test = first_generation_set_one_step_system_transition(400)
    # model2 = load_model('model2.h5')
    # print(delta(F_test[12], model2))
    # print(model2.predict(np.array([[1, 2, 4]])))
    #
    # print("=======================================================================================")
    # print("=================== Build Training Set for Parametric Q-Learning ======================")
    # print("=======================================================================================")
    #
    # X, y = build_training_set_parametric_Q_Learning(F_test, model2)
    # print(X, type(X), np.shape(X))
    # print(y, type(y), np.shape(y))

    print("=======================================================================================")
    print("=================== Q-Learning Algorithm parametric function ==========================")
    print("=======================================================================================")

    Number_of_iteration = 100  # number of iterations
    F = second_generation_set_one_step_system_transition(5000)
    model, delta_test = Q_learning_parametric_function(F, Number_of_iteration, 'NN')

    print("=======================================================================================")
    print("=========================== delta for each Q during Q-learning ========================")
    print("=======================================================================================")

    title = 'Result of the temporal difference along the iterations'
    xlabel = 'Number of Iteration'
    ylabel = 'Temporal Difference'
    show(np.reshape(range(Number_of_iteration), (-1, 1)), delta_test, title=title, xlabel=xlabel, ylabel=ylabel)
