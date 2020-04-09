
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
    assert model_delta is not None

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

        if model_build is None:
            # see the Q-learning update when Q=0 everywhere
            o = alpha * step[2]
        else:
            x = np.array([[step[0][0], step[0][1], step[1]]])
            o = model_build.predict(x)[0][0] + alpha * delta(step, model_build)

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


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


if __name__ == '__main__':
    F_test = first_generation_set_one_step_system_transition(200)
    F_test2 = second_generation_set_one_step_system_transition(200)




