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
from section5 import *

# ----------------------- Neural Network -------------------------

# Build training set input : x,u and output : Q_N_1 + alpha * delta(x,u)
# Fonction intermediaire Calculer delta(x, u) = r + gamma * max(Q_N_1(x_suivant, u)) - Q_N_1(x,u)
# Train ton neural network sauf que :
    # - recoder backpropagation : avec w = w + alpha * delta * (gradient de Q selon a)
    # La loss function c'est Q d'avant qui tend à maximiser le Q(x, u)


# ----------------------- Radial Basis function -------------------------

# Build training set input : x,u and output : Q_N_1 + alpha * delta(x,u)
# Fonction intermediaire Calculer delta(x, u) = r + gamma * max(Q_N_1(x_suivant, u)) - Q_N_1(x,u)
# Train ton ML Algorithm SVR Kernel RBF

# ----------------------- Derive the policy -------------------------

# policy(x) ----> take the argmax(Q(x,u1), Q(x, u2))
# return the u


# ----------------------- Expected Return of mu* -------------------------

# Calculate J with the policy mu*


# ----------------------- Expected Return of mu* -------------------------

# Design an experiment protocol to compare FQI and parametric Q-learning with
# both approximation architectures

# First, compute the speed between FQI and parametric Q-learning to compute
# Second, compare memory complexity, don't have to store every Q function
# Compare the result, the score of Q(x, u) ????




"""
=======================================================================
=======================================================================
=======================================================================
"""

if __name__ == '__main__':
    print('hello')

