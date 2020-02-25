import numpy as np

U = np.array([-4, 4])  # action space
m = 1
g = 9.81
gamma = 0.95
delta_t = 0.001

"""
def r(p, s, u):
    
    reward function
    Argument:
    ========
    p is the position
    s the speed
    u the action
    return:
    ======
    return an integer
    
    next_state = f(p, s, u) # use of the dynamic of the problem
    p_suivant = next_state[0]
    s_suivant = next_state[1]
    if p_suivant < -1 or np.abs(s_suivant) > 3:
        return -3
    else if p_suivant > 1 and np.abs(s_suivant) <= 3:
        return 3
    else:
        return 0

"""


def initial_state():
    """
    Initialization of the state with p0 uniform distribution
    between -0.1 and 0.1 and s0 always equal to 0

    return:
    ======
    The state [p0, s0] which is an array
    """
    p0 = (np.random.random_integers(-1, 1)) / 10
    s0 = 0
    return np.array([p0, s0])


if __name__ == '__main__':
    print('Hello World !')
