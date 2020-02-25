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


def derivee_seconde_p_inf_0(p, s, u):
    """
    Computes the second derivative of p for p<0

    Argument:
    ========
    p is the position
    s the speed
    u the action

    return:
    ======
    return the value of p"
    """
    # recurrent term contracted
    k = 2 * p + 1

    a = (u - k * (m * g * s + 2 * (s ** 5)) / (m * (1 + (s ** 3) * (k ** 2))))
    b = 1 + ((s ** 3) * (k ** 2) / (1 + (s ** 2) * (k ** 2)))
    if b == 0:
        # error to handle in the future
        print('Denominator equal zero !!')
        return None
    else:
        return a / b


def derivee_seconde_p_sup_equal_0(p, s, u):
    """
    derivative 2th of p when p >= 0

    Argument:
    ========
    p is the position
    s the speed
    u the action

    return:
    ======
    return the value of p"
    """
    denominateur_commun = 1 + ((s ** 2) / (1 + 5 * (p ** 2))) * ((1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2)))) ** 2)
    numerateur1 = u / m
    numerateur2 = g * (s / (np.sqrt(1 + 5 * (p ** 2)))) * (1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2))))
    numerateur3 = ((15 * (s ** 5) * p) / (1 + 5 * (p ** 2))) * (
            ((1 / (np.sqrt((1 + 5 * (p ** 2))))) * (1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2))))) ** 2)
    denominateur_commun2 = 1 + (
            ((1 / (np.sqrt((1 + 5 * (p ** 2))))) * (1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2))))) ** 2) * (s ** 3)

    if denominateur_commun == 0 or denominateur_commun2 == 0:
        # error to handle in the future
        print('Denominator equal zero !!')
        return None

    return ((numerateur1 + numerateur2 + numerateur3) / denominateur_commun) / denominateur_commun2


def is_final_state(p, s):
    """
    Check whether the system is in a final state or not
    """
    if abs(p) > 1 or abs(s) > 3:
        return True
    else:
        return False


if __name__ == '__main__':
    print('Hello World !')
