from section1 import *
import numpy as np

def d2pdt(p, s, u):
    """
    Computes the second derivative of p for p<0
    """
    # recurrent term contracted
    k = 2*p + 1

    a = (u - k*(m*g*s + 2*(s**5))/(m*(1 + (s**3)*(k**2))))
    b = 1 + ((s**3)*(k**2)/(1 + (s**2)*(k**2)))
    if b == 0:
        # error to handle in the future
        print('Denominator equal zero !!')
        return None
    else:
        return a/b


def is_final_state(p, s):
    """
    Check whether the system is in a final state or not
    """
    if abs(p) > 1 or abs(s) > 3:
        return True
    else:
        return False


def random_policy():
    """
    return, in a randomly way, 4 or -4
    """
    power = np.random.random_integers(0, 1)
    return ((-1)**power)*4


def euler(p, s, u):
    new_p = p
    new_s = s

    if p < 0:
        f = lambda f_p, f_s, f_u : derivee_seconde_p_inf_0(f_p, f_s, f_u)
    else:
        f = lambda f_p, f_s, f_u : derivee_seconde_p_sup_equal_0(f_p, f_s, f_u)

# euler method with 1000 step
    for i in range(1000):
        temp_p = new_p
        temp_s = new_s

        # we use the values of the previous step, not the new ones !!
        new_p += integration_step*temp_s
        new_s += integration_step*f(temp_p, temp_s, u)

    return new_p, new_s


def f(x, u):
    """
    dynamic of the system
    """

    # initialization
    p, s = x
    new_p = p
    new_s = s

    # check which Hill function to use
    if p < 0:
        f = lambda f_p, f_s, f_u: derivee_seconde_p_inf_0(f_p, f_s, f_u)
    else:
        f = lambda f_p, f_s, f_u: derivee_seconde_p_sup_equal_0(f_p, f_s, f_u)

    # euler method with 1000 step
    for i in range(1000):
        temp_p = new_p
        temp_s = new_s

        # we use the values of the previous step, not the new ones !!
        new_p += integration_step * temp_s
        new_s += integration_step * f(temp_p, temp_s, u)

    return new_p, new_s


if __name__ == '__main__':
    print('Hello World !')
