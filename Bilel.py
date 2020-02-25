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


def euler_positive(p, s, u):
    new_p = p
    new_s = s
    for i in range(1000):
        new


def f(x, u):
    p, s = x
    if p < 0:
        new_p, new_s = euler_positive(p, s, u)
    else:
        new_p, new_s = euler_negative(p, s, u)

    return new_p, new_s




if __name__ == '__main__':
    print('Hello World !')
