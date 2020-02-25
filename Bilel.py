from section1 import *


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


if __name__ == '__main__':
    print('Hello World !')
