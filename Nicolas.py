import numpy as np
from section1 import *

U = np.array([-4, 4])  # action space
m = 1
g = 9.81
gamma = 0.95
delta_t = 0.001


def derivee_seconde_p(p, s, u):
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


if __name__ == '__main__':
    print('Hello World !')
