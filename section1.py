import numpy as np

U = np.array([-4, 4])  # action space
m = 1
g = 9.81

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
    
    next_state = f(p, s, u)
    p_suivant = next_state[0]
    s_suivant = next_state[1]
    if p_suivant < -1 or np.abs(s_suivant) > 3:
        return -3
    else if p_suivant > 1 and np.abs(s_suivant) <= 3:
        return 3
    else:
        return 0

"""

if __name__ == '__main__':
    print('Hello World !')
