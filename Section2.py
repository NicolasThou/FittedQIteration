import numpy as np

U = np.array([-4, 4])  # action space
m = 1
g = 9.81
gamma = 0.95
delta_t = 0.1  # discretization time
integration_step = 0.001  # euler method step


def r(x, u):
    """
    reward function
    Argument:
    ========
    p is the position
    s the speed
    u the action
    return:
    ======
    return an integer
    """

    # use of the dynamic of the problem
    p, s = f(x, u)

    if p < -1 or np.abs(s) > 3:
        return -1
    elif p > 1 and np.abs(s) <= 3:
        return 1
    else:
        return 0


def initial_state():
    """
    Initialization of the state with p0 uniform distribution
    between -0.1 and 0.1 and s0 always equal to 0

    return:
    ======
    The state [p0, s0] which is an array
    """
    p0 = np.random.uniform(-0.1, 0.1)
    s0 = 0
    return np.array([p0, s0])


def f(x, u):
    """
    dynamic of the system
    """

    # initialization
    p, s = x[0], x[1]

    # euler method with 1000 step
    for i in range(1000):
        previous_p = p  # we keep the previous value
        previous_s = s  # we keep the previous value

        # we use the values of the previous step, not the new ones !!
        p = previous_p + integration_step * previous_s
        s = previous_s + integration_step * f_s(previous_p, previous_s, u)

    return np.array([p, s])


def f_s(p, s, u):
    if p < 0:
        # first and second derivatives of Hill(p)
        dH = 2 * p + 1
        ddH = 2
    else:
        # recurrent term
        k = 1 + 5 * (p ** 2)

        # Hill'(p)
        dH = 1 / (np.sqrt(k) ** 3)
        # Hill''(p)
        ddH = -(15 * p * np.sqrt(k)) / (k ** 3)

    first_term = u / (m * (1 + dH ** 2))
    second_term = -(g * dH) / (1 + dH ** 2)
    third_term = -((s ** 2) * dH * ddH) / (1 + dH ** 2)

    return first_term + second_term + third_term


def is_final_state(x):
    """
    Check whether the system is in a final state or not

    Argument:
    =======
    p the position
    s the speed

    Return:
    ======
    return a boolean
    """
    p, s = x
    if abs(p) > 1 or abs(s) > 3:
        return True
    else:
        return False


def random_policy(x):
    """
    return, in a randomly way, 4 or -4

    Return:
    ======
    return an action
    """
    power = np.random.randint(0, 2)
    return ((-1) ** power) * 4


def policy_accelerate(state):
    """
    return always 4
    """
    return 4


def policy_backward(x):
    """
    return always -4
    """
    return -4


def simulation_section2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    a = []
    for i in range(50):
        action = random_policy(state)  # use a random policy
        a.append(action)
        state = f(state, action)  # use the dynamic of the domain
        if is_final_state(state):
            print('We\'ve reached a finale state, actions are :')
            print(a)
            return None

    # if never reach a final state, print the actions
    print('Actions are :')
    print(a)


if __name__ == '__main__':
    print('Use of random policy')
    print()
    simulation_section2()
