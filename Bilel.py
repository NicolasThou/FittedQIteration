import numpy as np
from matplotlib import pyplot as plt


U = np.array([-4, 4])  # action space
m = 1
g = 9.81
gamma = 0.95
delta_t = 0.1  # discretization time
integration_step = 0.001  # euler method step


def f(x, u):
    """
    dynamic of the system
    """

    # initialization
    p, s = x
    new_p = p
    new_s = s

    # euler method with 1000 step (h=0.001)
    for i in range(1000):
        temp_p = new_p  # we keep the previous value
        temp_s = new_s  # we keep the previous value

        # we use the values of the previous step, not the new ones !!
        new_p = temp_p + integration_step * temp_s
        new_s = temp_s + integration_step * f_s(temp_p, temp_s, u)

    return np.array([new_p, new_s])


def f_s(p, s, u):
    if p < 0:
        dH = 2*p + 1
        ddH = 2
    else:
            # recurrent term
        k = 1 + 5 * (p ** 2)
            # Hill'(p)
        dH = 1 / (np.sqrt(k) ** 3)
            # Hill''(p)
        ddH = -(15*p*np.sqrt(k))/(k**3)

    first_term = u / (m * (1 + dH ** 2))
    second_term = -(g * dH) / (1 + dH ** 2)
    third_term = -((s ** 2) * dH * ddH) / (1 + dH ** 2)

    return first_term + second_term + third_term


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

    next_state = f(x, u)  # use of the dynamic of the problem
    next_p, next_s = next_state
    if next_p < -1 or np.abs(next_s) > 3:
        return -1
    elif next_p > 1 and np.abs(next_s) <= 3:
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


def random_policy():
    """
    return, in a randomly way, 4 or -4

    Return:
    ======
    return an action
    """
    power = np.random.randint(0, 2)
    return ((-1) ** power) * 4


def policy_alternative(action):
    """
    take the previous action, if the previous action was accelerate, then the policy
    return -4, but if the previous action was to slow down, then return +4
    """
    return -action


def hill(p):
    if p < 0:
        return (p**2) + p
    else:
        return p/(np.sqrt(1 + 5*(p**2)))


def simulation_section2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    p = []
    s = []
    for i in range(50):
        action = random_policy()  # use a random policy
        print(action)
        state = f(state, action)  # use the dynamic of the domain
        p.append(state[0])
        s.append(state[1])
        if is_final_state(state) == True:
            print('Nous avons atteint un état finale')
            break
    fig, axis = plt.subplots()
    axis.plot(p, s, '+')
    # axis.set(xlim=(-1, 1), ylim=(-3, 3))
    for i in range(len(p)):
        axis.annotate(str(i), (p[i], s[i]))
    plt.show()


def simulation_section2_2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    print(state)
    action = -4
    for i in range(50):
        action = policy_alternative(action)
        print(action)
        state = f(state, action)  # use the dynamic of the domain
        print(state)
        if is_final_state(state) == True:
            print('Nous avons atteint un état finale')
            return None


def disp_hill():
    x = []
    y = []
    for p in range(-100, 100, 1):
        pos = (p*1.0)/100
        x.append(pos)
        y.append(hill(pos))

    fig, axis = plt.subplots()
    axis.plot(x, y, '+')
    axis.set(xlim=(-1, 1), ylim=(-0.5, 0.5))
    plt.show()


if __name__ == '__main__':
    assert is_final_state(np.array([-2, 0]))
    assert is_final_state(np.array([0, 5]))

    simulation_section2()

