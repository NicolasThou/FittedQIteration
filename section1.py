import numpy as np

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
    p, s = x[0], x[1]
    new_p = p
    new_s = s

    # check which Hill function to use
    if p < 0:
        g = lambda g_p, g_s, g_u: derivee_seconde_p_inf_0(g_p, g_s, g_u)
    else:
        g = lambda g_p, g_s, g_u: derivee_seconde_p_sup_equal_0(g_p, g_s, g_u)

    # euler method with 1000 step
    for i in range(1000):
        temp_p = new_p  # we keep the previous value
        temp_s = new_s  # we keep the previous value

        # we use the values of the previous step, not the new ones !!
        new_p = temp_p + integration_step * temp_s
        new_s = temp_s + integration_step * g(temp_p, temp_s, u)

    return np.array([new_p, new_s])


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
    
    next_state = f(x, u) # use of the dynamic of the problem
    p_suivant = next_state[0]
    s_suivant = next_state[1]
    if p_suivant < -1 or np.abs(s_suivant) > 3:
        return -1
    elif p_suivant > 1 and np.abs(s_suivant) <= 3:
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

    a = (u/m - (k * (g * s + 2 * (s ** 5)))) / (1 + (s ** 2) * (k ** 2))
    b = 1 + (((s ** 3) * (k ** 2)) / (1 + (s ** 2) * (k ** 2)))
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
    numerateur3 = ((15 * (s ** 5) * p) / (1 + 5 * (p ** 2))) * (((1 / (np.sqrt((1 + 5 * (p ** 2))))) * (1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2))))) ** 2)
    denominateur_commun2 = 1 + (((1 / (np.sqrt((1 + 5 * (p ** 2))))) * (1 - ((5 * (p ** 2)) / (1 + 5 * (p ** 2))))) ** 2) * (s ** 3)

    if denominateur_commun == 0 or denominateur_commun2 == 0:
        # error to handle in the future
        print('Denominator equal zero !!')
        return None

    return ((numerateur1 - numerateur2 + numerateur3) / denominateur_commun) / denominateur_commun2


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
    if action == 4:  # we change the action, we accelerate then slow down etc...
        return -4
    else:
        return 4



def simulation_section2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    print(state)
    for i in range(50):
        action = random_policy()  # use a random policy
        print(action)
        state = f(state, action)  # use the dynamic of the domain
        print(state)
        if is_final_state(state) == True:
            print('Nous avons atteint un état finale')
            return None


def simulation_section2_2():
    """
    Simulate the policy in the domain from an initial state and display the trajectory
    """
    state = initial_state()
    print(state)
    action = -4  # we begin with an acceleration
    for i in range(50):
        action = policy_alternative(action)
        state = f(state, action)  # use the dynamic of the domain
        print(state)
        if is_final_state(state) == True:
            print('Nous avons atteint un état finale')
            return None

if __name__ == '__main__':
    assert is_final_state(np.array([-2, 0]))
    assert is_final_state(np.array([0, 5]))
    print()
    print('Use of alternativ policy')
    print('If we change between accelerate and slow down each time, we won\'t reach a final state')
    simulation_section2_2()
    print()
    print('Use of random policy')
    print('If we accelerate two times, the car is too fast, so we reach a final state')
    simulation_section2()

    print("test")