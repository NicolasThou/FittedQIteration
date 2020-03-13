import ScikitExtraTree as SET
import numpy as np
import section5 as S5
import domain


if __name__ == '__main__':
    # trajectory
    trajectory = S5.first_generation_set_one_step_system_transition(500)

    # horizon of Q
    q_N = 3

    # Q functions
    Q = []

    # iteration N=1
    X, Y = S5.build_training_set(trajectory, None, 0)
    Q.append(SET.SkExtraTree(n_trees=50, n_min=2))
    Q[0].fit(X, Y)

    # iteration N>1
    for i in range(1, q_N):
        X, Y = S5.build_training_set(trajectory, Q[i-1], i)
        Q.append(SET.SkExtraTree(n_trees=50, n_min=2))
        Q[i].fit(X, Y)

    p, s = domain.initial_state()
    u = -4
    r = Q[-1].predict([[p, s, u]])

    print("{} {}".format(p, s))
    print(r)
