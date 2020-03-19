import ScikitExtraTree as SET
import ScikitLinearRegression as SLR
import numpy as np
from matplotlib import pyplot as plt
import section5 as S5
import domain


if __name__ == '__main__':
    # trajectory
    trajectory = [S5.first_generation_set_one_step_system_transition(500),
                  S5.second_generation_set_one_step_system_transition(500)]

    plt.subplot(2, 2)
    for number, h in enumerate(trajectory):

        # horizon of Q
        q_N = 5

        # Q functions
        Q_tree = []
        Q_reg = []

        """
            TRAINING
        """
        # iteration N=1
        X, Y = S5.build_training_set(h, None, 0)

        # add new models
        Q_tree.append(SET.SkExtraTree(n_trees=50, n_min=2))
        Q_reg.append(SLR.SkLinearRegression())

        # train the models
        Q_tree[0].fit(X, Y)
        Q_reg[0].fit(X, Y)

        # iteration N>1
        for i in range(1, q_N):
            X_tree, Y_tree = S5.build_training_set(h, Q_tree[i-1], i)
            X_reg, Y_reg = S5.build_training_set(h, Q_reg[i-1], i)

            Q_tree.append(SET.SkExtraTree(n_trees=50, n_min=2))
            Q_reg.append(SLR.SkLinearRegression())

            Q_tree[i].fit(X_tree, Y_tree)
            Q_reg[i].fit(X_reg, Y_reg)

        """
            TEST
        """
        tree_error = []
        reg_error = []

        n = range(q_N)
        for i in n:
            tree_n_error = []
            reg_n_error = []

            for k in range(50):
                tuple = S5.second_generation_set_one_step_system_transition(1)[0]
                p, s = tuple[0]
                u = tuple[1]

                tree_n_error.append(abs(Q_tree[i].predict([[p, s, u]]) - Q_tree[i+1].predict([[p, s, u]])))
                reg_n_error.append(abs(Q_reg[i].predict([[p, s, u]]) - Q_reg[i+1].predict([[p, s, u]])))

            tree_error.append(np.mean(tree_n_error))
            reg_error.append(np.mean(reg_n_error))

        plt.subplot(number+1, 1)
        plt.plot(n, tree_error)

        plt.subplot(number+1, 2)
        plt.plot(n, reg_error)

    plt.show()

