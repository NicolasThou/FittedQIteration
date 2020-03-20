import section5 as s5
import ScikitLinearRegression as SLR
import ScikitExtraTree as SET
from joblib import dump, load


if __name__ == '__main__':
    for T in range(100, 1100, 100):
        print('T = {}'.format(T))
        """
            First generation system
        """
        F = s5.first_generation_set_one_step_system_transition(T)

        """
            First stopping rule
        """
        reg_models = s5.fitted_Q_iteration_first_stopping_rule(F, SLR.SkLinearRegression())
        dump(reg_models, 'models/regression_{}_first_1.joblib'.format(T))
        tree_models = s5.fitted_Q_iteration_first_stopping_rule(F, SET.SkExtraTree())
        dump(tree_models, 'models/tree_{}_first_1.joblib'.format(T))

        """
            Second stopping rule
        """
        reg_models = s5.fitted_Q_iteration_second_stopping_rule(F, SLR.SkLinearRegression())
        dump(reg_models, 'models/regression_{}_first_2.joblib'.format(T))
        tree_models = s5.fitted_Q_iteration_second_stopping_rule(F, SET.SkExtraTree())
        dump(tree_models, 'models/tree_{}_first_2.joblib'.format(T))


        """
            Second generation
        """
        F = s5.second_generation_set_one_step_system_transition(T)

        """
            First stopping rule
        """
        reg_models = s5.fitted_Q_iteration_first_stopping_rule(F, SLR.SkLinearRegression())
        dump(reg_models, 'models/regression_{}_second_1.joblib'.format(T))
        tree_models = s5.fitted_Q_iteration_first_stopping_rule(F, SET.SkExtraTree())
        dump(tree_models, 'models/tree_{}_second_1.joblib'.format(T))

        """
            Second stopping rule
        """
        reg_models = s5.fitted_Q_iteration_second_stopping_rule(F, SLR.SkLinearRegression())
        dump(reg_models, 'models/regression_{}_second_2.joblib'.format(T))
        tree_models = s5.fitted_Q_iteration_second_stopping_rule(F, SET.SkExtraTree())
        dump(tree_models, 'models/tree_{}_second_2.joblib'.format(T))


