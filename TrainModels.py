from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump
import section5 as s5


if __name__ == '__main__':
    T = 1000
    bsize = 10
    ep = 100

    print('First generation - first stopping rule')
    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, LinearRegression())
    dump(models, 'models/regression_first_1.joblib')

    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, ExtraTreesRegressor(n_estimators=50))
    dump(models, 'models/tree_first_1.joblib')

    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, s5.baseline_model(), batch_size=bsize, epoch=ep)
    dump(models, 'models/neural_net_first_1.joblib')



    print('First generation - second stopping rule')
    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, LinearRegression())
    dump(models, 'models/regression_first_2.joblib')

    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, ExtraTreesRegressor(n_estimators=50))
    dump(models, 'models/tree_first_2.joblib')

    F = s5.first_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, s5.baseline_model(), batch_size=bsize, epoch=ep)
    dump(models, 'models/neural_net_first_2.joblib')



    print('Second generation - first stopping rule')
    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, LinearRegression())
    dump(models, 'models/regression_second_1.joblib')

    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, ExtraTreesRegressor(n_estimators=50))
    dump(models, 'models/tree_second_1.joblib')

    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_first_stopping_rule(F, s5.baseline_model(), batch_size=bsize, epoch=ep)
    dump(models, 'models/neural_net_second_1.joblib')



    print('Second generation - second stopping rule')
    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, LinearRegression())
    dump(models, 'models/regression_second_2.joblib')

    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, ExtraTreesRegressor(n_estimators=50))
    dump(models, 'models/tree_second_2.joblib')

    F = s5.second_generation_set_one_step_system_transition(T)
    models = s5.fitted_Q_iteration_second_stopping_rule(F, s5.baseline_model(), batch_size=bsize, epoch=ep)
    dump(models, 'models/neural_net_second_2.joblib')


