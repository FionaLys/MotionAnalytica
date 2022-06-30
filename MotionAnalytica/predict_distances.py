import pandas as pd
import sklearn.tree
from sklearn.linear_model import LinearRegression, PoissonRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

#Organizational
def read_work():
    return pd.read_pickle('data.pkl')

def configure_view():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

def supervised_regression_algorithm(algorithm):
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    sra = algorithm()
    sra.fit(x_train, y_train)
    y_pred = sra.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2)
    #plt_results(y_test, y_pred)
    '''if algorithm == RandomForestRegressor:
        print_prediction_vs_target(y_test, y_pred)'''
    print('Root Mean Square Error:', rmse, '-->', algorithm.__name__)

def supervised_regression_algorithm_with_itterations(algorithm, itterations):
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    sra = algorithm(max_iter=itterations)
    sra.fit(x_train, y_train)
    y_pred = sra.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2)
    #plt_results(y_test, y_pred)
    '''if algorithm == RandomForestRegressor:
        print_prediction_vs_target(y_test, y_pred)'''
    print('Root Mean Square Error:', rmse, '-->', algorithm.__name__)

def cross_fold_validation_multiple_feature(algorithm):
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    dt_result = cross_validate(algorithm, x, y, cv=5, return_train_score=True)
    return dt_result['test_score'].mean()

def cross_fold_validation_single_feature(algorithm, feature):
    x = feature.values.reshape(-1, 1)
    y = df['Distance']
    dt_result = cross_validate(algorithm, x, y, cv=5, return_train_score=True)
    return dt_result['test_score'].mean()

def print_prediction_vs_target(y_test, y_pred):
    for i, y in zip(y_test, y_pred):
        print(i, y)

def plt_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

def multiple_linear_regression(x):
    #x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    ml = LinearRegression()
    ml.fit(x_train, y_train)

    y_pred = ml.predict(x_test)
    '''
    for i, y in zip(y_test, y_pred):
        print(i, y)'''

    for i, y, m in zip(x.columns, ml.coef_, x.mean()):
        print(i, y / m)

    print(mean_squared_error(y_test, y_pred)**(1/2))
    print(r2_score(y_test, y_pred))

    # plt.figure(figsize = (15, 10))
    plt.scatter(y_test, y_pred)

    plt.xlabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Actual vs Predicted')

    plt.show()

def decision_tree_regression():
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)
    dt = DecisionTreeRegressor(max_depth = 4, random_state = 3)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    mse_dt = mean_squared_error(y_test, y_pred)
    rmse_dt = mse_dt**(1/2)
    print(rmse_dt)
    plt.scatter(y_test, y_pred)
    for i, y in zip(y_test, y_pred):
        print(i, y)

    plt.xlabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    print(r2_score(y_test, y_pred))
    #sklearn.tree.plot_tree(dt, feature_names = x.columns, fontsize = 8)
    #plt.show()


def single_feature_prediciton(feature):
    x = feature.values.reshape(-1, 1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (1 / 2)

    #Print Infos
    '''print_prediction_vs_target(y_test, y_pred)
    print(rmse)'''
    return rmse

#Main-Method
configure_view()
df = read_work()

def prediction_accuracy_all_single_feature():
    dict = {}
    dict_score = {}
    for feature in df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1):
        rmse = single_feature_prediciton(df[feature])
        dict[rmse] = 'Root Mean Sqaured Error for ' + str(feature) + ' is ' + str(round(rmse / 100, 5)) + ' meter'
    for key in sorted(dict.keys()):
        print(dict[key])
    for feature in df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1):
        score = cross_fold_validation_single_feature(LinearRegression(), df[feature])
        dict_score[score] = 'Score for  ' + str(feature) + ' is ' + str(score)
    for key in sorted(dict_score.keys()):
        print(dict_score[key])


#multiple_linear_regression()
prediction_accuracy_all_single_feature()
#mean_prediction_accuracy()

#single_feature_prediciton(df['Mean Acceleration Z'])
#cross_fold_validation_single_feature(LinearRegression(), df['Mean Acceleration Z'])

supervised_regression_algorithm(DecisionTreeRegressor)
supervised_regression_algorithm(LinearRegression)
supervised_regression_algorithm(RandomForestRegressor)
supervised_regression_algorithm(SVR)

print(cross_fold_validation_multiple_feature(DecisionTreeRegressor()))
print(cross_fold_validation_multiple_feature(LinearRegression()))
print(cross_fold_validation_multiple_feature(RandomForestRegressor()))
print(cross_fold_validation_multiple_feature(SVR()))


supervised_regression_algorithm_with_itterations(PoissonRegressor, 10000)
supervised_regression_algorithm_with_itterations(Lasso, 10000)




