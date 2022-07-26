import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

from yellowbrick.regressor import AlphaSelection
from yellowbrick.datasets import load_concrete
import sklearn.tree
from sklearn.linear_model import LinearRegression, PoissonRegressor, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


rmse_of_all_features = {}

#Organizational
def read_work():
    return pd.read_pickle('/Users/markus/PycharmProjects/MotionAnalytica/MotionAnalytica/Feature Engineering & Prediction/data.pkl')

def configure_view():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

def root_mean_squared_error(y_test, y_pred):        #Genutztes Fehlermaß
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (1 / 2)
    return rmse
#'File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower', 'Thrower1', 'Thrower5', 'Thrower4', 'Thrower6', 'Thrower3', 'Thrower2', 'Max Acceleration X', 'Count Duration > X', 'Mean Acceleration Y', 'Mean Acceleration Z', 'Min Acceleration Z'

def print_var_correlations():
    #data = df.drop(['File Name', 'Throw Data Frames', 'Predicted Distance', 'Error', 'Thrower', 'Thrower1', 'Thrower5', 'Thrower4', 'Thrower6', 'Thrower3', 'Thrower2'], axis=1)
    data = df.drop(['File Name', 'Throw Data Frames', 'Predicted Distance', 'Error', 'Thrower', 'Thrower1', 'Thrower5', 'Thrower4', 'Thrower6', 'Thrower3', 'Thrower2', 'Count Duration > Y', 'Count Duration > Z', 'Count Duration > Sum', 'Min Acceleration X', 'Abs Max Acceleration X', 'Minimal Acceleration from XYZ', 'Mean Acceleration Y', 'Mean Acceleration Z', 'Min Acceleration Z', 'Count Duration > X', 'Max Acceleration X'], axis=1)
    corr = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(corr.abs(), cmap='GnBu', vmin=0, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()



def calculate_error(y_test, y_pred):
    return root_mean_squared_error(y_test, y_pred)

def supervised_regression_algorithm(algorithm):
    #x = df.filter(['Sum Abs Max Acceleration', 'Mean Acceleration X'])
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    sra = algorithm()
    sra.fit(x_train, y_train)
    y_pred = sra.predict(x_test)
    error = calculate_error(y_test, y_pred)
    #print_prediction_vs_target(y_test, y_pred)
    #Show Test and Train Data
    plt.scatter(y_train, sra.predict(x_train), alpha=0.5, color='g', label = 'Train data')
    plt.scatter(y_test, y_pred, alpha=1, color = 'b', label = 'Test data')
    plt.plot(y_pred, sra.predict(x_test), color='k')            #PredictionAxis
    plt.legend()
    plt.title('')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.title(str(algorithm.__name__) + ' with Korrelation ' + str(round(np.corrcoef(x.values, y)[1][0] * 100)) + '%')
    plt.show()
    #plt_results(y_test, y_pred)
    return error

def execute_lasso(algorithm, alpha, max_iter):
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    sra = algorithm(max_iter = max_iter, alpha = alpha)
    sra.fit(x_train, y_train)
    y_pred = sra.predict(x_test)
    error = calculate_error(y_test, y_pred)
    #print(sra.coef_)
    #print_prediction_vs_target(y_test, y_pred)
    #Show Test and Train Data
    '''plt.scatter(y_train, sra.predict(x_train), alpha=0.2, color='g')
    plt.scatter(y_test, y_pred, alpha=1, color = 'b')
    plt.plot(y_pred, sra.predict(x_test), color='k')            #PredictionAxis
    plt.title('')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.title(str(algorithm.__name__) + ' with Korrelation ' + str(round(np.corrcoef(x.values, y)[1][0] * 100)) + '%')
    plt.show()'''
    #plt_results(y_test, y_pred)
    return error


def supervised_regression_algorithm_with_itterations(algorithm, itterations, alpha):
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    '''x = df.drop(
        ['Distance', 'File Name', 'Throw Data Frames', 'Predicted Distance', 'Error', 'Thrower', 'Thrower1', 'Thrower5',
         'Thrower4', 'Thrower6',
         'Thrower3', 'Thrower2', 'Count Duration > Y', 'Count Duration > Z', 'Count Duration > Sum', 'Min Acceleration X',
         'Abs Max Acceleration X', 'Minimal Acceleration from XYZ', 'Mean Acceleration Y', 'Mean Acceleration Z',
         'Min Acceleration Z', 'Count Duration > X', 'Max Acceleration X'], axis=1)'''
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    sra = algorithm(max_iter=itterations, alpha = alpha)
    sra.fit(x_train, y_train)
    y_pred = sra.predict(x_test)
    error = calculate_error(y_test, y_pred)
    #plt_results(y_test, y_pred)
    '''if algorithm == RandomForestRegressor:
        print_prediction_vs_target(y_test, y_pred)'''
    return error


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
        print(round(i), round(y))

def plt_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

def multiple_linear_regression():
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    ml = LinearRegression()
    ml.fit(x_train, y_train)

    y_pred = ml.predict(x_test)
    '''
    for i, y in zip(y_test, y_pred):
        print(i, y)'''

    for i, y, m in zip(x.columns, ml.coef_, x.mean()):
        print(i, y)

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
    dt = DecisionTreeRegressor(random_state = 3)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    error = calculate_error(y_test, y_pred)
    print(error)
    plt.scatter(y_test, y_pred)
    for i, y in zip(y_test, y_pred):
        print(i, y)

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    print(r2_score(y_test, y_pred))
    sklearn.tree.plot_tree(dt, feature_names = x.columns, fontsize = 8)
    plt.show()

def random_forest_regression():
    x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)
    rfr = RandomForestRegressor(random_state = 3)
    rfr.fit(x_train, y_train)
    y_pred = rfr.predict(x_test)
    error = calculate_error(y_test, y_pred)
    print(error)
    plt.scatter(y_test, y_pred)

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    sklearn.tree.plot_tree(rfr.estimators_[0], feature_names = x.columns, fontsize = 8)
    plt.show()
    sklearn.tree.plot_tree(rfr.estimators_[1], feature_names=x.columns, fontsize=8)
    plt.show()
    print(len(rfr.estimators_))


def single_feature_prediciton(feature):     #Predict with a single feature and plot test and train data
    x = feature.values.reshape(-1, 1)
    y = df['Distance']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    error = calculate_error(y_test, y_pred)

    #Show Test and Train Data


    plt.scatter(y_train, lr.predict(x_train), alpha=0.5, color='g', label = 'Train data')
    plt.scatter(y_test, y_pred, alpha=1, color = 'b', label = 'Test data')
    plt.plot(y_pred, lr.predict(x_test), color='k')            #PredictionAxis
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title(str(feature.name) + ' with Korrelation ' + str(round(np.corrcoef(feature.values, y)[1][0] * 100)) + '%')
    plt.show()
    #Print Infos
    '''print_prediction_vs_target(y_test, y_pred)
    print(rmse)'''
    return error

configure_view()
df = read_work()

def prediction_accuracy_all_single_feature():   #Print Accuracy with simple linear regression of all single feature
    dict = {}
    dict_score = {}
    for feature in df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1):
        rmse = single_feature_prediciton(df[feature])
        dict[rmse] = str(feature)
    for key in sorted(dict.keys()):
        print('RMS for', dict[key], 'is',  round(key/100,2), 'meter')
    '''
    for feature in df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1):
        score = cross_fold_validation_single_feature(LinearRegression(), df[feature])
        dict_score[score] = str(score)
    for key in sorted(dict_score.keys()):
        print(dict_score[key])'''


def test_lasso():
    result = []
    resulttime = []
    for algorithm in [Lasso]:
        for alpha in [1, 0.5]:
            start = datetime.now()
            sum_rmse = 0
            itterations = 100
            for i in range(itterations):
                rmse = execute_lasso(algorithm, max_iter=5000, alpha=alpha)
                sum_rmse += rmse
            avg_rmse = sum_rmse / itterations
            result.append(algorithm.__name__ + 'with alpha ' + str(alpha) + 'is:' + str(avg_rmse / 100))
            resulttime.append(str(datetime.now() - start))

    for i in range(len(result)):
        print(result[i])
        print(resulttime[i])

def find_optimal_alpha():
    # Load the regression dataset
    x, y = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1), \
           df['Distance']
    # Create a list of alphas to cross-validate against
    alphas = np.arange(0.001, 3, 0.001)
    # Instantiate the linear model and visualizer
    model = LassoCV(alphas=alphas)
    visualizer = AlphaSelection(model)
    visualizer.fit(x, y)
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    visualizer = AlphaSelection(model)
    visualizer.fit(x, y)
    visualizer.show()
print(df)
#find_optimal_alpha()

#print_var_correlations()
#prediction_accuracy_all_single_feature()
#supervised_regression_algorithm(LinearRegression)
#decision_tree_regression()
#print_var_correlations()
#random_forest_regression()
'''
for algorithm in [Ridge]:
    start = datetime.now()
    sum_rmse = 0
    itterations = 1000
    for i in range(itterations):
        rmse = supervised_regression_algorithm(algorithm)
        sum_rmse += rmse
    avg_rmse = sum_rmse / itterations
    print(algorithm.__name__, round(avg_rmse / 100, 2))
    print(datetime.now() - start)'''


'''for algorithm in [DecisionTreeRegressor, LinearRegression, RandomForestRegressor, SVR, Ridge]:
    sum_rmse = 0
    itterations = 1000
    for i in range(itterations):
        rmse = supervised_regression_algorithm(algorithm)
        sum_rmse += rmse
    avg_rmse = sum_rmse / itterations
    print(algorithm.__name__, round(avg_rmse / 100, 2))'''

#prediction_accuracy_all_single_feature()
#cross_fold_validation_single_feature(LinearRegression(), df['Mean Acceleration Z'])
#print(supervised_regression_algorithm(DecisionTreeRegressor))
#print(supervised_regression_algorithm(Lasso))
#print(execute_lasso(Lasso, 1, 1000))


#single_feature_prediciton(df['Sum Abs Max Acceleration'])
#supervised_regression_algorithm(Lasso)



'''
supervised_regression_algorithm(DecisionTreeRegressor)
supervised_regression_algorithm(LinearRegression)
supervised_regression_algorithm(RandomForestRegressor)
supervised_regression_algorithm(SVR)'''



'''
for feature in df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1):
    start = datetime.now()
    sum_rmse = 0
    itterations = 100
    for i in range(itterations):
        rmse = single_feature_prediciton(df[feature])
        sum_rmse += rmse
    avg_rmse = sum_rmse / itterations
    print(feature, round(avg_rmse / 100, 2))
    print(datetime.now() - start)'''


for algorithm in [LinearRegression]:    #
    start = datetime.now()
    sum_rmse = 0
    itterations = 1000
    for i in range(itterations):
        rmse = supervised_regression_algorithm(algorithm)
        sum_rmse += rmse
    avg_rmse = sum_rmse / itterations
    print(algorithm.__name__, round(avg_rmse / 100, 2))
    print(datetime.now() - start)

for algorithm in [Lasso, Ridge]:
    start = datetime.now()
    sum_rmse = 0
    itterations = 1000
    for i in range(itterations):
        rmse = supervised_regression_algorithm_with_itterations(algorithm, 7000, alpha = 0.418)
        sum_rmse += rmse
    avg_rmse = sum_rmse / itterations
    print(algorithm.__name__, round(avg_rmse / 100, 2))
    print(datetime.now() - start)

#rmse_of_all_features = {k: v for k, v in sorted(rmse_of_all_features.items(), key=lambda item: item[1])}




'''x = df.drop(['File Name', 'Throw Data Frames', 'Distance', 'Predicted Distance', 'Error', 'Thrower'], axis=1)
y = df['Distance']
x_train, x_test, y_train, y_test = train_test_split(x, y)
sra = LassoCV()
sra.fit(x_train, y_train)
y_pred = sra.predict(x_test)
error = calculate_error(y_test, y_pred)
print(error)
model = AlphaSelection(RidgeCV())
model.fit(x, y)'''
#model.show()




