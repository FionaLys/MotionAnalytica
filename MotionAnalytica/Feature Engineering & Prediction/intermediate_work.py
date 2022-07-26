import os

import sklearn.linear_model
from numpy import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from feature_enginieering import *

optimal_factors = []
features = {}
'''
NOT IN USE ANYMORE
'''


def predict_distance_single_feature(predictor):
    curr_error = 1000000000
    #x_train, x_test, y_train, y_test = train_test_split(predictor, df['Distance'], test_size=0.2)
    for factor in range(0, 10000):
        df['Predicted Distance'] = predictor * factor
        next_error = calculate_error()
        if next_error <= curr_error:
            curr_error = next_error
            optimal_factor = factor
        else:
            break
        df['Predicted Distance'] = round(predictor * optimal_factor, 0)
    optimal_factors.append(optimal_factor)





def add_error_per_throw():
    df['Error'] = df['Distance'] - df['Predicted Distance']

def calculate_error():
    return sum(abs(abs(df['Distance']) - abs(df['Predicted Distance'])))

def start_calculations(predictor):
    predict_distance_single_feature(predictor)
    add_error_per_throw()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

def print_average_error():
    error = calculate_error()  # Rework
    weighted_error = error / len(df)
    print('The Average error is', round(weighted_error / 100, 1), 'meter')

def calculate_weighted_feature(predictor):                         #Calculates weight of feature in relation to target variable
    df['calculate_weighted_feature'] = df['Distance'] / predictor

def print_features_below_x_meter(x):
    features_below_x = {}
    for feature in feature_names:
        start_calculations(df[feature])
        features[feature] = calculate_error() / len(df)
        if calculate_error() / len(df) < x * 100:
            features_below_x[feature] = round(calculate_error() / len(df) / 100, 1)
    features_below_x = {k: v for k, v in sorted(features_below_x.items(), key=lambda item: item[1])}       #Sort Features
    for feature, feature_error in features_below_x.items():
        print(feature, 'error is', feature_error, 'meter')

def safe_work():
    df.to_pickle('data.pkl')

def read_work():
    return pd.read_pickle('data.pkl')

def configure_view():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

#print_feature_correlation()
#calculate_weighted_feature()
#print(sum(df['calculate_weighted_feature'] / len(df)))
#print(df)
#print(df)

#print_features_below_x_meter(30)
#print(features)
#print(df)
#df.plot.scatter(x = 'Sum Abs Max Acceleration', y = 'Distance', alpha = 0.5)
#plt.show()
configure_view()
safe_work()
df = read_work()
#print(df['Distance'])







