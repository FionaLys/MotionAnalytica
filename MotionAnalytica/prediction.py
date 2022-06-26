import numpy

from feature_enginieering import *



def predict_distance(predictor):
    curr_error = 1000000000
    for factor in range(0, 100):
        df['Predicted Distance'] = predictor * factor
        next_error = calculate_error()
        if next_error <= curr_error:
            curr_error = next_error
            optimal_factor = factor
    df['Predicted Distance'] = round(predictor * optimal_factor, 0 )



def add_error_per_throw():
    df['Error'] = df['Distance'] - df['Predicted Distance']

def calculate_error():
    return sum(abs(abs(df['Distance']) - abs(df['Predicted Distance'])))

def start_calculations(predictor):
    predict_distance(predictor)
    add_error_per_throw()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

def print_average_error():
    error = calculate_error()  # Rework
    weighted_error = error / len(df)
    print('The Average error is', round(weighted_error / 100, 1), 'meter')

def calculate_weighted_feature(predictor):                         #Calculates weight of feature in relation to target variable
    df['calculate_weighted_feature'] = df['Distance'] / predictor


print_feature_correlation()
#calculate_weighted_feature()
#print(sum(df['calculate_weighted_feature'] / len(df)))
#print(df)

start_calculations(df['Sum Abs Max Acceleration'])      #Best so far
print_average_error()
print(df)








