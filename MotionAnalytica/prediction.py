from numpy import *

from feature_enginieering import *



def predict_distance(predictor):
    curr_error = 1000000000
    for factor in range(0, 100):
        df['Predicted Distance'] = predictor * factor
        next_error = calculate_error()
        if next_error <= curr_error:
            curr_error = next_error
            optimal_factor = factor
        df['Predicted Distance'] = round(predictor * optimal_factor, 0)
    '''
    if curr_error > 1000:
        for factor in range(0, 1000):
            df['Predicted Distance'] = predictor * factor
            next_error = calculate_error()
            if next_error <= curr_error:
                curr_error = next_error
                optimal_factor = factor
        df['Predicted Distance'] = round(predictor * optimal_factor, 0)'''





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

def print_features_below_x_meter(x):
    features_below_x = {}
    for feature in feature_names:
        start_calculations(df[feature])  # Best so far
        if calculate_error() / len(df) < x * 100:
            features_below_x[feature] = round(calculate_error() / len(df) / 100, 1)
    features_below_x = {k: v for k, v in sorted(features_below_x.items(), key=lambda item: item[1])}
    for feature, feature_error in features_below_x.items():
        print(feature, 'error is', feature_error, 'meter')



#print_feature_correlation()
#calculate_weighted_feature()
#print(sum(df['calculate_weighted_feature'] / len(df)))
#print(df)
#print(df)

print_features_below_x_meter(10)








