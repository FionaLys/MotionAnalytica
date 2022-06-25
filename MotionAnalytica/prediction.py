from featureEnginieering import *


def predict_distance():
    df['Predicted Distance'] = df['Count Duration > Y'] * 1000

def add_error_per_throw():
    df['Error'] = df['Distance'] - df['Predicted Distance']

def calculate_error():
    return int(sum((df['Predicted Distance'].multiply(df['Predicted Distance'])**(1/2))))

def start_calculations():
    predict_distance()
    add_error_per_throw()

print_feature_correlation()
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)

start_calculations()
error = calculate_error()
weighted_error = int(error / len(df))
print('The Average error is', weighted_error / 100, 'meter')
print(df)
print(error)