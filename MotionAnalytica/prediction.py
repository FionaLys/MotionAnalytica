from featureEnginieering import *


def predict_distance():
    df['Predicted Distance'] = df['Count Duration > Sum'] * 1000

def add_error_per_throw():
    df['Error'] = df['Distance'] - df['Predicted Distance']

def calculate_error():
    return int(sum((df['Predicted Distance'].multiply(df['Predicted Distance'])**(1/2))))
def start_calculations():
    predict_distance()
    add_error_per_throw()


start_calculations()
error = calculate_error()
weighted_error = int(error / len(df))
print('The Average error is', weighted_error / 100, 'meter')