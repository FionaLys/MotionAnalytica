import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

#Import File-names
from pandas import DataFrame

file_paths = '../data/2022_06_02_thirdTry/throwCsv'

#Data to be inserted into df
data = {}
feature_correlation = {}
feature_correlation_keys = feature_correlation.keys()
feature_correlation_values = feature_correlation.values()


#Base Functions of Feature Rating
def initialize_feature_rating():
    #Get filenames of throws
    files = [file for file in listdir(file_paths) if 'lost' not in file]        #Filters lost throws
    files_sorted_asc = sorted(files, key=lambda x: int(x.split('_')[0]))

    #Safe one df for every throw in list
    df_throws = []
    for file in files_sorted_asc:
        file_directory = '../data/2022_06_02_thirdTry/throwCsv/' + file
        df = pd.read_csv(file_directory)
        df_throws.append(df)

    #Extract throw distance out of file names
    throw_distance = [int(distance[distance.find('cm_')-4:distance.find('cm_')]) for distance in files_sorted_asc]

    #Insert into dictionary
    data['File Name'] = files_sorted_asc
    data['Throw Data Frames'] = df_throws
    data['Distance'] = throw_distance

def create_feature_rating():
    #Fill dataframe with data from dictionary
    df = pd.DataFrame()
    for column in data.keys():
        df[column] = data[column]
    return df

def insert_feature(feature_name, function):
    #Adds colum to FeatureDataFrame and fills it with feature
    new_column = []
    for i in df['Throw Data Frames']:
        new_column.append(function(i))
    df.insert(len(df.columns) - 1, feature_name, new_column)
    feature_correlation[feature_name] = calculate_correlation(df[feature_name])

def calculate_correlation(x):                   #Berechnung Korrelationskoeffizient
    y = df['Distance']
    correlation = np.corrcoef(x, y)
    return correlation[0][1]

def print_feature_correlation():
    #Sort Dict Descending
    feature_correlation_sorted = pd.DataFrame(list(zip(feature_correlation_keys, feature_correlation_values)), columns=['Name of Function', 'Korrelation'])
    feature_correlation_sorted = feature_correlation_sorted.sort_values('Korrelation', ascending = False)

    #Print Dict
    for k, v in zip(feature_correlation_sorted['Name of Function'], feature_correlation_sorted['Korrelation']):
        print(str(k) + ' korreliert zu ' + str(int(round((v * 100)))) + '% mit der Wurfweite.')

#Collection of Feature

def max_abs_acceleration_x(i):
    if abs(i['accelerometerAccelerationX(G)'].min()) > i['accelerometerAccelerationX(G)'].max():
        return abs(i['accelerometerAccelerationX(G)'].min())
    else:
        return i['accelerometerAccelerationX(G)'].max()

def max_abs_acceleration_y(i):
    if abs(i['accelerometerAccelerationY(G)'].min()) > i['accelerometerAccelerationY(G)'].max():
        return abs(i['accelerometerAccelerationY(G)'].min())
    else:
        return i['accelerometerAccelerationY(G)'].max()

def max_abs_acceleration_z(i):
    if abs(i['accelerometerAccelerationZ(G)'].min()) > i['accelerometerAccelerationZ(G)'].max():
        return abs(i['accelerometerAccelerationZ(G)'].min())
    else:
        return i['accelerometerAccelerationZ(G)'].max()

def sum_max_abs_acceleration(i):
    return max_abs_acceleration_x(i) + max_abs_acceleration_y(i) + max_abs_acceleration_z(i)

def min_acceleration_x(i):
    return i['accelerometerAccelerationX(G)'].min()

def min_acceleration_y(i):
    return i['accelerometerAccelerationY(G)'].min()

def min_acceleration_z(i):
    return i['accelerometerAccelerationZ(G)'].min()

def max_acceleration_x(i):
    return i['accelerometerAccelerationX(G)'].max()

def max_acceleration_y(i):
    return i['accelerometerAccelerationY(G)'].max()

def max_acceleration_z(i):
    return i['accelerometerAccelerationZ(G)'].max()

def sum_max_acceleration(i):
    return i['accelerometerAccelerationX(G)'].max() + \
           i['accelerometerAccelerationY(G)'].max() + \
           i['accelerometerAccelerationZ(G)'].max()

def mean_acceleration_x(i):
    return i['accelerometerAccelerationX(G)'].mean()

def mean_acceleration_y(i):
    return i['accelerometerAccelerationY(G)'].mean()

def mean_acceleration_z(i):
    return i['accelerometerAccelerationZ(G)'].mean()

def mean_acceleration_x_y(i):
    return (i['accelerometerAccelerationX(G)'].mean() + \
           i['accelerometerAccelerationY(G)'].mean())

def count_time_lg_x(i):
    return i['accelerometerAccelerationX(G)'][i['accelerometerAccelerationX(G)'] > 15].count()

def count_time_lg_y(i):
    return i['accelerometerAccelerationY(G)'][i['accelerometerAccelerationY(G)'] > 27].count()

def count_time_lg_z(i):
    return i['accelerometerAccelerationZ(G)'][i['accelerometerAccelerationZ(G)'] > 30].count()

def count_time_lg_yz(i):
    return count_time_lg_y(i) + count_time_lg_z(i)


#Intelligence
def predict_distance():
    df['Predicted Distance'] = df['Count Duration > Sum'] * 1000

def calculate_error():
    df['Error'] = df['Distance'] - df['Predicted Distance']


#Main-Method

##Create df with all throws and all features added to dictionary data
initialize_feature_rating()
df = create_feature_rating()

##Add Features to FeatureDataFrame

insert_feature('Max Acceleration X', max_acceleration_x)
insert_feature('Max Acceleration Y', max_acceleration_y)
insert_feature('Max Acceleration Z', max_acceleration_z)
insert_feature('Sum Max Acceleration', sum_max_acceleration)

insert_feature('Sum Mean Acceleration', mean_acceleration_x_y)
insert_feature('Mean Acceleration X', mean_acceleration_x)
insert_feature('Mean Acceleration Y', mean_acceleration_y)
insert_feature('Mean Acceleration Z', mean_acceleration_z)

insert_feature('Count Duration > X', count_time_lg_x)
insert_feature('Count Duration > Y', count_time_lg_y)
insert_feature('Count Duration > Z', count_time_lg_z)
insert_feature('Count Duration > Sum', count_time_lg_yz)

insert_feature('Min Acceleration X', min_acceleration_x)
insert_feature('Min Acceleration Y', min_acceleration_y)
insert_feature('Min Acceleration Z', min_acceleration_z)

insert_feature('Abs Max Acceleration X', max_abs_acceleration_x)
insert_feature('Abs Max Acceleration Y', max_abs_acceleration_y)
insert_feature('Abs Max Acceleration Z', max_abs_acceleration_z)
insert_feature('Sum Abs Max Acceleration', sum_max_abs_acceleration)

##Execution

print_feature_correlation()
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
predict_distance()
calculate_error()

error = int(sum((df['Predicted Distance'].multiply(df['Predicted Distance'])**(1/2))))
print(error)
weighted_error = int(error / len(df))
print('The Average error is', weighted_error / 100, 'meter')
print(df)

