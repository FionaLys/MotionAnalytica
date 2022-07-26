import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np



from pandas import DataFrame

file_paths = '../data/All_throws/throwCsv/'

#Data to be inserted into df
data = {}
feature_correlation = {}
feature_correlation_keys = feature_correlation.keys()
feature_correlation_values = feature_correlation.values()

def configure_view():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

#Base Functions of Feature Rating
def initialize_feature_rating():

    #Get filenames of throws
    files = [file for file in listdir(file_paths) if 'lost' not in file]        #Filters lost throws
    print(files[0].split('_')[0])
    print(files[1].split('_')[0])
    files_sorted_asc = sorted(files, key=lambda x: int(x.split('_')[0]))
    print(files_sorted_asc)

    #Safe one df for every throw in list
    df_throws = []
    for file in files_sorted_asc:
        file_directory = file_paths + file
        df = pd.read_csv(file_directory)
        df_throws.append(df)

    def find_nth(file, name, n):
        start = file.find(name)
        while start >= 0 and n > 1:
            start = file.find(name, start + len(name))
            n -= 1
        return start

    #Extract throw distance out of file names
    throw_distance = [int(distance[distance.find('cm_')-4:distance.find('cm_')]) for distance in files_sorted_asc]
    thrower = [file[find_nth(file, '_', 2) + 1:find_nth(file, '_', 3)] for file in files_sorted_asc]

    #Insert into dictionary
    data['File Name'] = files
    data['Throw Data Frames'] = df_throws
    data['Distance'] = throw_distance
    data['Thrower'] = thrower

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
        print(str(k) + ' ' + str(int(round((v * 100)))))

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

def sum_min_acceleration(i):
    return min_acceleration_x(i) + min_acceleration_y(i) + min_acceleration_z(i)

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

def motionRotationRateX(i):
    return i['motionRotationRateX(rad/s)'].mean()

def motionQuaternionX(i):
    return i['motionQuaternionX(R)'].mean()

def motionUserAccelerationX(i):
    return i['motionUserAccelerationX(G)'].mean()

def motionGravityX(i):
    return i['motionGravityX(G)'].mean()






def min_acceleration(i):
    return min([min_acceleration_x(i), min_acceleration_y(i), min_acceleration_z(i)])

def person():
    throwers = list(df['Thrower'].unique())
    for thrower in throwers:
        df[thrower] = np.where(df['Thrower'] == thrower, 1, 0)


#Main-Method

##Create df with all throws and all features added to dictionary data
initialize_feature_rating()
df = create_feature_rating()

##Add Features to FeatureDataFrame

feature_names = ['Max Acceleration X', 'Max Acceleration Y', 'Max Acceleration Z', 'Sum Max Acceleration',
                 'Sum Mean Acceleration', 'Mean Acceleration X', 'Mean Acceleration Y', 'Mean Acceleration Z',
                 'Count Duration > X', 'Count Duration > Y', 'Count Duration > Z', 'Count Duration > Sum',
                 'Min Acceleration X', 'Min Acceleration Y', 'Min Acceleration Z', 'Sum Min Acceleration',
                 'Abs Max Acceleration X', 'Abs Max Acceleration Y', 'Abs Max Acceleration Z', 'Sum Abs Max Acceleration',
                 'Minimal Acceleration from XYZ']
                #'Motion Rotation Rate X', 'Motion Quaternion X', 'Motion User Acceleration X', 'Motion GravityX'


feature_functions = [max_acceleration_x, max_acceleration_y, max_acceleration_z, sum_max_acceleration,
                     mean_acceleration_x_y, mean_acceleration_x, mean_acceleration_y, mean_acceleration_z,
                     count_time_lg_x, count_time_lg_y, count_time_lg_z, count_time_lg_yz,
                     min_acceleration_x, min_acceleration_y, min_acceleration_z, sum_min_acceleration,
                     max_abs_acceleration_x, max_abs_acceleration_y, max_abs_acceleration_z, sum_max_abs_acceleration,
                     min_acceleration]#,
                     #motionRotationRateX, motionQuaternionX, motionUserAccelerationX, motionGravityX

for x, y in zip(feature_names, feature_functions):
    insert_feature(x, y)

#person()
print_feature_correlation()
configure_view()
#print(df)


##Execution





