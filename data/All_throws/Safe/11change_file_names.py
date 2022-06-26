from os import *
import os

file_paths = '/Users/markus/PycharmProjects/MotionAnalytica/data/All_throws'
files = listdir(file_paths)

print(files)'''
for file in files:
    counter = 1
    rename(file, str(counter) + file)
    counter += 1
'''