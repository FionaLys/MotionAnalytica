'''from os import *
import os

file_paths = '/Users/markus/PycharmProjects/MotionAnalytica/data/All_throws'
files = listdir(file_paths)
counter = 1

for file in files:
    rename(file, str(counter) + '_' + file)
    counter += 1
'''