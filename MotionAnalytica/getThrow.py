import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os


column_x = 'accelerometerAccelerationX(G)'
column_y = 'accelerometerAccelerationY(G)'
column_z = 'accelerometerAccelerationZ(G)'
timestamp = 'accelerometerTimestamp_sinceReboot(s)'
x = 1
workmode = 1 #1 for whole folder, 0 for one file


while x == 1:
    folder = input('Welcher Ordner soll verwendet werden werden? ')
    dirPath = "../data/" + folder + "/csv/"
    plotDirPath = "../data/" + folder + "/throwPlot/"
    if not os.path.isdir(dirPath):
        print("Ordner existiert nicht")
    else:
        x = 0
        if workmode == 0:
            file = input('Welche Datei soll verwendet werden? ')
            files = file
        else:
            files = os.listdir(dirPath)
    print(files)

def save(filecount):
    filePath = dirPath + file
    print(filePath)
    if os.path.isfile(filePath):
        df = pd.read_csv(filePath)
        #df.info()
        x = 0
        print("----" + file + "----")
    else:
        print("Falscher Dateinname")
    try:
        clap_pos = df[df[column_z].gt(5)].index[0]
        clap_neg = df[df[column_z].le(-5)].index[0]
        if clap_neg > clap_pos:
            clap = clap_pos
        else:
            clap = clap_neg
        dfThrow = df[clap + 50:]
        throwStart = dfThrow[dfThrow[column_z].gt(5)].index[0] - 10
        throwEnd = throwStart + 60

        df = df[throwStart:throwEnd]
        modus = "Beschleunigung (Wurf)"
        plt.title(modus + ": " + file)
        plt.plot(df[timestamp], df[column_x], label='x')
        plt.plot(df[timestamp], df[column_y], label='y')
        plt.plot(df[timestamp], df[column_z], label='z')
        plt.legend()

        if not os.path.isdir(plotDirPath):
            os.mkdir(plotDirPath)
            print("Plot-Ordner erstellt: " + plotDirPath)
        plotPath = plotDirPath + file[:-4] + "_" + modus + ".png"
        plt.savefig(plotPath)
        if os.path.isfile(plotPath):
            print(str(filecount) + ": Plot erfolgreich gespeichert!")
        plt.close()

        # print(clap)
        # print(throwStart)
        # print(throwEnd)
    except:
        print("Plot konnte nicht erstellt werden!")




if workmode == 0:
    save(0)
elif workmode == 1:
    filecount = 0
    for file in files:
        filecount += 1
        save(filecount)






