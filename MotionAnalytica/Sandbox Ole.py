import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


column_x = 'accelerometerAccelerationX(G)'
column_y = 'accelerometerAccelerationY(G)'
column_z = 'accelerometerAccelerationZ(G)'
timestamp = 'accelerometerTimestamp_sinceReboot(s)'
dfEndValues = []
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

        indicator = df[df[column_x].le(-5)].index[0]
        dfIntegration = df.loc[indicator:]#.reset_index(drop=True)
        integrationEnd = dfIntegration[dfIntegration[column_x].gt(-5)].index[0]
        df = df.loc[:integrationEnd]

        #df[timestamp] = df[timestamp] - df[timestamp][0]

        df['geschw_x'] = np.cumsum(df[column_x] * 1/100) #(df[timestamp][1] - df[timestamp][0]))
        df['geschw_y'] = np.cumsum(df[column_y] * 1/100) #(df[timestamp][1] - df[timestamp][0]))
        df['geschw_z'] = np.cumsum(df[column_z] * 1/100) #(df[timestamp][1] - df[timestamp][0]))


        endValue = df.loc[integrationEnd]['geschw_x']

        global dfEndValues

        throwDist = file[-16:-12]
        dfEndValues = dfEndValues + [throwDist, endValue]

        modus = "Geschwindigkeit (Wurf)"
        plt.title(modus + ": " + file)
        plt.plot(df[timestamp], df['geschw_x'], label='x')
        plt.plot(df[timestamp], df['geschw_y'], label='y')
        plt.plot(df[timestamp], df['geschw_z'], label='z')
        plt.legend()

        if not os.path.isdir(plotDirPath):
            os.mkdir(plotDirPath)
            print("Plot-Ordner erstellt: " + plotDirPath)
        plotPath = plotDirPath + file[:-4] + "_" + "Integrated" + ".png"
        plt.savefig(plotPath)
        if os.path.isfile(plotPath):
            print(str(filecount) + ": Plot erfolgreich gespeichert!")
        plt.close()

        # print(clap)
        # print(throwStart)
        # print(throwEnd)
    except Exception as E:
        print("Plot konnte nicht erstellt werden!")




if workmode == 0:
    save(0)
elif workmode == 1:
    filecount = 0
    for file in files:
        filecount += 1
        save(filecount)

    print('x')
    print('y')






