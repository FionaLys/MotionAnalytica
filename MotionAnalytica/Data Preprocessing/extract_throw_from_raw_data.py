import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os

### Mode-Adjuster ###

filemode = 1  # 1 for whThrower6 folder, 0 for one file
savemode = "csv"  # "plot" for saving plots, "csv" for saving csvs

######


column_x = 'accelerometerAccelerationX(G)'
column_y = 'accelerometerAccelerationY(G)'
column_z = 'accelerometerAccelerationZ(G)'
timestamp = 'accelerometerTimestamp_sinceReboot(s)'
dirPath = ""
count = 1


def findFile(filemode):
    x = 1
    while x == 1:
        global dirPath
        folder = input('Welcher Ordner soll verwendet werden werden? ')
        dirPath = "../data/" + folder + "/csv/"
        if not os.path.isdir(dirPath):
            print("Ordner existiert nicht")
        else:
            x = 0
            if filemode == 0:
                file = input('Welche Datei soll verwendet werden? ')
                file_list = file
                # files = dirPath + "/csv/" + file
            else:
                file_list = os.listdir(dirPath)
                # files = [dirPath+ "/csv/" + file for file in os.listdir(dirPath)]
                # print(files)
        # print(files)
        return file_list


def readCsv(file):
    # print(file)
    print(dirPath + file)
    if os.path.isfile(dirPath + file):
        df = pd.read_csv(dirPath + file)
        # df.info()
        x = 0
        print("----" + file + "----")
    else:
        print("Falscher Dateinname")
    return df


def findThrow(df_raw):
    df_raw[column_x] = df_raw[column_x] * 9.81
    df_raw[column_y] = df_raw[column_y] * 9.81
    df_raw[column_z] = df_raw[column_z] * 9.81
    clap = df_raw[df_raw[column_z].abs().gt(30)].index[0]
    dfThrow = df_raw[clap + 50:]
    throwStart = dfThrow[dfThrow[column_z].abs().gt(30)].index[0] - 10
    throwEnd = throwStart + 60
    df = df_raw[throwStart:throwEnd]
    return df


def genPlot(df, file):
    modus = "Beschleunigung (Wurf)"
    plt.title(modus + ": " + file)
    plt.plot(df[timestamp], df[column_x], label='x')
    plt.plot(df[timestamp], df[column_y], label='y')
    plt.plot(df[timestamp], df[column_z], label='z')
    plt.legend()


def saveCsv(df, file):
    global count
    csvDirPath = dirPath[:-5] + "/throwCsv/"
    if not os.path.isdir(csvDirPath):
        os.mkdir(csvDirPath)
        print("csvThrow Ordner erstellt!")
    csvPath = csvDirPath + file[:-4] + "_" + "Beschleunigung_(Wurf)" + ".csv"
    df.to_csv(csvPath)
    if os.path.isfile(csvPath):
        print(str(count) + ": CSV erfolgreich gespeichert!")
    count += 1


def savePng(file):
    global count
    plotDirPath = dirPath[:-5] + "/throwPlot/"
    if not os.path.isdir(plotDirPath):
        os.mkdir(plotDirPath)
        print("Plot-Ordner erstellt: " + plotDirPath)
    plotPath = plotDirPath + file[:-4] + "_" + "Beschleunigung" + ".png"
    plt.savefig(plotPath)
    if os.path.isfile(plotPath):
        print(str(count) + ": Plot erfolgreich gespeichert!")
    count += 1
    plt.close()


def main():
    if filemode == 0:
        try:
            if savemode == "csv":
                file = findFile(filemode)
                saveCsv(findThrow(readCsv(file)), file)
            elif savemode == "plot":
                file = findFile(filemode)
                genPlot(findThrow(readCsv(file)), file)
                savePng(file)
                plt.close()
        except Exception as e:
            print (e)

    elif filemode == 1:
        files = findFile(filemode)
        if savemode == "csv":
            for file in files:
                try:
                    saveCsv(findThrow(readCsv(file)), file)
                except Exception as e:
                    print(e)
        elif savemode == "plot":
            for file in files:
                try:
                    genPlot(findThrow(readCsv(file)), file)
                    savePng(file)
                    plt.close()
                except Exception as e:
                    print(e)


main()
