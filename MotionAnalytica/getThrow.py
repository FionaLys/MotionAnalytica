import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os


column_x = 'accelerometerAccelerationX(G)'
column_y = 'accelerometerAccelerationY(G)'
column_z = 'accelerometerAccelerationZ(G)'
timestamp = 'accelerometerTimestamp_sinceReboot(s)'
workmode = 0 #1 for whole folder, 0 for one file
dirPath = ""
count = 1


def findFile():
    x = 1
    while x == 1:
        global dirPath
        folder = input('Welcher Ordner soll verwendet werden werden? ')
        dirPath = "../data/" + folder + "/csv/"
        if not os.path.isdir(dirPath):
            print("Ordner existiert nicht")
        else:
            x = 0
            if workmode == 0:
                file = input('Welche Datei soll verwendet werden? ')
                file_list = file
                #files = dirPath + "/csv/" + file
            else:
                file_list = os.listdir(dirPath)
                #files = [dirPath+ "/csv/" + file for file in os.listdir(dirPath)]
                #print(files)
        #print(files)
        return file_list

def readCsv(file):
    #print(file)
    print(dirPath + file)
    if os.path.isfile(dirPath +  file):
        df = pd.read_csv(dirPath + file)
        # df.info()
        x = 0
        print("----" + file + "----")
    else:
        print("Falscher Dateinname")
    return df

def findThrow(df_raw):
    clap_pos = df_raw[df_raw[column_z].gt(5)].index[0]
    clap_neg = df_raw[df_raw[column_z].le(-5)].index[0]
    if clap_neg > clap_pos:
        clap = clap_pos
    else:
        clap = clap_neg
    dfThrow = df_raw[clap + 50:]
    throwStart = dfThrow[dfThrow[column_z].gt(5)].index[0] - 10
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
    csvDirPath = dirPath + "/throwCsv/"
    if not os.path.isdir(csvDirPath):
        os.mkdir(csvDirPath)
        print("csvThrow Ordner erstellt!")
    csvPath = csvDirPath + file[:-4] + "_" + "Beschleunigung (Wurf)" + ".csv"
    df.to_csv(os.PathLike(csvPath))
    if os.path.isfile(csvPath):
        print(str(count) + ": CSV erfolgreich gespeichert!")
    count += 1

def savePng(file):
    global count
    plotDirPath = dirPath + "/throwPlot/"
    if not os.path.isdir(plotDirPath):
        os.mkdir(plotDirPath)
        print("Plot-Ordner erstellt: " + plotDirPath)
    plotPath = plotDirPath + file[:-4] + "_" + "Beschleunigung" + ".png"
    plt.savefig(plotPath)
    if os.path.isfile(plotPath):
        print(str(count) + ": Plot erfolgreich gespeichert!")
    count += 1
    plt.close()

def lost():
    if workmode == 0:
        save(0)
    elif workmode == 1:
        filecount = 0
        for file in file:
            filecount += 1
            save(filecount)

def main():
    if workmode == 0:
        file = findFile()
        genPlot(findThrow(readCsv(file)), file)
        plt.show()

    #elif workmode == 1:


main()




