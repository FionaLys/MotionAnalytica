import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os


column_x = 'accelerometerAccelerationX(G)'
column_y = 'accelerometerAccelerationY(G)'
column_z = 'accelerometerAccelerationZ(G)'
timestamp = 'accelerometerTimestamp_sinceReboot(s)'


df = pd.read_csv("32_Nick_2800cm_100Hz.csv")

clap = df[df[column_z].gt(5)].index[0]
dfThrow = df[clap + 50:]
throwStart = dfThrow[dfThrow[column_z].gt(5)].index[0] - 10
throwEnd = throwStart + 60

df = df[throwStart:throwEnd]

modus = "Beschleunigung"
plt.title(modus + ": ")
plt.plot(df[timestamp], df[column_x], label='x')
plt.plot(df[timestamp], df[column_y], label='y')
plt.plot(df[timestamp], df[column_z], label='z')

plt.legend()
plt.show()

print(clap)
print(throwStart)
print(throwEnd)


