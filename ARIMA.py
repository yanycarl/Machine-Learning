
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels
from scipy.linalg import svdvals
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv

plt.style.use('fivethirtyeight')


class ReadFile:
    matrix = []

    @classmethod
    def openFile(cls):
        with open('D:\\Data\\Data.csv', newline='\n') as csvFile:
            read = csv.reader(csvFile)
            i = 0
            for row in read:
                temp = []
                if i > 1:
                    for item in row:
                        temp.append(item)
                    cls.matrix.append(temp)
                i += 1
        cls.matrix = np.matrix(cls.matrix).A
        cls.matrix = np.transpose(cls.matrix)[3]
        cls.matrix = cls.matrix.astype(float)


ReadFile.openFile()
print(ReadFile.matrix)
y = ReadFile.matrix[0: 100]

print(y)

plt.plot(y)
plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore")  # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            print('ARIMA{}x{}1 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 1, 1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(type(results))
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.predict(0, 160, dynamic=False)

plt.plot(ReadFile.matrix, label='One-step ahead Forecast')
plt.plot(pred, label='One-step ahead Forecast')

plt.show()

