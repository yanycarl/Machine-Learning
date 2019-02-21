
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.kde import KDEUnivariate

file = pd.read_csv("D:\\DATA\\ML\\W6\\uk_IMD_2010.csv", sep=",")
data = file.iloc[:, [5]]
matrix_data = np.matrix(data).A
array_data = matrix_data.reshape(32482)

# Histogram Density of different bins
yh10, xh10 = np.histogram(array_data, 10, density=True)
yh30, xh30 = np.histogram(array_data, 30, density=True)
yh70, xh70 = np.histogram(array_data, 70, density=True)
plt.step(xh30, np.concatenate((np.zeros(1), yh30)), '--r', label='30-bin Histogram')
plt.step(xh10, np.concatenate((np.zeros(1), yh10)), '+b', label='10-bin Histogram')
plt.step(xh70, np.concatenate((np.zeros(1), yh70)), '*-g', label='70-bin Histogram')
plt.legend()
plt.show()

# Kernel distribution and rug lines
sns.distplot(array_data, bins=30, rug=True)
plt.show()

# Jitter Plot
jitter_data = []
np.random.seed(0)
for i in array_data:
    jitter_data.append(np.random.rand(1))
sns.scatterplot(array_data, np.matrix(jitter_data).A.reshape(32482))
plt.ylim((0, 1))
plt.show()

# box plot
sns.boxplot(data=array_data)
plt.show()

# Central Tendency
mean = np.mean(array_data)
median = np.median(array_data)
mode = sp.stats.mode(array_data).mode[0]
print("Mean:", mean, "Median:", median, "Mode:", mode)

#The plot
plt.plot([mean, mean], [0, 0.05], c='r', label="Mean")
plt.plot([median, median], [0, 0.05], c='g', label="Median")
plt.plot([mode, mode], [0, 0.05], c='b', label="Mode")
plt.legend()
plt.show()

# Variance
stand_var = np.var(array_data)  # biased
print("Biased Variance:", stand_var)

unbiased_stand_var = np.var(array_data, ddof=1)  # Unbiased
print("Unbiased Variance:", unbiased_stand_var)

# Skewness
stand_dev = sp.std(array_data)
skewness = sp.stats.moment(matrix_data, 3)/(stand_dev**3)
print("The Skewness:", skewness[0])

# KDE
kde = KDEUnivariate(array_data)
kde.fit(kernel="uni", fft=False)
support = kde.support
density = kde.density
plt.plot(support, density)
plt.show()

# Different kernel contraction
sns.kdeplot(array_data, kernel='gau', label='Gaussian Kernel')
sns.kdeplot(array_data, kernel='uni', label='Uniform Kernel')
sns.kdeplot(array_data, kernel='tri', label='Triangle Kernel')
plt.legend()
plt.show()

# Kurtosis
stand_dev = np.std(array_data)
kurtosis = sp.stats.moment(array_data, 4)/(stand_dev**4) - 3
print("kurtosis:", kurtosis)

# ECDF
cum_density = ECDF(array_data)
# plt.step(cum_density.x, cum_density.y)

p25 = np.percentile(array_data, 25)
p50 = np.percentile(array_data, 50)
p75 = np.percentile(array_data, 75)
p95 = np.percentile(array_data, 95)
p99 = np.percentile(array_data, 99)

plt.plot([0, p25, p25], [0.25, 0.25, 0], '-k')
plt.plot([0, p50, p50], [0.50, 0.50, 0], '-k')
plt.plot([0, p75, p75], [0.75, 0.75, 0], '-k')
plt.plot([0, p95, p95], [0.95, 0.95, 0], '-k')
plt.plot([0, p99, p99], [0.99, 0.99, 0], '-k')
plt.xticks((0,p25,p50,p75,p95,p99),('0','$P_{25}$','$P_{50}$','$P_{75}$','$P_{95}$','$P_{99}$'))
plt.show()

print(p25, p50, p75, p95, p99)

# IQR
IQR = p75 - p25
print("IQR: P75 - P25 = ", IQR)

# Tail
plt.step(cum_density.x, 1-cum_density.y)
plt.yscale('log')
plt.show()
