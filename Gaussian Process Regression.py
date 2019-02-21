
import pandas as pd
import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Read file and get data
data = pd.read_csv("D:\\Data\\Ml\\w5\\revised_all_GP_training.csv")
dates1 = np.zeros((24, 1))
for i in range(0, 24):
    dates1.put(indices=[i, 1], values=data.iat[i, 6])

means1 = np.zeros((24, 1))
for i in range(0, 24):
    means1.put(indices=[i, 1], values=data.iat[i, 4])

# Calculate A and ell
A1 = np.std(means1)
ell1 = 30

# Build Kernel
rbf_kernel = (A1 ** 2) * RBF(length_scale=ell1, length_scale_bounds=(ell1/20, 5 * ell1))

# Build GP model by using Kernel
gp1 = GaussianProcessRegressor(kernel=rbf_kernel)

# Fit the model and print the parameters
gp1.fit(dates1, means1)
fitted_kernel = gp1.kernel_
fitted_params = fitted_kernel.get_params()
ell1 = fitted_params["k2__length_scale"]
A1 = math.sqrt(fitted_params["k1__constant_value"])

# For next 10 years
dates2 = np.zeros((24, 1))
for i in range(0, 24):
    dates2.put(indices=[i, 1], values=data.iat[i+24, 6])

means2 = np.zeros((24, 1))
for i in range(0, 24):
    means2.put(indices=[i, 1], values=data.iat[i+24, 4])

# Calculate A and ell
A2 = np.std(means2)
ell2 = 30

# Build Kernel
rbf_kernel2 = (A2 ** 2) * RBF(length_scale=ell2, length_scale_bounds=(ell2/20, 5 * ell2))

# Build GP model by using Kernel
gp2 = GaussianProcessRegressor(kernel=rbf_kernel2)

# Fit the model and print the parameters
gp2.fit(dates2, means2)
fitted_kernel = gp2.kernel_
fitted_params2 = fitted_kernel.get_params()
ell2 = fitted_params2["k2__length_scale"]
A2 = math.sqrt(fitted_params2["k1__constant_value"])
print(ell1)
print(A1)

x1 = np.arange(1, 366).reshape(365, 1)
x2 = range(1, 366)

# Means and sd for region(GP1)
y_mean1, y_std1 = gp1.predict(x1, return_std=True)

# Means and sd for region(GP2)
y_mean2, y_std2 = gp2.predict(x1, return_std=True)

# Prepare Fitted plot
lower1 = []
upper1 = []
for i in range(0, 365):
    lower1.append(float(y_mean1[i]) - float(y_std1[i]))
    upper1.append(float(y_mean1[i]) + float(y_std1[i]))

lower2 = []
upper2 = []
for i in range(0, 365):
    lower2.append(float(y_mean2[i]) - float(y_std2[i]))
    upper2.append(float(y_mean2[i]) + float(y_std2[i]))

# add sample points (GP1)
plt.plot(x2, y_mean1)
plt.fill_between(x2, lower1, upper1, alpha=0.2)
plt.scatter(dates1, means1, marker="o", s=16)
plt.show()

# add sample points (GP2)
plt.plot(x2, y_mean2)
plt.fill_between(x2, lower2, upper2, alpha=0.2)
plt.scatter(dates2, means2, marker="o", s=16)
plt.show()

# Sampling(GP1) and distribution
nSamples1 = 10000
samples1 = gp1.sample_y(x1, nSamples1)

distribution = []

for i in range(0, nSamples1):
    y = []
    for j in range(0, 365):
        y.append(float(samples1[j][0][i]))
    plt.plot(x2, y, lw=1)
    distribution.append(min(y))
plt.show()

mu = np.mean(distribution)
sigma = np.std(distribution)
plt.title(r'Distribution of Minimal: $\mu='+str(mu)+'$,$\sigma='+str(sigma)+'$')
n, bins, patches = plt.hist(distribution, 80, alpha=0.5, normed=True)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.show()

# Sampling(GP2) and distribution
nSamples2 = 10000
samples2 = gp2.sample_y(x1, nSamples2)

distribution2 = []

for i in range(0, nSamples2):
    y = []
    for j in range(0, 365):
        y.append(float(samples2[j][0][i]))
    plt.plot(x2, y, lw=1)
    distribution2.append(min(y))
plt.show()

mu = np.mean(distribution2)
sigma = np.std(distribution2)
plt.title(r'Distribution of Minimal: $\mu='+str(mu)+'$,$\sigma='+str(sigma)+'$')
n, bins, patches = plt.hist(distribution2, 80, alpha=0.5, normed=True)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.show()

# Parameters Explore(A)
A = 5
plot_Y = []
while A < 100:
    ell1 = 30
    rbf_kernel = (A ** 2) * RBF(length_scale=ell1, length_scale_bounds=(ell1/20, 5 * ell1))
    gp3 = GaussianProcessRegressor(kernel=rbf_kernel)
    y_mean3, y_std3 = gp3.predict(x1, return_std=True)
    A += 1
    plot_Y.append(np.average(y_std3))
plt.plot(range(5, 100), plot_Y)
plt.show()

# Parameters Explore(L)
L = 0.1
A = 30
plot_Y = []
while L < 99.9:
    rbf_kernel = (A ** 2) * RBF(length_scale=L, length_scale_bounds=(ell1/20, 5 * L))
    gp3 = GaussianProcessRegressor(kernel=rbf_kernel)
    y_mean3, y_std3 = gp3.predict(x1, return_std=True)
    L += 0.1
    samples3 = gp3.sample_y(x1, 1000)
    plot_Y.append(np.var(samples3))
plt.plot(np.arange(0.1, 100, 0.1), plot_Y)
plt.show()


# Log Likelihood Calculate(A)
logLikelihood = []
for A in range(10, 100, 1):
     logLikelihood.append(gp1.log_marginal_likelihood([math.log(A * A), math.log(30)]))
print(logLikelihood)
plt.plot(range(10, 100), logLikelihood)
plt.show()

# Log Likelihood Calculate(L)
logLikelihood = []
L = 0.1
while L < 32:
    logLikelihood.append(gp1.log_marginal_likelihood([math.log(30 * 30), math.log(L)]))
    L += 0.1
print(logLikelihood)
plt.plot(np.arange(0.1, 32, 0.1), logLikelihood)
plt.show()





