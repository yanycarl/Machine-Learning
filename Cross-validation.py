
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def powers(temp, n):
    return np.power(np.expand_dims(temp, axis=-1), [np.arange(n)])


def polynomial(temp, coefficient):
    deg = np.shape(coefficient)[-1]
    return np.dot(powers(temp, deg), np.transpose(coefficient))


def fit_polynomial(temp, temp_y, n):
    temp_poly_x = powers(temp, n+1)
    coefficient, _, _, _ = np.linalg.lstsq(temp_poly_x, temp_y, rcond=None)
    return coefficient


def bias_sq(temp_X, coefficient, true_y):
    mean_y = np.mean(polynomial(temp_X, coefficient), axis=-1)
    return np.square(mean_y - true_y)


def variance(temp_X, coefficient):
    return np.var(polynomial(temp_X, coefficient), axis=-1)


def square_error(temp_X, coefficient, temp_y):
    predicted = polynomial(temp_X, coefficient)
    return np.square(predicted - np.expand_dims(temp_y, -1))


def get_fold(temp_X, temp_y, fold, temp_folds):
    folds_X = np.array_split(temp_X, temp_folds)
    test_X = folds_X.pop(fold)
    train_X = np.concatenate(folds_X)
    folds_y = np.array_split(temp_y, temp_folds)
    test_y = folds_y.pop(fold)
    train_y = np.concatenate(folds_y)
    return train_X, train_y, test_X, test_y


# Build some polynomial samples
example_coefficients = [1, -4, 1]
X = np.random.uniform(-4, 4, size=15)
eps = np.random.normal(scale=1, size=15)
y = polynomial(X, example_coefficients) + eps

# Fit example polynomial model
example_degree = 4
learned_coefficients = fit_polynomial(X, y, example_degree)

# Draw all example points and lines
plt.scatter(X, y)
plt.ylim(y.min()-1, y.max()+1)
all_X = np.arange(-4.0, 4.0, 0.2)
poly_y = polynomial(all_X, example_coefficients)
plt.plot(all_X, poly_y, '--', label='True Line')
poly_y = polynomial(all_X, learned_coefficients)
plt.plot(all_X, poly_y, label='Fitted Line')
plt.legend()
plt.show()

# fit class8_generated_data using different degrees
k = 5  # The sample split parts
polynomial_degrees = [2, 3, 4, 5, 6, 7]
X, y, test_X, test_y, test_y_gt = np.load("C:\\Users\\caoya\\Documents\\current8\\class8_generated_data.npy")
fig, subplots = plt.subplots(nrows=k+1, ncols=len(polynomial_degrees))
folds_X = np.split(X, k)
folds_y = np.split(y, k)
all_X = np.arange(-3.0, 3.0, 0.2)

for j in range(len(polynomial_degrees)):
    learned_coefficients = np.zeros((k, polynomial_degrees[j]+1))
    for i in range(k):
        sub_X = folds_X[i]
        sub_y = folds_y[i]
        subplots[i, j].scatter(sub_X, sub_y)
        learned_coefficients[i] = fit_polynomial(sub_X, sub_y, polynomial_degrees[j])
        poly_y = polynomial(all_X, learned_coefficients[i])
        subplots[i, j].plot(all_X, poly_y)

    subplots[0, j].set_title("Degree {:d}".format(polynomial_degrees[j]))
    mse = np.mean(square_error(test_X, learned_coefficients, test_y))
    bias = np.mean(bias_sq(test_X, learned_coefficients, test_y_gt))
    var = np.mean(variance(test_X, learned_coefficients))
    subplots[-1, j].bar(['Total Error', 'Bias^2', 'Variance'], [mse, bias, var])
plt.show()

# FTSE 100 data read in
file = pd.read_csv("C:\\Users\\caoya\\Documents\\current8\\class8_data_FTSE100_2.csv", sep='\t')
price = np.mat(file.iloc[:, 1]).A[0]
temp_list = price.tolist()
y = []
for i in range(len(temp_list)-1, 0, -1):
    y.append(temp_list[i]/1e3)
X = np.arange(np.size(y))

# Plot FTSE 100 data
plt.scatter(X, y)

# K-fold cross validation(k<n)
num_folds = y.__len__()
model_degree = 3
efficient = fit_polynomial(X, y, model_degree)
plt.plot(X, polynomial(X, efficient))
plt.ylabel("Price (£x1000)")
plt.xlabel("Months since September 2008")
plt.title("FTSE 100 index price")
plt.show()

# Shuffle samples
np.random.shuffle(X)
new_y = []
for i in range(len(X)):
    new_y.append(y[X[i]])
print(new_y)

# Calculate time whole process of CV
print("Running cross validation on polynomial model of degree {:d}".format(model_degree))
start_t = time.time()
total_loss = 0
losses = []

for i in range(num_folds):
    train_X, train_y, test_X, test_y = get_fold(X, new_y, i, num_folds)
    learned_coefficients = fit_polynomial(train_X, train_y, model_degree)
    predicted_y = polynomial(test_X, learned_coefficients)
    fold_loss = np.square(predicted_y - test_y)
    mean_fold_loss = np.mean(fold_loss)
    losses.append(mean_fold_loss)
    # print("Mean loss for fold {:d}/{:d}: {:.5f}".format(i + 1, num_folds, mean_fold_loss))
# end_t = time.time()
# elapsed_t_ms = (end_t - start_t) * 1e3
# print("Completed {:d} folds in {:.2f} milliseconds. \n".format(num_folds, elapsed_t_ms))

print("The mean of the loss in the model:(d=3)", np.mean(losses))
print("The sd of the loss in the model:(d=3)", np.std(losses))

# Two models
mu, sigma = 0.1426, 0.1856
sampleNo = 1000
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)
plt.hist(s, bins=20, label="d=1", density=True)

mu, sigma = 0.1045, 0.1119
sampleNo = 1000
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)
plt.hist(s, bins=20, label="d=3", density=True)
plt.xlabel("The value of the model")
plt.ylabel("The appearance frequency")
plt.legend()
plt.show()

