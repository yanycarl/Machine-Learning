
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("D:\\DATA\\ML\\W7\\Data_Eng_eco_msoa_2011.csv", sep=',')
data2 = pd.read_csv("D:\\DATA\\ML\\W7\\Data_Eng_msoa_qualification_2011.csv", sep=',')

un_employ = np.mat(data.iloc[:, 13]).A[0]
high_edu = np.mat(data2.iloc[:, 9]).A[0]
un_employ2 = np.mat(data.iloc[:, 13]).A.reshape(6791, 1)
high_edu2 = np.mat(data2.iloc[:, 9]).A.reshape(6791, 1)
print(un_employ)
print(high_edu)

sns.distplot(un_employ, bins=50, rug=True)
plt.title("The histogram of unemployment rate")
plt.xlabel("The unemployment rate")
plt.ylabel("The appear time")
plt.show()

print(np.mean(un_employ))
print(np.median(un_employ))
print(sp.stats.mode(un_employ).mode[0])
print(np.var(un_employ))
stand_dev = sp.std(un_employ)
skewness = sp.stats.moment(un_employ, 3)/(stand_dev**3)
print("The Skewness:", skewness)
kurtosis = sp.stats.moment(un_employ, 4)/(stand_dev**4) - 3
print("kurtosis:", kurtosis)

sns.distplot(high_edu, bins=50, rug=True)
plt.title("The histogram of high qualification rate")
plt.xlabel("The high qualification rate")
plt.ylabel("The appear time")
plt.show()

print(np.mean(high_edu))
print(np.median(high_edu))
print(sp.stats.mode(high_edu))
print(np.var(high_edu))
stand_dev = sp.std(high_edu)
skewness = sp.stats.moment(high_edu, 3)/(stand_dev**3)
print("The Skewness:", skewness)
kurtosis = sp.stats.moment(high_edu, 4)/(stand_dev**4) - 3
print("kurtosis:", kurtosis)

X = [un_employ, high_edu]
print(X)

means = np.mean(X, 1)
print(means)

cov = np.cov(X)
print(cov)

cor = np.corrcoef(X)
print(cor)

plt.scatter(X[0], X[1])
plt.xlabel('Unemployment rate')
plt.ylabel('High education qualification')
plt.tight_layout()
plt.show()

plt.hist2d(X[0], X[1], cmap='Blues')
plt.xlabel('Unemployment rate')
plt.ylabel('High education qualification')
plt.tight_layout()
plt.show()

sns.kdeplot(X[0], X[1], cmap="Blues")
plt.xlabel('Unemployment rate')
plt.ylabel('High education qualification')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[0], X[1], X[1], marker='o', c=X[1], cmap='seismic')
ax.set_xlabel('Engine size')
ax.set_ylabel('Weight')
ax.set_zlabel('Miles per gallon')
plt.legend()
plt.show()

df = pd.DataFrame(np.column_stack((X[0], X[1])), columns=['Variable 1', 'Variable 2'])
sns.jointplot(x="Variable 1", y="Variable 2", data=pd.DataFrame(df), kind='kde')
plt.show()

p = sns.jointplot(x="Variable 1", y="Variable 2", data=pd.DataFrame(df), kind='reg', size=1)
p = p.plot(sns.regplot, sns.distplot)
plt.show()

sns.jointplot(x="Variable 1", y="Variable 2", data=pd.DataFrame(df), kind='hex')
plt.show()

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(un_employ2)
model = LinearRegression()
model.fit(X_poly, high_edu2)

plt.hist2d(X[0], X[1], bins=100)
plt.scatter(np.mat(X_poly).transpose().A[1], model.predict(X_poly), color='white')
plt.title("The 2D histogram with regression line")
plt.xlabel('The Unemployment rate')
plt.ylabel('The high education qualification rate')
plt.show()
