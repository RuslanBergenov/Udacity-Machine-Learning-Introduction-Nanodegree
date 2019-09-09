# LINKS

#https://stackoverflow.com/questions/36009907/numpy-reshape-1d-to-2d-array-with-1-column

#https://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn?newreg=6169b51183574f4e8bd42a56d2572ba5

#https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

#https://stackoverflow.com/questions/46096347/plot-polynomial-regression-in-python-with-scikit-learn

#https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html


#Import basic packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score

os.chdir('C:\\Users\\Ruslan\\Documents\\Projects\\RD\\Intro to Machine Learning Nanodegree\\Supervised Learning\\2 Linear Regression\\Quiz Polynomial Regression')


# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header=0)
print(train_data)

x = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values

type(x)


plt.scatter(x, y)
plt.show()


# Running simple linear Regression first using statsmodel OLS

import statsmodels.api as sm

model = sm.OLS(y, x).fit()
ypred = model.predict(x) 

plt.scatter(x,y)
plt.plot(x,ypred)
print(model.summary())


#Generate Polynomials
from sklearn.preprocessing import PolynomialFeatures
degree = 5
polynomial_features= PolynomialFeatures(degree=degree)
xp = polynomial_features.fit_transform(x)
xp.shape

xp == x

#Running regression on polynomials using statsmodel OLS
import statsmodels.api as sm

model = sm.OLS(y, xp).fit()
y_poly_pred = model.predict(xp) 

ypred.shape



print(model.summary())





rmse_train = np.sqrt(mean_squared_error(y,y_poly_pred))
r2_train = r2_score(y,y_poly_pred)
print("The model performance for the training set")
print("-------------------------------------------")
print("Polynomial degree is {}".format(degree))
print("RMSE of training set is {}".format(rmse_train))
print("R2 score of training set is {}".format(r2_train))


# Plot
X_plot=np.linspace(-3,3,100).reshape(-1,1)
X_plot_poly=polynomial_features.fit_transform(X_plot)
plt.plot(x,y,"b.")
plt.plot(X_plot_poly[:,1],model.predict(X_plot_poly),'-r')
plt.show()
