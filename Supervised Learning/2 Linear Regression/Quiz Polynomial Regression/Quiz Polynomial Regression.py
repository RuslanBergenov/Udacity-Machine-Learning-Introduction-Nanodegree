# LINKS

#https://stackoverflow.com/questions/36009907/numpy-reshape-1d-to-2d-array-with-1-column

#https://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn?newreg=6169b51183574f4e8bd42a56d2572ba5

#https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

#https://stackoverflow.com/questions/46096347/plot-polynomial-regression-in-python-with-scikit-learn



# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import os

os.chdir('C:\\Users\\Ruslan\\Documents\\Projects\\RD\\Intro to Machine Learning Nanodegree\\Supervised Learning\\2 Linear Regression\\Quiz Polynomial Regression')



# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv')
X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values

type(X)


plt.scatter(X, y)
plt.show()

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)


y_poly_pred = poly_model.predict(X_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)


# Plot
X_plot=np.linspace(-3,3,100).reshape(-1,1)
X_plot_poly=poly_feat.fit_transform(X_plot)
plt.plot(X,y,"b.")
plt.plot(X_plot_poly[:,1],poly_model.predict(X_plot_poly),'-r')
plt.show()

