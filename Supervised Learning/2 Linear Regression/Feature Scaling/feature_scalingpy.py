# TODO: Add import statements
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso 
from sklearn.preprocessing import StandardScaler
import os


try: 
    os.chdir('C:\\Users\\Ruslan\\Documents\\Projects\\RD\\Intro to Machine Learning Nanodegree\\Supervised Learning\\2 Linear Regression\\Quiz Regularization')
except: 
    print('cannot set directory')


# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header=None)

#https://stackoverflow.com/questions/30673684/pandas-dataframe-first-x-columns
X = train_data.iloc[:, : 6].values

y = train_data[6].values
# TODO: Create the standardization scaling object.
scaler = StandardScaler()

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X_scaled, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)