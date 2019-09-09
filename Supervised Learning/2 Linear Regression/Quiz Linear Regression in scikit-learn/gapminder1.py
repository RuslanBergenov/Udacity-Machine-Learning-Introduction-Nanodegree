import os

os.chdir('C:\\Users\\Ruslan\\Documents\\Projects\\RD\\Intro to Machine Learning Nanodegree\\Supervised Learning\\2 Linear Regression\\Quiz Linear Regression in scikit-learn')


# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv") 
print(bmi_life_data.head())

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
model = LinearRegression()
bmi_life_model = model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = model.predict([[21.07931]])

print(laos_life_exp)

#1. The data was loaded correctly!
#2. Well done, you fitted the model!
#3. Well done, your prediction of a life expectancy 60.31564716399306 is correct!