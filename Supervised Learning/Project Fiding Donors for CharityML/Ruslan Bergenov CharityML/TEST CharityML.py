# =============================================================================
# ## Exploring the Data
# =============================================================================


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# =============================================================================
# Implementation: Data Exploration
# =============================================================================

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
data.groupby('income').income.count()

# 2 ways
n_greater_50k = data.groupby('income').income.count()[1]
n_greater_50k = data.loc[data['income'] == '>50K'].shape[0]


# TODO: Number of records where individual's income is at most $50,000
# 2 ways
n_at_most_50k = data.groupby('income').income.count()[0]
n_at_most_50k = data.loc[data['income'] == '<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (data.loc[data['income'] == '>50K'].shape[0] / data.shape[0])*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
#https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
print("Percentage of individuals making more than $50,000: {}%".format("%.2f" % greater_percent))



# =============================================================================
# Transforming Skewed Continuous Features
# =============================================================================

data.describe()



# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)






# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)





# =============================================================================
# ### Normalizing Numerical Features
# =============================================================================



# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


features_log_minmax_transform.describe()


# =============================================================================
# Implementation: Data Preprocessing
# =============================================================================


# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = (income_raw == '>50K').map({True:1, False:0})

# DI check
print(np.mean(income)) # matches greater_percent
print(np.mean(income)*100 - greater_percent) # zero is good
print()

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print (encoded)





# =============================================================================
# ### Question 1 - Naive Predictor Performace
# =============================================================================

'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# TODO: Calculate accuracy, precision and recall
accuracy =  (np.sum(income) + 0) / n_records
recall = np.sum(income)  / (np.sum(income) +0)
precision = np.sum(income) / n_records

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1 + beta**2) * ((precision * recall) / (beta**2 * precision + recall))

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

