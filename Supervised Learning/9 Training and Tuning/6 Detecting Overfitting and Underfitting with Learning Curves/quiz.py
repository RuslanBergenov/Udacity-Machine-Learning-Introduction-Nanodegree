#https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/

exec(open('vis.py').read())

# Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

np.random.seed(55)

exec(open('utils.py').read())

# Fix random seed

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
print()
print('LogisticRegression')
estimator = LogisticRegression(solver='lbfgs')
draw_learning_curves(X, y, estimator, num_trainings=9)





### Decision Tree
#np.random.seed(55)
print()
print('GradientBoostingClassifier')
estimator = GradientBoostingClassifier()
draw_learning_curves(X, y, estimator, num_trainings=9)
#np.random.seed(55)
### Support Vector Machine
print()
print('SVC')
estimator = SVC(kernel='rbf', gamma=1000)
draw_learning_curves(X, y, estimator, num_trainings=9)



exec(open('solutionBoundary.py').read())
