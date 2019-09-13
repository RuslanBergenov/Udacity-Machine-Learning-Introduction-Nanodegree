# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]




#https://stackoverflow.com/questions/47006268/matplotlib-scatter-plot-with-color-label-and-legend-specified-by-c-option?noredirect=1&lq=1
scatter_x = data[:,0:1]
scatter_y = data[:,1:2]
group = data[:,2]
cdict = {1: 'red', 0: 'blue'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
#ax.legend()
plt.show()

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
#model = SVC(kernel = 'poly', degree=5, C=0.05, gamma='scale')
#
#model = SVC(kernel = 'poly', degree=6, C=0.5, gamma='scale')
myC = 9
myGamma = 9

model = SVC(kernel = 'rbf', C=myC, gamma=myGamma)

# TODO: Fit the model.
model.fit(X,y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y_pred, y)
print('C: ', format(myC), 'gamma:', format(myGamma), 'Accuracy score: ', format(acc))



# =============================================================================
# PLOTTING DECISION REGION
# =============================================================================
#https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
from mlxtend.plotting import plot_decision_regions

# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X, 
                      y=y.astype(np.integer),
                      clf=model, 
                      legend=None,
                      #legend=2
                      zoom_factor=3.0)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('x0')
plt.ylabel('x1')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Solution Boundary', size=16)

plt.show()


