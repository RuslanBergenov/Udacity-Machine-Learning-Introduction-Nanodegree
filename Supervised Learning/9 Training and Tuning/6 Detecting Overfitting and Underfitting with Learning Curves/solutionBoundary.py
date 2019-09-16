from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# =============================================================================
# PLOTTING DECISION REGION
# =============================================================================
#https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
from mlxtend.plotting import plot_decision_regions


def plotDecisionRegion():
    # Plot Decision Region using mlxtend's awesome plotting function
    plot_decision_regions(X=X, 
                          y=y.astype(np.integer),
                          clf=model, 
                          legend=None,
#                          legend=2,
                          zoom_factor=3.0,
                          colors ='red,blue')
    
    
    # Update plot object with X/Y axis labels and Figure Title
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Solution Boundary', size=16)
    
    plt.show()




print()
print('LogisticRegression - Underfitting')
model =  LogisticRegression(solver='lbfgs')
model.fit(X,y)
plotDecisionRegion()

print()
print('Decision Tree - Just Right')
model =  GradientBoostingClassifier()
model.fit(X,y)
plotDecisionRegion()



print()
print('Support Vector Machine - Overfitting')
model =  SVC(kernel='rbf', gamma=1000)
model.fit(X,y)
plotDecisionRegion()
