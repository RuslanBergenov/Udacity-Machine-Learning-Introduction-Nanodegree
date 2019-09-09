from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

tree.plot_tree(clf.fit(iris.data, iris.target)) 





import lightgbm as lgb
from sklearn.datasets import load_iris

%matplotlib inline

X, y = load_iris(True)
clf = lgb.LGBMClassifier()
clf.fit(X, y)
lgb.plot_tree(clf)