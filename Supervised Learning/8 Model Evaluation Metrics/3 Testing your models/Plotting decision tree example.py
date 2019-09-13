#https://pypi.org/project/dtreeplt/
# https://github.com/nekoumei/dtreeplt
# You should prepare trained model,feature_names, target_names.
# in this example, use iris datasets.
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from dtreeplt import dtreeplt


iris = load_iris()
model = DecisionTreeClassifier()
model.fit(iris.data, iris.target)

dtree = dtreeplt(
    model=model,
    feature_names=iris.feature_names,
    target_names=iris.target_names
)
fig = dtree.view()
#if you want save figure, use savefig method in returned figure object.
fig.savefig('output_test_Community.png')





# exploring the data
#https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
iris = load_iris()

# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..
# the original dataset would probably call this ['Species']
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
