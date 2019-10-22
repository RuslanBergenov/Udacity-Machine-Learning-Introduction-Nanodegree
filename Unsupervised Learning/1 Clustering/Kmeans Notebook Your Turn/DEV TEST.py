import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import helpers2 as h
import tests as t
from IPython import display

import os
os.chdir('C:/Users/ruslan.bergenov/RD LOCAL/Udacity-Machine-Learning-Introduction-Nanodegree/Unsupervised Learning/Kmeans Notebook Your Turn')


#%matplotlib inline

# Make the images larger
plt.rcParams['figure.figsize'] = (16, 9)


data = h.simulate_data(200,5,4) # Create a dataset with 200 points, 5 features and 4 centers

# This will check that your dataset appears to match ours before moving forward
t.test_question_1(data)




k_value = 4# What should the value of k be?

# Check your solution against ours.
t.test_question_2(k_value)





# Try instantiating a model with 4 centers
kmeans_4 = KMeans(4) #instantiate your model

# Then fit the model to your data using the fit method
model_4 = kmeans_4.fit(data)#fit the model to your data using kmeans_4

# Finally predict the labels on the same data to show the category that point belongs to
labels_4 = model_4.predict(data) #predict labels using model_4 on your dataset

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_4)




# Try instantiating a model with 2 centers
kmeans_2 = KMeans(2)

# Then fit the model to your data using the fit method
model_2 = kmeans_2.fit(data)

# Finally predict the labels on the same data to show the category that point belongs to
labels_2 = model_2.predict(data)

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_2)





# Try instantiating a model with 7 centers
kmeans_7 = KMeans(7)

# Then fit the model to your data using the fit method
model_7 = kmeans_7.fit(data)

# Finally predict the labels on the same data to show the category that point belongs to
labels_7 = model_7.predict(data)

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_7)


range(1,10,10)

print(abs(model_2.score(data)))
print(abs(model_4.score(data)))
print(abs(model_7.score(data)))



# A place for your work - create a scree plot - you will need to
# Fit a kmeans model with changing k from 1-10
# Obtain the score for each model (take the absolute value)
# Plot the score against k
#https://www.pythoncentral.io/pythons-range-function-explained/
k =[]
distances =[]
for i in range(1,11):
    kmeans_i = KMeans(i)
    model_i = kmeans_i.fit(data)
    dist_i = abs(model_i.score(data))
    k.append(i)
    distances.append(dist_i)
    
    
#https://howtothink.readthedocs.io/en/latest/PvL_H.html
plt.plot(k, distances) 
plt.title("Elbow method in finding k")
plt.xlabel("k")
plt.ylabel("distances")
plt.show()
    
    
    
    
    