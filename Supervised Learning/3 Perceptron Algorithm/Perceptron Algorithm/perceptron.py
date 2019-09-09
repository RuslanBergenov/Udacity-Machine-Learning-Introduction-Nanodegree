#https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib


os.chdir('C:\\Users\\Ruslan\\Documents\\Projects\\RD\\Intro to Machine Learning Nanodegree\\Supervised Learning\\3 Perceptron Algorithm\\Perceptron Algorithm')

data = pd.read_csv('data.csv', header = None)
colors = ['red','blue']

plt.scatter(data[[0]], data[[1]], c=data[[2]], label= data[[2]], cmap=matplotlib.colors.ListedColormap(colors))
#plt.legend()
plt.show()


# split dataset into X and y
X = data.iloc[:, : 2].values
y = data[2].values



# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

W=np.array([1.0,1.0])

b=np.array([1.0])

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

prediction(X, W, b)

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    

perceptronStep(X, y, W=W, b=b, learn_rate = 0.01)

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

train = trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25)