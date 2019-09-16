print('Data Exploration')


#https://stackoverflow.com/questions/47006268/matplotlib-scatter-plot-with-color-label-and-legend-specified-by-c-option?noredirect=1&lq=1
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = np.asarray(pd.read_csv('data.csv'))

scatter_x = data[:,0:1]
scatter_y = data[:,1:2]
group = data[:,2]


cdict = {1.: 'blue', -1.: 'red'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()