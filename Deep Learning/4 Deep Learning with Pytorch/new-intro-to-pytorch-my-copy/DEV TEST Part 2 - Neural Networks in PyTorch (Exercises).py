#Import necessary packages

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt



### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)



dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)




plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');




## Your solution
images_reshape = images.reshape(64, 784)

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))


## Your solution
### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

n_input = images_reshape.shape[1]
print('n_input:',n_input)
n_hidden = 256
n_output = 10

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

hidden_layer_output = activation(torch.mm(images_reshape,W1) + B1)

# output of your network, should have shape (64,10)
out = torch.mm(hidden_layer_output,W2)+B2

print('out_shape:', out.shape)

print('out:', out)


def softmax(X):
    ## TODO: Implement the softmax function here
    # X = out # this is for testing
    expX = np.exp(X)
    sumExpX = expX.sum(dim=1)
    #sumExpX.shape # this is for debugging
    sumExpX = sumExpX.reshape(64,1)
    #sumExpX.shape # this is for debugging
    
    result = expX / sumExpX
    
    return result

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))






