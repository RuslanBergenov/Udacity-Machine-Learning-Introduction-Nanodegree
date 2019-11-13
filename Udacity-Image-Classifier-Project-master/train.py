# Imports libraries

# import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms, models

# from PIL import Image

# import numpy as np

# import function to parse arguments
import argparse

# import utility functions
from utility_functions import load_and_transform_data

# model functions
from model import set_device, select_pretrained_model, build_my_classifier, train_model, test_model, save_checkpoint 
# create a parser object
parser = argparse.ArgumentParser(description="Udacity Image Classifier: train a neural network")

# add arguments

# device
parser.add_argument('--device_selection', help = "Select device: 'cuda' (GPU) or CPU. If you don't, then it'll select cuda, if available. Otherwise, it'll select CPU", action = 'store', type = str)

# dir
parser.add_argument('--flowers_data_directory', help = 'flowers data folder. Default = "flowers"', action = 'store', type = str, default = 'flowers')

# pretrained model
parser.add_argument('--pretrained_model_selection', help = 'Select pretrained model: either vgg19_bn or vgg16. Default = vgg19_bn', action = 'store', type = str, default = 'vgg19_bn')

# classifier
parser.add_argument('--hidden_1', help = 'Set hidden layer #1 size: between 25088 and 102. Default = 4096', action = 'store', type = int, default = 4096)
parser.add_argument('--hidden_2', help = 'Set hidden layer #2 size: between 25088 and 102. Default = 1024', action = 'store', type = int, default = 1024)
parser.add_argument('--dropout', help = 'Set dropout', action = 'store', type = int, default = 0.2)
parser.add_argument('--learning_rate', help = 'Select learning rate. Defalt =  0.001', action = 'store', type = int, default = 0.001)

# training
parser.add_argument('--epochs', help = "How many epochs? Default = 5", action = 'store', type = int, default = 5)
parser.add_argument('--dev_mode', help = "Are you just testing? if yes, training will stop after 10 steps", action = 'store', type = str, default = 'yes')

# checkpoint
parser.add_argument('--checkpoint_filename', help = "Checkpoint filename", action = 'store', type = str, default = 'checkpoint_image_classifier_part2_app.pth')

# save results 
args = parser.parse_args()

# convert argument parser input to a variable used in a function
data_dir = args.flowers_data_directory
pretrained_model_selection = args.pretrained_model_selection
hidden_1 = args.hidden_1
hidden_2 = args.hidden_2
dropout = args.dropout
learning_rate = args.learning_rate
device_selection = args.device_selection
epochs = args.epochs
dev_mode = args.dev_mode
checkpoint_filename = args.checkpoint_filename

#select device 
device = set_device(device_selection)

# Extract and Transform data
train_data, valid_data, test_data, trainloader, testloader, validloader = load_and_transform_data(data_dir)

# select pretrained model
model = select_pretrained_model(pretrained_model_selection, device)

# classifier 
criterion, optimizer = build_my_classifier(hidden_1, hidden_2, dropout, learning_rate, device)

# make sure the model is good
print("the model we'll be training is as follows:")
print(model)

#train model
train_model(epochs, trainloader, validloader, testloader, device, criterion, optimizer, dev_mode)

# test model
test_model(testloader, device, model)

########################
# save checkpoint
# comment it out during development not to overwite the checkoint with the model I trained
########################
### save_checkpoint(checkpoint_filename, model, train_data,optimizer, pretrained_model_selection, epochs)