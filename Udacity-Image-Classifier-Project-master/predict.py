# Imports libraries

import torch
#from torch import nn
#from torch import optim
#import torch.nn.functional as F
#from torchvision import datasets, transforms, models

#from PIL import Image
#
#import pandas as pd
#import numpy as np
#
#import os 

# import function to parse arguments
import argparse

# import functions

from utility_functions import load_and_transform_data

# model functions
from model import load_checkpoint, predict_one_image, set_device, load_cat_to_name,  test_model

# create a parser object
parser = argparse.ArgumentParser(description="Udacity Image Classifier: test a neural network")

# selection has to be the same which was used during training 

parser.add_argument('--device_selection', help = "Select device: 'cuda' (GPU) or CPU. If you don't, then it'll select cuda, if available. Otherwise, it'll select CPU", action = 'store', type = str)

parser.add_argument('--flowers_data_directory', help = 'flowers data folder. Default = "flowers"', action = 'store', type = str, default = 'flowers')

parser.add_argument('--pretrained_model_selection', help = 'Select pretrained model: either vgg19_bn or vgg16. Default = vgg19_bn', action = 'store', type = str, default = 'vgg19_bn')

parser.add_argument('--checkpoint_filename', help = "Checkpoint filename", action = 'store', type = str, default = 'checkpoint_image_classifier_part2_app.pth')

parser.add_argument('--learning_rate', help = 'Select learning rate. Defalt =  0.001', action = 'store', type = int, default = 0.001)

parser.add_argument('--cat_to_name_filename', help = 'Select filename for label mapping', action = 'store', type = str, default = 'cat_to_name.json')

parser.add_argument('--image_filepath', help = 'Select filename for label mapping', action = 'store', type = str, default = 'flowers/test/68/image_05903.jpg')

parser.add_argument('--topk', help = 'Top K classes to predict. Default is 5.', action = 'store', type = int, default = 3)

# save results 
args = parser.parse_args()

# convert argument parser input to a variable used in a function
device_selection = args.device_selection
data_dir = args.flowers_data_directory
pretrained_model_selection = args.pretrained_model_selection
checkpoint_filename = args.checkpoint_filename
learning_rate = args.learning_rate
cat_to_name_filename = args.cat_to_name_filename
image_filepath = args.image_filepath
topk = args.topk

#select device 
device = set_device(device_selection)

# load model
loaded_model, criterion, optimizer, checkpoint = load_checkpoint(checkpoint_filename, pretrained_model_selection, learning_rate,device)

# Extract and Transform data
train_data, valid_data, test_data, trainloader, testloader, validloader = load_and_transform_data(data_dir)

# check device
print("Is our device GPU?")
print(device == torch.device("cuda"))

# test the model but only if it's GPU, on CPU it'll run forever. Purpose: to see if the model is fine after saving a checkpoint and loading it
if device == torch.device("cuda"): 
    test_model(testloader, device, loaded_model)
else:
    pass
    
# label mapping
cat_to_name = load_cat_to_name(cat_to_name_filename)

# predict
probs_top_list, classes_top_list = predict_one_image(image_filepath, loaded_model, topk, cat_to_name)