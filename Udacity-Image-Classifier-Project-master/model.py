# Imports libraries

import torch
from torch import nn
from torch import optim
# import torch.nn.functional as F
from torchvision import models

# from PIL import Image

import pandas as pd
import numpy as np

import re

from workspace_utils import active_session

from utility_functions import process_image

import json

def set_device(device_selection):
    if device_selection:
    # if u selected a device
        try:
            device = torch.device(device_selection)
        except:
            # if the code above fails, that's maybe cus u selected GPU and we don't have it
            print("Sorry, you may have selected GPU, which we don't have. Setting device to CPU")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if u didn't select a device        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Our device is:', device)
    
    return device

def select_pretrained_model(pretrained_model_selection, device):
    global model
    if pretrained_model_selection == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    elif pretrained_model_selection == 'vgg16':
        model = models.vgg16(pretrained=True)
    else: 
        print("Please select either vgg19_bn or vgg16")
        
#     print(model)
    model.to(device)
    return model        
            
def build_my_classifier(hidden_1, hidden_2, dropout, learning_rate, device):
    
    global criterion, optimizer

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
                        nn.Linear(25088, hidden_1),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_1, hidden_2),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_2, 102),
                        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    
    return criterion, optimizer
          
def train_model(epochs, trainloader, validloader, testloader, device, criterion, optimizer, dev_mode):
    """
Code adapted from Udacity Deep Learning with Pytorch lesson
"""

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5

    with active_session():

        for epoch in range(epochs):
            
            # this is just for testing purposes, remove later. Break lop early
            if dev_mode.upper() == 'YES':
                if epoch > 1:
                    print('Interrupted after epoch 1 to speed up development and testing')
                    break
            
            for inputs, labels in trainloader:
                steps += 1
                
                # this is just for testing purposes, remove later. Break after 10 steps
                if dev_mode.upper() == 'YES':
                    if steps > 10:
                        print('Interrupted after step 10 to speed up development and testing')
                        break
                
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
  
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Step {steps}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

def test_model(testloader, device, model):
    inputs, labels = next(iter(testloader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    model.to(device)

    # Get the class probabilities
    ps = torch.exp(model(inputs))

    # Make sure the shape is appropriate, we should get 102 class probabilities for 64 examples
    print(ps.shape)

    top_p, top_class = ps.topk(1, dim=1)
    # Look at the most likely classes for the first 10 examples
#     print(top_class[:10,:])

    equals = top_class == labels.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')
    
    return accuracy

def save_checkpoint(checkpoint_filename, model, train_data,optimizer, pretrained_model_selection, epochs):
    # save labels
    model.class_to_idx = train_data.class_to_idx 
    model.to('cpu')
    torch.save({
            'model_state_dict': model.state_dict(),         
            'optimizer_state_dict': optimizer.state_dict(),
            'classifier': model.classifier,        
            'structure': pretrained_model_selection,
            'epochs': epochs,
            'class_to_idx': model.class_to_idx
            }, checkpoint_filename)
    
def load_checkpoint(checkpoint_filename, pretrained_model_selection, learning_rate,device):
    
    global model, criterion, optimizer, checkpoint
    
    if pretrained_model_selection == 'vgg19_bn':
        model = models.vgg19_bn()
    elif pretrained_model_selection == 'vgg16':
        model = models.vgg16()

    criterion = nn.NLLLoss()

    PATH = checkpoint_filename
    
    #https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745
    if device == torch.device("cuda"):
        checkpoint = torch.load(PATH)
    elif device == torch.device('cpu'): 
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
      
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.optimizer_state_dict = checkpoint['optimizer_state_dict']
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model.to(device);

    model.eval()
  # making sure the model loaded okay  
#     print("Our model: \n\n", model, '\n')
#     print("The state dict keys: \n\n", model.state_dict().keys())
#     print("The state dict: \n\n", model.state_dict())
    
    return model, criterion, optimizer, checkpoint

def load_cat_to_name(cat_to_name_filename):
    with open(cat_to_name_filename, 'r') as f:
        cat_to_name = json.load(f)
#     print(cat_to_name['1'])     # making sure it acutally loaded       
    return cat_to_name    

def predict_one_image(image_filepath, model, topk, cat_to_name):
    ''' Predict flower species.
        To be honest, I struggled with this part and got stuck so badly, that I had to take a peek at waht other students had submitted
        The code inn this cell  is adapted from:
        ttps://github.com/S-Tabor/udacity-image-classifier-project/blob/master/Image%20Classifier%20Project.ipynb
        by S-Tabor
        I didn't copy paste the code, but I got the general idea for implementation.
        
        I made sure to avoid plagiarism
        
        https://udacity.zendesk.com/hc/en-us/articles/360001451091-What-is-plagiarism-
        Not Plagiarism:
        Looking at someone else’s code to get a general idea of implementation, then putting it away and starting to write your own code from scratch.
    '''
#     process the image
    img = process_image(image_filepath)
    
   # Converting to torch tensor from Numpy array
    #https://discuss.pytorch.org/t/how-to-convert-array-to-tensor/28809
    #https://discuss.pytorch.org/t/difference-between-tensor-and-torch-floattensor/24448/2

    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Adding a dimension to image
    #https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155

    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients to speed up the following step
    model.eval()
    
    with torch.no_grad():
        # Running image through network
        try:
            output = model(img_add_dim)
        except:
            output = model(img_add_dim.cuda()) 

    # probabilities
    probabilities = torch.exp(output)
#     print(probabilities)
    top_probabilities = probabilities.topk(topk)[0]
    top_indexes = probabilities.topk(topk)[1]
    
    # Converting probabilities and outputs to numpy arrays
    probs_top_list = np.array(top_probabilities)[0]
#     print(type(probs_top_list))
    index_top_list = np.array(top_indexes[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    #https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping
    indx_to_class = {A: B for B, A in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
    
   
    #############################
    class_names = []
    for i in classes_top_list:
        class_names += [cat_to_name[i]]  

    df = pd.DataFrame(
    {'classes_top_list': classes_top_list,
     'top_class_names': class_names,
     'probs_top_list': probs_top_list 
    })
            
    print()
    print(image_filepath)
    
    # extract folder number from jpeg filepath and then get actual class name from it
    # https://stackoverflow.com/questions/5041008/how-to-find-elements-by-class
    # https://stackoverflow.com/questions/25353652/regular-expressions-extract-text-between-two-markers
    # https://stackoverflow.com/questions/7167279/regex-select-all-text-between-tags
    pattern = '\/test\/(.*?)\/image_'
    folder_number = re.findall(pattern, image_filepath)
    folder_number = folder_number[0]
    actual_class = cat_to_name[folder_number]
    
    print()
    print('Actual class')
    print(actual_class)
#     print('top probs', probs_top_list)
#     print('top_classes',classes_top_list)
    print()
    print('Prediction')
    print (df.to_string(index=False))

    return probs_top_list, classes_top_list