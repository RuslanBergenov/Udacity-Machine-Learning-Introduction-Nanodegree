# Imports libraries

import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
from torchvision import datasets, transforms #, models


from PIL import Image

import numpy as np

# TODO: Define your transforms for the training, validation, and testing sets
def load_and_transform_data(data_dir):

    """
    Source: Udacity Deep Learning with Pytorch lesson

    """
    global train_data, valid_data, test_data, trainloader, validloader, testloader
    # directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/2
    # my understanding is that len(dataloader) = len(dataset) / batch_size
#     print('Loaded data. Length of data loaders:')
#     print(len(trainloader))
#     print(len(validloader))
#     print(len(testloader))

    return train_data, valid_data, test_data, trainloader, validloader, testloader


def process_image(image_filepath):
    ''' Open, transform (resize, centre crop, normalize (between zero and 1 ), convert to a Numpy array)
    '''
    # Converting image to PIL image using image file path
    PIL_Image = Image.open(f'{image_filepath}')

    # Building image transform
    """this transform is taken from Udacit intro to Pytorch lesson"""
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    PIL_Transformed = transform(PIL_Image)
    
    # Converting to Numpy array 
    """https://kite.com/python/examples/4887/pil-convert-between-a-pil-%60image%60-and-a-numpy-%60array%60"""
    image_np_array = np.array(PIL_Transformed)
    
    return image_np_array