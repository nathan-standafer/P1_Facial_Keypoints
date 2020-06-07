## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.25)
        
        #assume 224 X 224 image  size
        final_dim = (((224-4)/2) - 4) / 2
        self.flattened_size = int(64 * final_dim**2)
        
        self.fc1 = nn.Linear(self.flattened_size , 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256 , int(68*2))
        
        self.leakyRelu = nn.LeakyReLU(0.1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(self.leakyRelu(self.conv1(x)))
        x = self.pool(self.conv2_bn(self.leakyRelu(self.conv2(x))))
        #print("x.shape after conv2 and pooling: {}".format(x.shape))
        
        #flatten
        x = x.view(-1, self.flattened_size)
        #print("x.shape after flattening: {}".format(x.shape))
        
        # a modified x, having gone through all the layers of your model, should be returned
        #x = self.drop(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.drop(self.leakyRelu(self.fc1_bn(self.fc1(x))))
                           
        x = self.fc2(x)
        
        return x
