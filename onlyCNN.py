import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import time

import os
import cv2
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

class AE1(nn.Module):

    def __init__(self):
    
        super(AE, self).__init__()
        
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.l1 = nn.Linear(32*32*3,32*32*3,True)
        self.l2 = nn.Linear(32*32*3,32*32*3,True)
        self.l3 = nn.Linear(32*32*3,32*32*3,True)
        self.out = nn.Linear(32*32*3,32*32*3,True)       

    def forward(self, image):
    
        im = image.reshape(3072)
                    
        im = self.l1(im)
        im = self.dropout(im)
        
        im = self.l2(im)
        im = self.dropout(im)
        
        im = self.l3(im)
        im = self.dropout(im)
        
        im = self.out(im)
        
        return im

class AE(nn.Module):
    
    def __init__(self):
        
        super(AE,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6,3,3)
        self.convt1 = nn.ConvTranspose2d(3,6,3)
        self.convt2 = nn.ConvTranspose2d(6,3,3)
        
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convt1(x)
        x = self.convt2(x)
        
        return x   

class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
       
    def forward(self, output, target):

        output = output.reshape(3,32,32)
        X = output
        Y = target
        gauss_loss = torch.log(torch.sum( torch.abs(X-Y)**torch.abs(X-Y) )) 
        return 100*gauss_loss/3072
        
def addNoise(image, l):
    
    f = (torch.max(image) - torch.min(image) )*l/100
    Noise = f*torch.randn(image.shape)
    image = image+Noise
    return image
    
def train(Model,epochs):
    
    optimizer = optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
    Loss_fn = Loss()
    Losses = []
    
    for epoch in range(epochs):
        
        for u, (image,id) in enumerate(trainloader):
            
            im = image[0]
            im = im.view(1,3,32,32)
            imN = addNoise(im,10)
            im, imN = im.cuda(), imN.cuda()
            imN = Model(imN)
            loss = Loss_fn(imN,im)
            
        loss.backward()
        optimizer.step()
        Losses.append(loss)
        
        print('Epoch: ', epoch,  'Loss: ', loss)
            
    return Losses
    
Net = AE()
if(int(input('Load Trained? '))):
    Net.load_state_dict(torch.load(input('Enter BaseModel Name: ')))
Net1 = Net.cuda()
epochs = int(input('Enter Epochs: '))
model_title = input('Enter Model title: ')

start_time = time.time()
Loss_trends = np.array(train(Net1,epochs))
end_time = time.time()
L = pd.DataFrame()
L['Losses'] = Loss_trends
L.to_csv(model_title + 'Losses.csv')
torch.save(Net1.state_dict(), model_title + '.pth.tar')
print('........')
print('Training Time: ', end_time - start_time, ' seconds')
print('Saved at: ', model_title + '.pth.tar')
 
