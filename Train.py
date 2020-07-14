import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

class AE(nn.Module):

    def __init__(self):
    
        super(AE, self).__init__()
        
        self.size = 32*32*3
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.l1 = nn.Linear(self.size,self.size/4,True)
        self.l2 = nn.Linear(self.size/4,self.size,True)
        self.l3 = nn.Linear(self.size,self.size/4,True)
        self.out = nn.Linear(self.size/4,self.size,True)       

    def forward(self, image):
    
        im = image.flatten()
    
        im = self.l1(im)
        im = self.dropout(im)
        
        im = self.l2(im)
        im = self.dropout(im)
        
        im = self.l3(im)
        im = self.dropout(im)
        
        im = self.l4(im)
        im = self.dropout(im)
        
        return im
        
class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
       
    def forward(self, image, result):

        res = result.reshape(3,32,32)
        X = image
        Y = res
        gauss_loss = torch.sum(1-torch.exp(-(X-Y)**2))
        return gauss_loss
        
def addNoise(image, l):
    
    f = (torch.max(image) - torch.min(image) )*l/100
    Noise = f*torch.randn(image.shape)
    image = image+Noise
    
    
def train(Model,epochs):
    
    optimizer = optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
    Loss_fn = Loss()
    Losses = []
    ims = 0
    
    for epoch in range(epochs):
        
        for batch_num, images in enumerate(trainloader):
            
            ims = 0
            for im in images:

                imN = addNoise(im)
                imN = Model(imN)
                loss += Loss_fn(im,imN)
                ims+=1
        
            loss = loss/ims
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
