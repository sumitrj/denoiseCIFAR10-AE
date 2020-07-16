# Denoise CIFAR10 dataset using autoencoders

## Goal:
To make an autoencoder which takes input of images and returns denoised images.

## Approach:
The images in CIFAR-10 dataset are of size 32*32*3. 
Since the image size is small, fully connected encoders can be used. 
Convolutional autoencoders can also be used to reduce image size further and check feature extraction.

### Fully connected autoencoder:

4 layers of fully connected nature with 3072 units in each layer is used. 1 Input layer, 2 hidden layers, 1 output layer.

### Noise source:

Random numbers of normal distribution are used to introduce noise.
Noise, as a function of noise level l (%) is given as:
  
    f = ( (max(image) - (image) )*l/100 )*randn(image.shape) + image


To check noise reduction, following loss function is used:

    loss = log(sum( abs(X-Y)**torch.abs(X-Y) )) *100/3072

Following is a plot of loss function against noise level l:

![Plot of Loss vs noise level for a single image](https://github.com/sumitrj/dcf10/blob/master/plotimages/download.png?raw=true)

