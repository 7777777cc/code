import os
import numpy as np 

import torch
import torch.nn as nn

# import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.metrics import normalized_mutual_info_score
import MutualInformation
device = 'cuda:0'

img1 = Image.open('grad1.jpg').convert('L')
img2 = Image.open('grad.jpg').convert('L')

img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

before_grad = torch.load('before_grad.pt')
after_grad = torch.load('after_grad.pt')

# scale
# before_grad = before_grad * (255 / before_grad.max())
# after_grad = after_grad * (255 / after_grad.max())

before_grad = (before_grad).unsqueeze(dim=0).unsqueeze(dim=0)
after_grad = (after_grad).unsqueeze(dim=0).unsqueeze(dim=0)
print(before_grad.mean(), before_grad.std(), before_grad.max(), before_grad.min())
print(after_grad.mean(), after_grad.std(), after_grad.max(), after_grad.min())

a= torch.ones((300,300), requires_grad=False).cuda()
b = torch.zeros((300,300),requires_grad =True).cuda()
a[:,1:3] = torch.zeros((1,1)).cuda()
a = (a).unsqueeze(dim=0).unsqueeze(dim=0)
b = (b).unsqueeze(dim=0).unsqueeze(dim=0)

# Pair of different images, pair of same images
# shape: (2, 1, 300, 300)s
MI = MutualInformation.MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)
score = MI(img1, img2)
scores = MI(img1, img1)
# print(score, scores)

# print( 'a MI', MI(a,a))  
# print( 'b MI', MI(b,b))  
# print('a and b MI', MI(a,b))

ls = nn.MSELoss()
before_loss = ls(before_grad,before_grad)
after_loss = ls(after_grad,after_grad)
before_after_loss = ls(before_grad,after_grad)

print('\n')
print('before and before', MI(before_grad, before_grad), before_loss)
print('after and after', MI(after_grad, after_grad),after_loss)
print('before and after', MI(before_grad,after_grad), before_after_loss)





######Fourier Transform########
fourier_img1 = torch.fft.fft2(img1)
fourier_img2 = torch.fft.rfft2(img1)

# import pdb;pdb.set_trace()