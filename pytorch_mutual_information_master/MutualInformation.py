import os
import numpy as np 

import torch
import torch.nn as nn

# import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.metrics import normalized_mutual_info_score


class MutualInformation(nn.Module):

	def __init__(self, sigma=0.4, num_bins=256, normalize=True):
		super(MutualInformation, self).__init__()

		self.sigma = 2*sigma**2
		self.num_bins = num_bins
		self.normalize = normalize
		self.epsilon = 1e-10
		self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device='cuda').float(), requires_grad=True)


	def marginalPdf(self, values):

		residuals = values - self.bins.unsqueeze(0).unsqueeze(0)  
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2)) ###MISE
		# kernel_values = torch.exp(abs(-0.5*(residuals / self.sigma))) #####MIAE doenst work
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
	
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
		pdf = joint_kernel_values / normalization

		return pdf



	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''

		# Torch tensors for images between (0, 1)
		input_1 = input1*255*5
		input_2 = input2*255*5
		input1 = torch.cat([input_1, input_2])
		input2 = torch.cat([input_2, input_2])
		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''
		return self.getMutualInformation(input1, input2)

