# Define a custom Pytorch Dataset class to use in our model
# Do some preprocessing (resize imgs, add labels) to our data
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import PIL
from PIL import Image
import cv2
import os, glob

# define custom dataset class
class EyeDataset(Dataset):
	
	def __init__(self, path):

		# get size of dataset
		self.left = [(path + '/left/' + name, 0) for name in os.listdir(path + '/left/') if '.png' in name]
		self.up = [(path + '/up/' + name, 1) for name in os.listdir(path + '/up/') if '.png' in name]
		self.right = [(path + '/right/' + name, 2) for name in os.listdir(path + '/right/') if '.png' in name]
		self.down = [(path + '/down/' + name, 3) for name in os.listdir(path + '/down/') if '.png' in name]
		self.center = [(path + '/center/' + name, 4) for name in os.listdir(path + '/center/') if '.png' in name]

		# store (filename, label) as concise way of holding datums
		# 0 = left, 1 = up, 2 = right, 3 = down, 4 = center
		self.names_array = self.left + self.up + self.right + self.down + self.center
		# print(self.names_array)
		self.data_len = len(self.left) + len(self.right) + len(self.up) + len(self.down) + len(self.center)

		# # define transforms
		self.to_tensor = transforms.ToTensor()
		self.resize = transforms.Resize((140, 250))

	def __getitem__(self, index):
		# load image, apply transforms, return (Tensor, label)

		img_path, label = self.names_array[index]
		
		img = Image.open(img_path)
		img = self.resize(img)
		img = self.to_tensor(img)

		return (img, label)


	def __len__(self):
		return self.data_len


# ### TESTING ######
# d = EyeDataset('../classification_data/dataset/test')
# print(len(d))
# img, label = d.__getitem__(40)
# print(img, label)
# img.show()


