import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

num_epochs = 10
num_classes = 5
learning_rate = 0.001

# define the model
class ConvNet(nn.module):
	def __init__(self):
		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
			)