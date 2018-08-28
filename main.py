from models import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.utils import make_grid
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
#parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')

opt = parser.parse_args()

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download dataset
if opt.dataset == 'mnist':
  dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
elif opt.dataset == 'cifar10':
  dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=True,\
                         transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

# train method
def train(model, optimiser, criterion, epochs):
  losses = []
  for epoch in range(epochs):
    for idx, (data, label) in enumerate(dataloader):
      model.zero_grad()
      x = data.to(device)
      output = model(x)
      loss = criterion(output, x)
      losses.append(loss)
      loss.backward()
      optimiser.step()
      print('Done: [%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, idx, len(dataloader), loss.item()))
  return losses

# initialize
cae = ConvAutoEncoder().to(device)
optimizer = torch.optim.Adam(cae.parameters(), lr = opt.lr, weight_decay=1e-5)
loss_func = nn.MSELoss()

# train
losses = train(cae, optimizer, loss_func, opt.niter)

# plot
plt.figure()
plt.plot(losses)
