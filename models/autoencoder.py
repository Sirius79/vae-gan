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

class ConvAutoEncoder(nn.Module):
  
  def __init__(self, nch):
    
    super(ConvAutoEncoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(nch, 512, 4, 2, 0, bias=False), 
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 8, 4, 2, 1, bias=False),
        nn.Sigmoid()
     )
    
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(8, 32, 4, 2, 0, bias=False), 
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(32, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(512, nch, 4, 2, 1, bias=False),
        nn.Sigmoid()
     )
    
  def forward(self, input):
    encoded = self.encoder(input)
    output = self.decoder(encoded)
    return output
