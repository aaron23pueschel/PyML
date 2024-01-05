
## Aaron Pueschel
## Final Project Optimization
#############################################

## Imports: make sure you have Torch, numpy, matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import randrange
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as fn



## DEFINE CLASSES
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim,bias=False)
        self.act1 = nn.SiLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.act2 = nn.SiLU()
        self.layer3 = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.act3 = nn.SiLU()
        self.layer4 = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.act4 = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim,output_dim,bias=False)

        

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x) 
        x = self.layer2(x)
        x = self.act2(x) 
        x = self.layer3(x) 
        x = self.act3(x) 
        x = self.layer4(x)
        x = self.act4(x) 
        x = fn.softmax(self.out_layer(x),dim=1)
        return x

## Create a dataloader class for minibatching
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
############################################
    
    # FUNCTIONS   

#############################################

class functions():
    def generate_data(xrange,yrange,num_vals):
        x1 = np.linspace(xrange[0],xrange[1],num_vals)
        x2 = np.linspace(yrange[0],yrange[1],num_vals)
        np.random.shuffle(x1)
        np.random.shuffle(x2)
        return x1,x2
        
    def himmelblau_noisy(x1_train, x2_train, noise_level=0.001):
        true_value = (x1_train**2 + x2_train - 11)**2 + (x1_train + x2_train**2 - 7)**2
        noise = np.random.normal(0, noise_level, true_value.shape)
        return true_value + noise

    def rosenbrock_noisy(x1, x2, noise_level=0.001):
        a = 1
        b = 3
        true_value = (a * (x2 - x1**2)**2) + (b * (1 - x1)**2)
        noise = np.random.normal(0, noise_level, true_value.shape)
        return true_value + noise
    def ackley(x1, x2, noise_level=0.001):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = x1**2 + x2**2
        sum2 = np.cos(c * x1) + np.cos(c * x2)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / 2))
        term2 = -np.exp(sum2 / 2)
        result = term1 + term2 + a + np.exp(1)
        noise = np.random.normal(0, noise_level)
        return result + noise


