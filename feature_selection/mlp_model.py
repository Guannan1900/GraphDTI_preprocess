"""
Created on Sat Feb 29 11:35:02 2020

@author: lgn
"""

import torch.nn as nn
import torch.nn.functional as F

class mlp_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 2nd hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.softmax(output)


        return output
