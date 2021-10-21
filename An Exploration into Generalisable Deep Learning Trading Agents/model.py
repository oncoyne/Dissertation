import torch
from torch import nn
from typing import NamedTuple
#from torch.nn import functional as F

#Instantiate the TraderNet Class
class TraderNet(nn.Module):
    def __init__(self, number_features: int):
        super().__init__()
        
        #10 LSTM Units
        self.lstm = nn.LSTM(
            input_size= number_features,
            hidden_size=10,
        )

        #5 fully connected neurons
        self.fc1 = nn.Linear(
            in_features = 10,
            out_features = 5,
        )

        #3 fully conncected neurons
        self.fc2 = nn.Linear(
            in_features = 5,
            out_features = 3,
        )

        #1 neuron
        self.fc3 = nn.Linear(
            in_features = 3,
            out_features = 1,
        )

        #Defined activation functions instead of using functional because it was causing errors with gradientSHAP
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()


    #Forward pass of the model
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        #Unsqueeze the LSTM input sequence to just a single value
        features = torch.unsqueeze(features, 0)

        #Apply each step of the model in the forward pass
        x, (hn, cn) = self.lstm(features)
        x = self.relu1(x)

        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)
        x = self.relu3(x)

        x = self.fc3(x)

        #Re-squeeze for convenient output
        x = torch.squeeze(x)

        return x

