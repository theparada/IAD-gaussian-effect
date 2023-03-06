import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_number, l1 = 64, l2 = 64, l3 =64):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(input_number, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu_stack(x)
        return x


class NeuralNetworkDropout(nn.Module):
    def __init__(self, input_number, l1 = 64, l2 = 64, l3 =64):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(input_number, l1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(l3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu_stack(x)
        return x


class NeuralNetworkMoreLayers(nn.Module):
    def __init__(self, input_number, nodes=64):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(input_number, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu_stack(x)
        return x


#customized BCELoss
def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss