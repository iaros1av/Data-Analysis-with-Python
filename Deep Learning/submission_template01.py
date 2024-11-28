import numpy as np
import torch
from torch import nn

from torch import nn
def create_model():
    model = nn.Sequential(nn.Leaner(784,256, bias=True), 
                        nn.ReLU(),
                        nn.Leaner(256,16, bias=True),
                        nn.ReLU(),
                        nn.Leaner(16,10, bias=True)
    )
    return model

def count_parameters(model):
    counter = model.parameters(model)
    return counter
