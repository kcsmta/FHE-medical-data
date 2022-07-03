from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts


class Covid19_tenseal_neural_network(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        out = self.fc1(x)
        # using square activation function as a replacement of sigmoid activation function
        out = out*out
        out = self.fc2(out)
        return out

class Enc_Covid19_tenseal_neural_network:

    def __init__(self, plain_model):
        # TenSEAL processes lists and not torch tensors,
        # so we take out the parameters from the PyTorch model
        self.weight_fc1 = plain_model.fc1.weight.T.data.tolist()
        self.bias_fc1 = plain_model.fc1.bias.data.tolist()
        self.weight_fc2 = plain_model.fc2.weight.T.data.tolist()
        self.bias_fc2 = plain_model.fc2.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.mm(self.weight_fc1) + self.bias_fc1
        enc_out.square_()
        enc_out = enc_out.mm(self.weight_fc2) + self.bias_fc2
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
class Covid19_concreteml_neural_network(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        return out

class Enc_Covid19_concreteml_neural_network:

    def __init__(self, plain_model):
        # TenSEAL processes lists and not torch tensors,
        # so we take out the parameters from the PyTorch model
        self.weight_fc1 = plain_model.fc1.weight.T.data.tolist()
        self.bias_fc1 = plain_model.fc1.bias.data.tolist()
        self.weight_fc2 = plain_model.fc2.weight.T.data.tolist()
        self.bias_fc2 = plain_model.fc2.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.mm(self.weight_fc1) + self.bias_fc1
        enc_out.square_()
        enc_out = enc_out.mm(self.weight_fc2) + self.bias_fc2
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)