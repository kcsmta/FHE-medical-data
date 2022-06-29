from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts


class Covid19_neural_network(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

class Encrypted_Covid19_neural_network:

    def __init__(self, plain_model):
        # TenSEAL processes lists and not torch tensors,
        # so we take out the parameters from the PyTorch model
        self.weight_fc1 = plain_model.fc1.weight.data.tolist()[0]
        self.bias_fc1 = plain_model.fc1.bias.data.tolist()
        self.weight_fc2 = plain_model.fc2.weight.data.tolist()[0]
        self.bias_fc2 = plain_model.fc2.bias.data.tolist()
        
    def forward(self, enc_x):
        # We don't need to perform sigmoid as this model
        # will only be used for evaluation, and the label
        # can be deduced without applying sigmoid
        enc_out = enc_x.dot(self.weight_fc1) + self.bias_fc1
        enc_out = enc_out.dot(self.weight_fc2) + self.bias_fc2
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    ################################################
    ## You can use the functions below to perform ##
    ## the evaluation with an encrypted model     ##
    ################################################
    
    def encrypt(self, context):
        self.weight_fc1 = ts.ckks_vector(context, self.weight_fc1)
        self.bias_fc1 = ts.ckks_vector(context, self.bias_fc1)
        self.weight_fc2 = ts.ckks_vector(context, self.weight_fc2)
        self.bias_fc2 = ts.ckks_vector(context, self.bias_fc2)
        
    def decrypt(self, context):
        self.weight_fc1 = self.weight_fc1.decrypt()
        self.bias_fc1 = self.bias_fc1.decrypt()
        self.weight_fc2 = self.weight_fc2.decrypt()
        self.bias_fc2 = self.bias_fc2.decrypt()