import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl

class NormalEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(NormalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=False)

        # init
        nn.init.normal_(self.emb_layer.weight, 0.0, 1.0)

    def forward(self, x):
        emb = self.emb_layer(x)
        return emb

class OrthogonalEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(OrthogonalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=False)

        # init
        nn.init.orthogonal_(self.emb_layer.weight)
    
    def forward(self, x):
        emb = self.emb_layer(x)
        return emb
        
class EquivariantEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EquivariantEmbedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=False)

        # init
        nn.init.normal_(self.emb_layer.weight[:,0], 0.0, 1.0)
        emb_column = self.emb_layer.weight[:,0]
        with torch.no_grad():
            for i in range(1, self.input_dim):
                self.emb_layer.weight[:,i].data.copy_(torch.roll(emb_column, i, 0))

    def reset_parameters(self):
        self.emb_layer.reset_parameters()

    def forward(self, x):
        emb = self.emb_layer(x)
        return emb
