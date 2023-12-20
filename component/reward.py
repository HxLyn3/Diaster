import torch
import numpy as np
import torch.nn as nn
from .network import MLP
from torch.nn import functional as F

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class RNNReward(nn.Module):
    """ rnn-based reward model """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        rnn_num_layers=1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, 2)

        self.register_parameter('max_logvar', nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(1) * -10, requires_grad=True))

        self.hidden = None

    def init_hidden(self):
        self.hidden = None

    def forward(self, input):
        batch_size, num_timesteps, _ = input.shape

        rnn_output, self.hidden = self.rnn_layer(input, self.hidden)
        rnn_output = rnn_output.reshape(-1, self.hidden_dim)
        output = self.output_layer(rnn_output)
        output = output.view(batch_size, num_timesteps, -1)
        mean, logvar = torch.chunk(output, 2, dim=-1)

        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.sqrt(torch.exp(logvar))
        output = mean + torch.normal(0, 1, size=mean.shape, device=input.device)*std
        return output.view(batch_size, num_timesteps)

class MLPReward(nn.Module):
    """ r(s,a) """
    def __init__(self, obs_shape, hidden_dims, action_dim):
        super(MLPReward, self).__init__()
        self.backbone = MLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
        latent_dim = getattr(self.backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1)

    def forward(self, obs, actions):
        """ return r(s,a) """
        net_in = torch.cat([obs, actions], dim=1)
        logits = self.backbone(net_in)
        values = self.last(logits)
        return values
