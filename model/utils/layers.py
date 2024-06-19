from model.utils.prompt_tools import SkelMaPLe
import torch.nn as nn

class HiddenLayer(nn.Module):
    def __init__(self, n_hidden_layers, hidden_size):
        super().__init__()
        if n_hidden_layers != 0:
            self.hidden_layers = []
            for i in range(n_hidden_layers):
                self.hidden_layers += [SkelMaPLe(hidden_size,hidden_size)]
            self.hidden_layers = nn.Sequential(*self.hidden_layers)
        else:
            self.hidden_layers = nn.Identity()

    def forward(self, x):
        return self.hidden_layers(x)

