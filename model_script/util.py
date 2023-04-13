

import torch.nn as nn
import torch
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


class DenseReluDense(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class LayerFF(nn.Module):

    def __init__(self, d_model, d_ff=1024, layer_norm_epsilon=1e-7, dropout_rate=0.15):
        super().__init__()
        self.DenseReluDense = DenseReluDense(d_model, d_ff, dropout_rate)
        self.layer_norm = LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output



if __name__ == '__main__':

    x=torch.rand((2,3,768))
    lff=LayerFF(d_model=x.size()[-1])
    y=lff(x)
    print(y.size())
