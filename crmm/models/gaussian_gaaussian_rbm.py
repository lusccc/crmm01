import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GaussianGaussianRBM(nn.Module):
    """
    energy function:
    $$E(v,h)=\dfrac{1}{2}\sum_i\dfrac{(v_i-b_i)^2}{\sigma_i^2}-\dfrac{1}{2}\sum_j\dfrac{(h_j-c_j)^2}{\tau_j^2}-
    \sum_{i,j}\dfrac{v_iW_{ij}h_j}{\sigma_i\tau_j}\quad\text{}$$

    input X,
    X -> encoder -> v0 -> h0 -> decoder -> v1
    loss is calculated between v0 and v1
    """

    def __init__(self, name, visible_units, hidden_units, dropout=0.5, encoder=None, pretrained=False):
        super(GaussianGaussianRBM, self).__init__()
        self.name = name
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.encoder = encoder
        self.pretrained = pretrained

        self.visible_input_bn = nn.BatchNorm1d(self.visible_units)

        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(visible_units))
        self.b_h = nn.Parameter(torch.zeros(hidden_units))
        self.sigma_v = nn.Parameter(torch.ones(visible_units), requires_grad=True)
        self.tau_h = nn.Parameter(torch.ones(hidden_units), requires_grad=True)

    def visible_to_hidden(self, v):
        v = self.visible_input_bn(v)
        h_means = F.relu(F.linear(v / self.sigma_v ** 2, self.W, self.b_h))
        return h_means

    def hidden_to_visible(self, h):
        v_means = F.relu(F.linear(h / self.tau_h ** 2, self.W.t(), self.b_v))
        return v_means

    def check_x(self, x):
        if self.name in ['num', 'cat', 'joint']:
            # these feature extractors forward accept `x` as parameters
            return {'x': x}
        elif self.name == 'text':
            # text feature extractor forward accept `input_ids,attention_mask` as parameters, already is dict
            return x

    def pretrain_step(self, x):
        """
        Pretrain a step of RBM. Only called in pretraining!
        """
        x = self.check_x(x)  # is a dict! to easily pass to encoder forward!
        x = self.encoder(**x)

        h0_means = self.visible_to_hidden(x)
        h0 = h0_means + torch.randn_like(h0_means) * self.tau_h

        # Apply dropout to the hidden layer
        h0 = F.dropout(h0, p=self.dropout, training=True)

        v1_means = self.hidden_to_visible(h0)
        v1 = v1_means + torch.randn_like(v1_means) * self.sigma_v

        h1_means = self.visible_to_hidden(v1)
        h1 = h1_means + torch.randn_like(h1_means) * self.tau_h

        positive_phase = torch.einsum('ij,ik->ijk', x, h0).mean(0)
        negative_phase = torch.einsum('ij,ik->ijk', v1, h1).mean(0)

        loss = (positive_phase - negative_phase).norm()
        return loss

    def extra_features_step(self, x):
        """
        Get hidden value (used as feature).
        """
        x = self.check_x(x)  # is a dict! to easily pass to encoder forward!
        x = self.encoder(**x)
        h0_means = self.visible_to_hidden(x)
        return h0_means

    def forward(self, x):
        output = self.extra_features_step(x) if self.pretrained else self.pretrain_step(x)
        return output
