import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        out = self.dropout2(out)
        return out


class NumFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(NumFeatureExtractor, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # 中间层
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate))

        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out)

        for block in self.residual_blocks:
            out = block(out)

        out = self.output_layer(out)
        return out

    def loss_function(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        l1_reg_loss = torch.tensor(0., requires_grad=True).to(y_pred.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg_loss += torch.norm(param, 1)
        total_loss = mse_loss + 0.0001 * l1_reg_loss
        return total_loss
