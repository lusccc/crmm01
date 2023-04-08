import torch
import torch.nn as nn
import torch.nn.functional as F


# class JointFeatureExtractor(nn.Module):
#     def __init__(self, modality_feat_dims):
#         super().__init__()
#         self.modality_feat_dims = modality_feat_dims
#         self.ws = nn.ParameterDict()  # weights for modality features; i.e. weighted fusion
#         for modality, feat_dim in modality_feat_dims.items():
#             self.ws[modality] = nn.Parameter(torch.ones(1))
#
#     def forward(self, x):
#         x_weighted = []
#         for modality, x in x.items():
#             x = x * self.ws[modality]  # weighted
#             x_weighted.append(x)
#
#         x_concat = torch.cat(x_weighted, dim=1)
#         return x_concat
#
#     def get_output_dim(self):
#         return sum(self.modality_feat_dims.values())


class JointFeatureExtractor(nn.Module):
    def __init__(self, modality_feat_dims, hidden_dims, dropout):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.fcs = nn.ModuleDict()
        self.ws = nn.ParameterDict()  # weights for modality features; i.e. weighted fusion
        for modality, feat_dim in modality_feat_dims.items():
            fc = []
            input_dim = feat_dim
            for i, hidden_dim in enumerate(hidden_dims):
                fc.append(nn.Linear(input_dim, hidden_dim))
                fc.append(nn.ReLU(inplace=True))
                fc.append(nn.Dropout(p=dropout))
                input_dim = hidden_dim
            self.fcs[modality] = nn.Sequential(*fc)
            self.ws[modality] = nn.Parameter(torch.ones(1))

        self.num_modalities = len(modality_feat_dims)
        self.conv1d = nn.Conv1d(in_channels=self.num_modalities, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_aligned = []  # aligned features to hidden_dim
        for modality, x in x.items():
            # Apply linear transformation
            x = self.fcs[modality](x)
            x = x * self.ws[modality]  # weighted
            x_aligned.append(x.unsqueeze(1))  # Add channel dimension (B, C, L) -> (B, 1, L)

        # Stack modality features along the channel dimension
        x_stacked = torch.cat(x_aligned, dim=1)  # (B, num_modalities, L)

        # Apply convolution for feature fusion
        x_res = x_stacked
        x_fused = self.conv1d(x_stacked)

        # Add residual connection
        # x_fused = x_fused + x_res[:, :1, :]  # Make sure dimensions match before adding
        # x_fused = x_fused + sum([x_res[:, i, :] for i in range(1, self.num_modalities)]).unsqueeze(1)
        x_fused = x_fused + torch.mean(x_res, dim=1, keepdim=True)
        # Apply Max Pooling
        x_fused = self.pool(x_fused)

        # Apply ReLU activation
        x_fused = F.relu(x_fused)
        x_fused = x_fused.squeeze(1)  # Remove channel dimension (B, 1, L) -> (B, L)
        output = self.flatten(x_fused)

        return output

    def get_output_dim(self):
        return 1 * self.hidden_dims[-1] // 2


if __name__ == '__main__':
    model = JointFeatureExtractor(modality_feat_dims={'audio': 128, 'text': 768}, hidden_dim=128)
    print(model)
