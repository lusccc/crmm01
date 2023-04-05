import torch
import torch.nn as nn
import torch.nn.functional as F


class JointFeatureExtractor(nn.Module):
    def __init__(self, modality_feat_dims, hidden_dim, output_dim):
        super().__init__()
        self.fcs = nn.ModuleDict()
        for modality, feat_dim in modality_feat_dims.items():
            self.fcs[modality] = nn.Linear(feat_dim, hidden_dim)

        self.num_modalities = len(modality_feat_dims)
        self.conv1d = nn.Conv1d(in_channels=self.num_modalities, out_channels=1, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.output_layer = nn.Linear(int(hidden_dim / 2), output_dim)

    def forward(self, x):
        x_aligned = []  # aligned features to hidden_dim
        for modality, x in x.items():
            # Apply linear transformation
            x = self.fcs[modality](x)
            x_aligned.append(x.unsqueeze(1))  # Add channel dimension (B, C, L) -> (B, 1, L)

        # Stack modality features along the channel dimension
        x_stacked = torch.cat(x_aligned, dim=1)  # (B, num_modalities, L)

        # Apply convolution for feature fusion
        x_res = x_stacked
        x_fused = self.conv1d(x_stacked)

        # Add residual connection
        x_fused = x_fused + x_res[:, :1, :]  # Make sure dimensions match before adding

        # Apply Max Pooling
        x_fused = self.max_pool(x_fused)

        # Apply ReLU activation
        x_fused = F.relu(x_fused)
        x_fused = x_fused.squeeze(1)  # Remove channel dimension (B, 1, L) -> (B, L)
        output = self.output_layer(x_fused)

        return output


if __name__ == '__main__':
    model = JointFeatureExtractor(modality_feat_dims={'audio': 128, 'text': 768}, hidden_dim=128)
    print(model)
