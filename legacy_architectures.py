'''
Models were trained by 
pytorch_regression_system_v2.2.py
The architectures are defined in that file and saved in .pkl files.
'''
import torch
import torch.nn as nn

class DeepFFN(nn.Module):
    """Deep Feedforward Network"""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super(DeepFFN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WideNetwork(nn.Module):
    """Wide Network with fewer layers"""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
        super(WideNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4 if i == 0 else 0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.block(x) + x)

class ResNet(nn.Module):
    """Residual Network"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_block = ResidualBlock(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block(x)
        return self.output_layer(x)

class EnsembleNet(nn.Module):
    """Ensemble-like Network with multiple pathways"""
    def __init__(self, input_dim, output_dim, path1_dim=32, path2_dim=64):
        super(EnsembleNet, self).__init__()
        # Pathway 1: Deep and narrow
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, path1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(path1_dim, path1_dim // 2),
            nn.ReLU()
        )
        # Pathway 2: Shallow and wide
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, path2_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Combine
        combined_dim = path1_dim // 2 + path2_dim
        self.output = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        combined = torch.cat([p1, p2], dim=1)
        return self.output(combined)

class AutoEncoderNet(nn.Module):
    """Autoencoder-style Network"""
    def __init__(self, input_dim, output_dim, bottleneck_dim=16):
        super(AutoEncoderNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# 別名對照表 (確保 model_inference_ui.py 解析 .md 中的名字時能正確對應)
AutoEncoder_Net = AutoEncoderNet
Wide_Network = WideNetwork
Deep_Network = DeepFFN
Ensemble_Net = EnsembleNet
