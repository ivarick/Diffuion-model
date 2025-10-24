import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # downsampling path
        for feature in features:
            self.downs.append(Conv(in_channels, feature))
            in_channels = feature
        
        # bottleneck
        self.bottleneck = Conv(features[-1], features[-1] * 2)
        
        # upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(Conv(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        
        # time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, features[-1] * 2),
            nn.ReLU(),
            nn.Linear(features[-1] * 2, features[-1])
        )

    def forward(self, x, t):
        skip_connections = []
        
        # embd time
        t_emb = self.time_emb(t.unsqueeze(1).float())
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1)
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = x + t_emb  # embd before bottleneck
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)
