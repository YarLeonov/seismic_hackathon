import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(conv3D => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        """
        Simple 3D U-Net implementation for block-based seismic fault detection.
        in_channels: 1 (seismic amplitude block)
        out_channels: 1 (probability of fault)
        """
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Build Encoder
        for feature in features:
            self.encoder.append(DoubleConv3D(in_channels, feature))
            in_channels = feature
            
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Build Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Skip connections going from last to first
        skip_connections = skip_connections[::-1]

        # Decoder path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i//2]
            
            # Ensure sizes match (handling arbitrary crop sizes)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i+1](concat_skip)

        return self.final_conv(x)

if __name__ == "__main__":
    # Test inference output size
    model = UNet3D(in_channels=1, out_channels=1)
    dummy_input = torch.randn((1, 1, 64, 64, 64)) # Batch, Channels, D, H, W
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
