import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Conv layer 1: Input size (1, 64, 64) -> (64, 32, 32)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 2: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (batch, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 3: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (batch, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 4: (256, 8, 8) -> (512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (batch, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer to output a single value per image
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output: (batch, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)  # Flatten to [batch_size]

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Conv layer 1: Input size (1, 64, 64) -> (64, 32, 32)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 2: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (batch, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 3: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (batch, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv layer 4: (256, 8, 8) -> (512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (batch, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Adaptive pooling to resize to (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (batch, 512, 1, 1)
            
            # Final layer to output a single value per image
            nn.Conv2d(512, 1, kernel_size=1),  # Output: (batch, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)  # Flatten to [batch_size]

