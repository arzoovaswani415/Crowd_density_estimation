import torch
import torch.nn as nn

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU()
        )

        self.fuse = nn.Conv2d(48, 1, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1, x2, x3], dim=1)  # ✅ FIX HERE
        x = self.fuse(x)
        return x
