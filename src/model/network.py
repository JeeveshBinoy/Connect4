import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Connect4Net(nn.Module):
    def __init__(self, rows=6, cols=7, num_res_blocks=9, num_channels=128):
        super(Connect4Net, self).__init__()
        self.cols = cols
        self.rows = rows
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels, num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, cols)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * rows * cols, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch_size, 2, rows, cols)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Note: Apply softmax outside if needed, or here
        # Log-softmax is often used for NLLoss
        p = F.log_softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
