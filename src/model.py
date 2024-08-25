# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # Importing F here for usage within the model
import matrix_mul_extension  # Import the custom CUDA extension

class CNN_CUSTOM(nn.Module):
    def __init__(self):
        super(CNN_CUSTOM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Use custom CUDA matrix multiplication in some part of the forward pass
        if x.is_cuda:
            A = torch.randn(1024, 1024).cuda()
            B = torch.randn(1024, 1024).cuda()
            result = matrix_mul_extension.matrix_mul(A, B)
            #print("Custom CUDA matrix multiplication result:", result)

        return F.log_softmax(x, dim=1)

class CNN_STOCK(nn.Module):
    def __init__(self):
        super(CNN_STOCK, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Example matrix multiplication using PyTorch's built-in function (no custom kernel)
        if x.is_cuda:
            A = torch.randn(1024, 1024, device=x.device)  # Random matrix A
            B = torch.randn(1024, 1024, device=x.device)  # Random matrix B
            result = torch.matmul(A, B)  # PyTorch's built-in matrix multiplication

        
        return F.log_softmax(x, dim=1)