# train.py

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Importing F here for usage within the model
from src.model import CNN_CUSTOM
from tqdm import tqdm

# Training settings
batch_size = 64
epochs = 1
learning_rate = 0.01
verbosity = 0

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('dataset', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer
device = torch.device("cuda")
model = CNN_CUSTOM().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Check if cuDNN is enabled
if torch.backends.cudnn.enabled:
    print(f"cuDNN is enabled. cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("cuDNN is not enabled.")

# Synchronize GPU (to ensure accurate measurement)
torch.cuda.synchronize()
# Start time (GPU timer)
start_gpu = torch.cuda.Event(enable_timing=True)
end_gpu = torch.cuda.Event(enable_timing=True)
start_gpu.record()

# Training function
def train():
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)


            if verbosity==3:
                # Check if data is on the GPU
                if data.is_cuda:
                    print(f"Batch {batch_idx}: Data is on GPU.")

                # Check if cuDNN will be used (only applicable on the GPU)
                if torch.backends.cudnn.is_available() and data.is_cuda:
                    print(f"Batch {batch_idx}: cuDNN will be used.")

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')

    # End time (GPU timer)
    end_gpu.record()

    # Wait for everything to finish
    torch.cuda.synchronize()

    # Calculate time
    elapsed_time = start_gpu.elapsed_time(end_gpu)  # Time in milliseconds
    print(f"Custom CUDA matrix multiplication time: {elapsed_time:.6f} milliseconds")

    # Save the trained model
    torch.save(model.state_dict(), "cnn_custom_model.pth")
    print("Model saved to cnn_model.pth")

if __name__ == "__main__":
    train()
