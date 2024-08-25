import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import CNN_CUSTOM, CNN_STOCK
import time

# Training settings
batch_size = 64
epochs = 1
learning_rate = 0.01

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_custom = CNN_CUSTOM().to(device)
model_stock = CNN_STOCK().to(device)

optimizer_custom = optim.SGD(model_custom.parameters(), lr=learning_rate)
optimizer_stock = optim.SGD(model_stock.parameters(), lr=learning_rate)

# Training function with timing
def train_model(model, optimizer, model_name):
    model.train()
    start_time = time.time()  # Start timing
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')
    
    end_time = time.time()  # End timing
    print(f"{model_name} training completed in {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    # Train and time the custom kernel model
    print("Training model with custom CUDA kernel:")
    train_model(model_custom, optimizer_custom, "Custom Kernel Model")
    
    # Train and time the stock PyTorch model
    print("\nTraining model with PyTorch's built-in matmul:")
    train_model(model_stock, optimizer_stock, "Stock PyTorch Model")
