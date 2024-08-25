# test.py

import torch
from src.model import CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Importing F here for usage within the model

# Load test dataset
test_dataset = datasets.MNIST('dataset', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
device = torch.device("cuda")
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_model.pth"))  # Assuming you save the model after training

# Test the model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

if __name__ == "__main__":
    test()
