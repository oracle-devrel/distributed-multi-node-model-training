import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 

# Define the hyperparameters and data loaders
batch_size = 64
learning_rate = 0.001
num_epochs = 10

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', prertained=False)

# Change model head classifier to dataset num_classes
model.fc = nn.Linear(512, 10)

# Move the model to device
model = model.to(0)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print('Training started')
progress_bar = tqdm(range(num_epochs))
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(0), labels.to(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Print statistics
    progress_bar.update(1)
    progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} train_loss: {running_loss / len(train_loader)}")
    running_loss = 0.0

print("Saving model...")
torch.save(model.state_dict(), "model.pth")

print('\nFinished training')