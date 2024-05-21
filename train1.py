# Libraries used in the distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os

# Initializes the distributed backend which will take care of synchronizing nodes/GPUs
# only works with torch.distributed.launch // torch.run

# Set the URL for communication
dist_url = "env://" # default

# Retrieve world_size, rank and local_rank
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ["RANK"])
local_rank = int(os.environ['LOCAL_RANK'])

# Initialize the process group
dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

# synchronizes all the threads to reach this point before moving on
dist.barrier()

#Signle GPU code
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

train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=False)
train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

#Signle GPU code
#train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

# Change model head classifier to dataset num_classes
model.fc = nn.Linear(512, 10)

# Move the model to device
model.to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

#Signle GPU code
#model = model.to(0)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print('Training started')
progress_bar = tqdm(range(num_epochs))
for epoch in range(num_epochs):
    train_loader.sampler.set_epoch(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(local_rank), labels.to(local_rank)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

#Signle GPU code
#for epoch in range(num_epochs):
#    running_loss = 0.0
#    for i, data in enumerate(train_loader):
#        inputs, labels = data
#        inputs, labels = inputs.to(0), labels.to(0)
#        optimizer.zero_grad()
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()


# Print statistics
progress_bar.update(1)
progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} train_loss: {running_loss / len(train_loader)}")
running_loss = 0.0

if local_rank == 0:
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print('\nFinished training')

#Signle GPU code
#print("Saving model...")
#torch.save(model.state_dict(), "model.pth")
#print('\nFinished training')



