import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import wandb


# Image display
import numpy as np
import random

train_dataset_path = '../dataset/handwritten_math_symbols/train/'
eval_dataset_path = '../dataset/handwritten_math_symbols/eval/'

model_input_shape = (32, 32)
train_dataset_transform = transforms.Compose([
    transforms.Resize((model_input_shape[0]+4, model_input_shape[0]+4)),
    transforms.RandomCrop((model_input_shape)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
#     transforms.Lambda(lambd=lambda x: 1 - x),
    transforms.RandomRotation(10, fill=1),
#     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
eval_dataset_transform = transforms.Compose([
    transforms.Resize(model_input_shape),
    transforms.Grayscale(1),
    transforms.ToTensor(),
#     transforms.Lambda(lambd=lambda x: 1 - x),
#     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_dataset_transform)
eval_dataset = torchvision.datasets.ImageFolder(root=eval_dataset_path, transform=eval_dataset_transform)

assert len(train_dataset.classes) == len(eval_dataset.classes)

batch_size = 16

training_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

validation_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=True,
)


lr = 0.1
epochs = 25
loss_function = nn.BCEWithLogitsLoss
lr_scheduler = optim.lr_scheduler.ExponentialLR
lr_gamma = 0.9
classes = train_dataset.classes
model_name = 'symbols-v1'

run = wandb.init(project="symbols", entity="kaizen", name=model_name)
wandb.config = {
    "learning_rate": lr,
    "lr_gamma": lr_gamma,
    "epochs": epochs,
    "batch_size": batch_size,
    "loss_function": loss_function,
    "lr_scheduler": lr_scheduler,
    "v_dataset__len__": len(eval_dataset),
    "v_dataloader__len__": len(validation_loader),
    "t_dataset__len__": len(train_dataset),
    "t_dataloader__len__": len(training_loader),
    "classes": classes,
    "model_name": model_name
}



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5) # 28 * 28 * 4
        self.bn1 = nn.BatchNorm2d(4)
#         self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 5, 5) # 24 * 24 * 5
        self.bn2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(5, 6, 5) # 20 * 20 * 6
        self.bn3 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 8, 5) # 16 * 16 * 8
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8, 12, 5) # 12 * 12 * 12
        self.bn5 = nn.BatchNorm2d(12)
        self.conv6 = nn.Conv2d(12, 16, 5) # 8 * 8 * 16
        self.bn6 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = (x-0.5)*2
        x = self.conv1(x) # 28 * 28 * 4
        x = self.bn1(x)
        x = F.relu(x) # 28 * 28 * 4
#         x = self.pool(x)
        x = self.conv2(x) # 24 * 24 * 5
        x = self.bn2(x)
        x = F.relu(x)  # 24 * 24 * 5
        x = self.conv3(x) # 20 * 20 * 6
        x = self.bn3(x)
        x = F.relu(x)  # 20 * 20 * 6
        x = self.conv4(x) # 16 * 16 * 8
        x = self.bn4(x)
        x = F.relu(x)  # 16 * 16 * 8
        x = self.conv5(x) # 12 * 12 * 12
        x = self.bn5(x)
        x = F.relu(x)  # 12 * 12 * 12
        x = self.conv6(x) # 8 * 8 * 16
        x = self.bn6(x)
        x = F.relu(x)  # 8 * 8 * 16
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = F.softmax(x, dim=-1)
        return x


net = Net()
criterion = loss_function()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = lr_scheduler(optimizer, gamma=lr_gamma)


for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    # runnig over batches
    total_batches = 0
    total_data = 0
    for i, data in enumerate(training_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        logits = nn.functional.one_hot(labels, len(classes)).float()
        loss = criterion(outputs, logits)
        loss.backward()
        optimizer.step()
        total_batches += 1

        wandb.log({
            'loss': loss.item(),
            'lr': float(scheduler.get_last_lr()[0]),
        })
        total_data += len(outputs)
        running_loss += loss.item()

    avg_loss = running_loss/total_batches
    running_vloss = 0.0
    vaccuracy = 0.0
    total_vdata = 0

    net.train(False) # Don't need to track gradents for validation
    for j, vdata in enumerate(validation_loader, 0):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        vlogits = nn.functional.one_hot(vlabels, len(classes)).float()
        vloss = criterion(voutputs, vlogits)

        running_vloss += vloss.item()
        vaccuracy += (voutputs.argmax(axis=1)==vlabels).sum()
        total_vdata += len(vlabels)
    net.train(True) # Turn gradients back on for training

    avg_vloss = running_vloss / len(validation_loader)
    avg_vaccuracy = float(vaccuracy)/ total_vdata
    if epoch%5==0:
        scheduler.step()
    wandb.log({
        'epoch': epoch,
        'epoch_vloss': avg_vloss,
        'epoch_vaccuracy': avg_vaccuracy * 100,
        'epoch_loss': avg_loss
    })
    artifact = wandb.Artifact('model', type='model')
    model_file = f'./models/{model_name}-{run.name}-{epoch}'
    torch.save(net.state_dict(), model_file)
    artifact.add_file(model_file)
    run.log_artifact(artifact)


print('Finished Training')
