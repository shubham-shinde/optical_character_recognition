import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data_loaderv1 import load_dataloader
import wandb

batch_size = 32
model_input_shape = (1, 32, 32)
extra_pixels_before_crop = 4
fill = 1
random_rotation_train = 10

data_loaders, datasets, data_loader_version = load_dataloader(
    batch_size=batch_size,
    model_input_shape = model_input_shape,
    extra_pixels_before_crop = extra_pixels_before_crop,
    random_rotation_train = random_rotation_train,
    fill = fill
)
train_dataset, eval_dataset = datasets
training_loader, validation_loader = data_loaders

lr = 0.15
epochs = 30
loss_function = nn.BCEWithLogitsLoss
lr_scheduler = optim.lr_scheduler.ExponentialLR
lr_gamma = 0.9
classes = train_dataset.classes
model_version = 'v1'
model_name = 'symbols-' + model_version
weight_initializer =nn.init.kaiming_uniform_
config = {
    "learning_rate": lr,
    "lr_gamma": lr_gamma,
    "epochs": epochs,
    "batch_size": batch_size,
    "model_input_shape": model_input_shape,
    "extra_pixels_before_crop": extra_pixels_before_crop,
    "random_rotation_train": random_rotation_train,
    "fill": fill,
    "loss_function": loss_function,
    "lr_scheduler": lr_scheduler,
    "weight_initializer": weight_initializer,
    "v_dataset__len__": len(eval_dataset),
    "v_dataloader__len__": len(validation_loader),
    "t_dataset__len__": len(train_dataset),
    "t_dataloader__len__": len(training_loader),
    "classes": classes,
    "model_name": model_name,
    "model_version": model_version,
    "data_loader_version": data_loader_version
}

run_name = f'{model_version}-{str(batch_size)}-{str(epochs)}-{str(lr)}'
run = wandb.init(project="symbols", entity="kaizen", name=run_name, config=config)
run.name = f'{run.name}-{run.id}'
run.save()
run_name = run.name

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5) # 28 * 28 * 4
        self.bn1 = nn.BatchNorm2d(4)
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
            weight_initializer(m.weight.data,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            weight_initializer(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = (x-0.5)*2
        x = self.conv1(x) # 28 * 28 * 4
        x = self.bn1(x)
        x = F.relu(x) # 28 * 28 * 4
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
    if epoch%4==0 and epoch > 0:
        scheduler.step()
    wandb.log({
        'epoch': epoch,
        'rate': scheduler.get_last_lr()[0],
        'epoch_vloss': avg_vloss,
        'epoch_vaccuracy': avg_vaccuracy * 100,
        'epoch_loss': avg_loss
    })
    artifact = wandb.Artifact('model', type='model')
    model_file = f'./saved_models/{run_name}_{epoch}'
    torch.save(net.state_dict(), model_file)
    artifact.add_file(model_file)
    run.log_artifact(artifact)

x = torch.randn(1, model_input_shape[0], model_input_shape[1], model_input_shape[2], requires_grad=True)
torch.onnx.export(net, x, f'./onnx_exports/{run_name}.onnx', input_names=['input'], output_names=['output'])

print('Finished Training')
