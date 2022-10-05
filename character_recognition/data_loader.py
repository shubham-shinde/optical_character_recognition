import torch
import torchvision
import torchvision.transforms as transforms

def load_dataloader(
    batch_size,
    model_input_shape,
    extra_pixels_before_crop,
    random_rotation_train,
    random_ratation_fill,
    train_dataset_path = '../dataset/handwritten_math_symbols/train/',
    eval_dataset_path = '../dataset/handwritten_math_symbols/eval/'
):

    channels, height, width = model_input_shape
    train_dataset_transform = transforms.Compose([
        transforms.Resize((height+extra_pixels_before_crop, width+extra_pixels_before_crop)),
        transforms.RandomCrop((height, width)),
        transforms.Grayscale(channels),
        transforms.ToTensor(),
#     transforms.Lambda(lambd=lambda x: 1 - x),
        transforms.RandomRotation(random_rotation_train, fill=random_ratation_fill),
#     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    eval_dataset_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.Grayscale(channels),
        transforms.ToTensor(),
#     transforms.Lambda(lambd=lambda x: 1 - x),
#     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_dataset_transform)
    eval_dataset = torchvision.datasets.ImageFolder(root=eval_dataset_path, transform=eval_dataset_transform)

    assert len(train_dataset.classes) == len(eval_dataset.classes)

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

    datasets = (train_dataset, eval_dataset)
    data_loaders = (training_loader, validation_loader)
    return data_loaders, datasets

