import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os

loader_version = 'v3'

class Erosin(ImageOnlyTransform):
    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 4, 2)))
        img = cv2.erode(img, kernel, iterations=1)
        return img

class Dilation(ImageOnlyTransform):
    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 2, 2)))
        img = cv2.dilate(img, kernel, iterations=1)
        return img

class ToBlackAndWhite(ImageOnlyTransform):
    def apply(self, img, **params):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(float)/255

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        paths = []
        class_to_idx = {}
        idx_to_class = {}
        for i, label in enumerate(os.listdir(root)):
            paths += [[root+label+'/'+p, label] for p in os.listdir(root+label)]
            class_to_idx[label] = i+0
            idx_to_class[i+0] = label
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.classes = list(class_to_idx.keys())
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_filepath, label = self.paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, self.class_to_idx[label]

def load_dataloader(
    batch_size,
    model_input_shape,
    extra_pixels_before_crop,
    random_rotation_train,
    fill,
    train_dataset_path = '../dataset/handwritten_math_symbols/train/',
    eval_dataset_path = '../dataset/handwritten_math_symbols/eval/'
):

    channels, height, width = model_input_shape
    train_dataset_transform = A.Compose([
        ToBlackAndWhite(p=1),
        A.Resize(width=100, height=100),
        A.Perspective(
            scale=(0.08, 0.12),
            p=0.5,
            fit_output=True,
            pad_val=(fill),
        ),
        Dilation(p=0.5),
        Erosin(p=0.5),
        A.Rotate(limit=random_rotation_train, p=0.5, value=(fill), border_mode=cv2.BORDER_CONSTANT),
        A.Resize(width=width, height=height),
        A.PadIfNeeded(
            min_width=width+extra_pixels_before_crop,
            min_height=height+extra_pixels_before_crop,
            value=(fill),
            border_mode=cv2.BORDER_CONSTANT,
            p=1
        ),
        A.RandomCrop(width, height),
        ToTensorV2()
    ])

    eval_dataset_transform = A.Compose([
        A.Resize(width=width, height=height),
        ToBlackAndWhite(p=1),
        ToTensorV2()
    ])

    train_dataset = ImageDataset(root=train_dataset_path, transform=train_dataset_transform)
    eval_dataset = ImageDataset(root=eval_dataset_path, transform=eval_dataset_transform)

    assert len(train_dataset.classes) == len(eval_dataset.classes)

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    datasets = (train_dataset, eval_dataset)
    data_loaders = (training_loader, validation_loader)
    return data_loaders, datasets, loader_version

