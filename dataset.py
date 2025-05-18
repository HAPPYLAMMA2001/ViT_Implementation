import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        
        if train:
            # Load training batches
            for i in range(1, 6):
                batch_file = os.path.join(root_dir, f'data_batch_{i}')
                batch_data = unpickle(batch_file)
                self.data.append(batch_data[b'data'])
                self.targets.extend(batch_data[b'labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        else:
            # Load test batch
            batch_file = os.path.join(root_dir, 'test_batch')
            batch_data = unpickle(batch_file)
            self.data = batch_data[b'data'].reshape(-1, 3, 32, 32)
            self.targets = batch_data[b'labels']
            
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
        
        # Default transform if none provided
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),  # ViT typically uses 224x224
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2023, 0.1994, 0.2010])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224),  # ViT typically uses 224x224
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2023, 0.1994, 0.2010])
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    return train_transform, val_transform


