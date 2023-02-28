"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = 0
NUM_WORKERS = os.cpu_count()

# Calculate the means and stds of the dataset
def get_means_stds(output_dir):

    train_data = datasets.ImageFolder(root = output_dir+'/train', transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

        means /= len(train_data)
        stds /= len(train_data)
    
    return means, stds

# Create data augmentation
def get_transforms(means, stds):

    data_augmentation = {
        'train': transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                     transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                     transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                                     transforms.ToTensor(),
                                     #transforms.Normalize(mean = means, std = stds),
        ]),
        'val': transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                   transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                   transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                                   transforms.ToTensor(),
                                   #transforms.Normalize(mean = means, std = stds),
        ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                    #transforms.Normalize(mean = means, std = stds),
        ]),
        'cust_test': transforms.Compose([transforms.ToTensor(),
                                         #transforms.Normalize(mean = means, std = stds),
        ]),
    }

    return data_augmentation

def create_dataloaders(
    output_dir: str, 
    data_augmentation: dict, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    image_datasets = {x: datasets.ImageFolder(os.path.join(output_dir, x), 
                                                          data_augmentation[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True, 
                                 drop_last=True) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names