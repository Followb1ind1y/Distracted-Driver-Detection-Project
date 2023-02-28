"""
Trains a PyTorch image classification model.
"""

import os
import torch
import splitfolders
import torch.optim as optim
import data_setup, engine, model_builder, utils, predictions

from torch.optim import lr_scheduler

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 16

data_dir = 'imgs/train'
splitfolders.ratio(data_dir, output="train_split", ratio=(0.8, 0.1, 0.1))
output_dir = 'train_split'

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Calculate the means and stds of the dataset
means, stds = data_setup.get_means_stds(output_dir=output_dir)

# Create data augmentation
data_augmentation = data_setup.get_transforms(means=means, stds=stds)

# Create DataLoaders with help from data_setup.py
dataloaders, dataset_sizes, class_names = data_setup.create_dataloaders(
    output_dir=output_dir, 
    data_augmentation=data_augmentation, 
    batch_size=BATCH_SIZE
)

# Create model from model_builder.py
MobileNet_V3_model = model_builder.MobileNet_V3().to(device)

# Set loss, optimizer and lr_scheduler
criterion_MobileNet_V3 = torch.nn.CrossEntropyLoss()
optimizer_MobileNet_V3 = optim.SGD(MobileNet_V3_model.parameters(), lr=0.005, momentum=0.9)
exp_lr_scheduler_MobileNet_V3 = lr_scheduler.StepLR(optimizer_MobileNet_V3, step_size=7, gamma=0.1)

# Start training with help from engine.py
MobileNet_V3_output_model = engine.train_model(model=MobileNet_V3_model, 
                                               dataloaders=dataloaders,
                                               dataset_sizes=dataset_sizes,
                                               epochs=NUM_EPOCHS, 
                                               criterion=criterion_MobileNet_V3, 
                                               optimizer=optimizer_MobileNet_V3,
                                               scheduler=exp_lr_scheduler_MobileNet_V3,
                                               save_dir="Output",
                                               device=device)

# Predict the result from predictions.py
MobileNet_V3_test_acc = predictions.evaluate_model(MobileNet_V3_model, dataloaders, dataset_sizes, device)
print('Test acc: ', MobileNet_V3_test_acc)