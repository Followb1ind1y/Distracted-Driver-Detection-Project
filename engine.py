"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import time
import copy
import utils
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

def train_model(model: torch.nn.Module, 
                dataloaders: torch.utils.data.DataLoader,
                dataset_sizes: dict,
                epochs: int, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                save_dir: str,
                device: torch.device):
    """
    Train the Neural Networks.
    """
    start_time = time.time()

    best_weight = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # training mode
            else:
                model.eval()  # evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f} {phase} acc: {epoch_acc:.4f}', end=', ')

            if phase == 'train':
                scheduler.step()
                results["train_acc"].append(epoch_acc.cpu().detach())
                results["train_loss"].append(epoch_loss)
            else:
                results["val_acc"].append(epoch_acc.cpu().detach())
                results["val_loss"].append(epoch_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weight = copy.deepcopy(model.state_dict())

            # save checkpoints every epoch
            utils.save_model(model, save_dir, epoch)

        epoch_dur = round(time.time() - epoch_start,2)
        print(f'Epoch time:  {epoch_dur // 60:.0f}m {epoch_dur % 60:.0f}s')

    time_elapsed = time.time() - start_time
    print('-' * 20)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # save accuracy and loss
    np.savetxt(save_dir + "/" + f"model_{model.name}_train_acc.csv", results["train_acc"])
    np.savetxt(save_dir + "/" + f"model_{model.name}_train_loss.csv", results["train_loss"])
    np.savetxt(save_dir + "/" + f"model_{model.name}_val_acc.csv", results["val_acc"])
    np.savetxt(save_dir + "/" + f"model_{model.name}_val_loss.csv", results["val_loss"])

    # plot traininng curve
    train_accs, val_accs = np.array(results["train_acc"]), np.array(results["val_acc"])
    train_losses, val_losses = np.array(results["train_loss"]), np.array(results["val_loss"])

    plt.plot(np.arange(epochs, step=1), train_losses, label='Train loss')
    plt.plot(np.arange(epochs, step=1), train_accs, label='Train acc')
    plt.plot(np.arange(epochs, step=1), val_accs, label='Val acc')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    return model