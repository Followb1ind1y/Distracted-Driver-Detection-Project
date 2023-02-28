"""
Contains Ensemble Methods like Majority vote, Average likelihood, Weighted Majority.
"""

import torch

from torch import nn 
from scipy.stats import mode

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.num_models = len(models)
        self.models = models
        self.weights = torch.ones(self.num_models,10)

# Majority vote
def ensemble_pred_majority(model, dataloaders, device):
    """
    majority vote
    """
    model.to(device)
    for net in model.models:
        net.eval()

    labels_list = []
    pred_labels = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #track history only in training phase
        with torch.set_grad_enabled(False):
            out = model.models[0](inputs)
            _, pred = torch.max(out, 1)
            pred = pred[None, :]    # add one dimension

            for i in range(1, model.num_models):
                out = model.models[i](inputs)
                _, new_pred = torch.max(out, 1)
                new_pred = new_pred[None, :]
                pred = torch.cat((pred, new_pred), dim=0)
            
            pred = pred.cpu().detach()
            pred = torch.tensor(mode(pred)[0])

            labels_list.append(labels.cpu())
            pred_labels.append(pred.cpu())

    labels_list = torch.cat(labels_list, dim = 0)
    pred_labels = torch.cat(pred_labels, dim = 0)
    

    return labels_list, torch.flatten(pred_labels)

# Average likelihood
def ensemble_pred_avg(model, dataloaders, device):
    """
    average likelihood
    """
    model.to(device)
    for net in model.models:
        net.eval()

    labels_list = []
    pred_labels = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #track history only in training phase
        with torch.set_grad_enabled(False):
            out_sum = torch.zeros((len(inputs), 10)).to(device)
            for i in range(model.num_models):
                out = model.models[i](inputs)
                out_sum += out
                _, pred = torch.max(out_sum, 1)

            labels_list.append(labels.cpu())
            pred_labels.append(pred.cpu())

    labels_list = torch.cat(labels_list, dim = 0)
    pred_labels = torch.cat(pred_labels, dim = 0)
    

    return labels_list, torch.flatten(pred_labels)

def ensemble_weighted_pred(model, dataloaders, dataset_sizes, device):
    """
    weighted majority vote
    """
    model.to(device)
    for net in model.models:
        net.eval()

    multi_class_pred = torch.zeros((model.num_models, dataset_sizes['test'], 10)).to(device)
    multi_class_idx = torch.zeros((model.num_models, dataset_sizes['test'])).to(device)

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            for i in range(model.num_models):
                out = model.models[i](inputs)
                _, pred = torch.max(out, 1)
                multi_class_idx[i] = pred
                for single_pred, idx in zip(multi_class_pred[i], pred):
                    single_pred[idx] = model.weights[i][idx]

        weight_sum = torch.sum(multi_class_pred, dim=0)
        _, final_pred = torch.max(weight_sum, 1)

    return final_pred, multi_class_idx