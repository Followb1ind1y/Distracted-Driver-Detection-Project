"""
Utility functions to make predictions.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Find the test accuracy of a target model
def evaluate_model(model: torch.nn.Module, 
                   dataloaders: torch.utils.data.DataLoader,
                   dataset_sizes: dict,
                   device: torch.device):
    """
    Evaluate model performance on testset
    """
    model.eval()
    model.to(device)

    test_acc = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #track history only in training phase
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        test_acc += torch.sum(preds == labels.data).item()

    test_acc = test_acc / dataset_sizes['test']

    return test_acc

# Predict probs and pred labels
def get_predictions(model: torch.nn.Module, 
                    dataloaders: torch.utils.data.DataLoader,
                    device: torch.device):
    """
    Get predictions on testset
    """
    model.eval()
    model.to(device)

    labels_list = []
    probs_list = []
    pred_labels = []

    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #track history only in training phase
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            y_prob = F.softmax(outputs, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            labels_list.append(labels.cpu())
            probs_list.append(y_prob.cpu())
            pred_labels.append(top_pred.cpu())

    labels_list = torch.cat(labels_list, dim = 0)
    probs_list = torch.cat(probs_list, dim = 0)
    pred_labels = torch.cat(pred_labels, dim = 0)
    

    return labels_list, probs_list, pred_labels

# Plot the confusion matrix
def plot_confusion_matrix(labels, pred_labels, class_names):
    
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = class_names);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.delaxes(fig.axes[1]) #delete colorbar
    plt.xticks(rotation = 90)
    plt.xlabel('Predicted Label', fontsize = 10)
    plt.ylabel('True Label', fontsize = 10)