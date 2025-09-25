import sys
import os
import torch
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
#from cm_metrics import compute_confusion_stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from models.baseline_models import *
#from models.dist_layer_models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def test(model, loader, criterion, num_classes):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            y_true.extend(target.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            #correct += pred.eq(target).sum().item()
            #total += target.size(0)

    cm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    TP = []
    FP = []
    FN = []
    TN = []

    for i in range(num_classes):
        tn, fp, fn, tp = cm[i].ravel()
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        TN.append(tn)

    TP_sum = np.sum(TP)
    FP_sum = np.sum(FP)
    FN_sum = np.sum(FN)
    TN_sum = np.sum(TN)


    # Metrics (macro-averaged for multiclass)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'test_loss': round(total_loss / len(loader), 5),
        'test_acc': round(acc * 100, 5),
        'accuracy': round(acc, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'f1_score': round(f1, 5),
        'TP': TP_sum,
        'FP': FP_sum,
        'FN': FN_sum,
        'TN': TN_sum
        }

def safe_confusion_matrix_stats(y_true, y_pred, num_classes):
    """Safely compute confusion matrix statistics"""
    try:
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        TP = []
        FP = []
        FN = []
        TN = []
        
        for i in range(num_classes):
            tn, fp, fn, tp = cm[i].ravel()
            TP.append(tp)
            FP.append(fp)
            FN.append(fn)
            TN.append(tn)
        
        TP_sum = np.sum(TP)
        FP_sum = np.sum(FP)
        FN_sum = np.sum(FN)
        TN_sum = np.sum(TN)
        
        return TP_sum, FP_sum, FN_sum, TN_sum
    
    except Exception:
        # Fallback to simple calculations if confusion matrix fails
        total_samples = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct, total_samples - correct, 0, 0

def safe_sklearn_metrics(y_true, y_pred):
    """Safely compute sklearn metrics with fallback values"""
    try:
        acc = accuracy_score(y_true, y_pred)
    except Exception:
        acc = 0.0
    
    try:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        precision = 0.0
    
    try:
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        recall = 0.0
    
    try:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        f1 = 0.0
    
    return acc, precision, recall, f1

def test_minimal(model, loader, criterion, num_classes):
    """Minimal test function that avoids embedding calculations and handles sklearn warnings"""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Ensure we only get the output, not embeddings
            if hasattr(model, 'forward'):
                # Check if the model expects return_embedding parameter
                forward_params = model.forward.__code__.co_varnames
                if 'return_embedding' in forward_params:
                    output = model(data, return_embedding=False)
                else:
                    output = model(data)
            else:
                output = model(data)
                
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            y_true.extend(target.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    # Safely compute confusion matrix statistics
    TP_sum, FP_sum, FN_sum, TN_sum = safe_confusion_matrix_stats(y_true, y_pred, num_classes)
    
    # Safely compute sklearn metrics
    acc, precision, recall, f1 = safe_sklearn_metrics(y_true, y_pred)

    return {
        'test_loss': round(total_loss / len(loader), 5),
        'test_acc': round(acc * 100, 5),
        'accuracy': round(acc, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'f1_score': round(f1, 5),
        'TP': TP_sum,
        'FP': FP_sum,
        'FN': FN_sum,
        'TN': TN_sum
    }
