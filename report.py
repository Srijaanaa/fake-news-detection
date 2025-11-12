import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def generate_metrics(y_true, y_pred):
    """
    Calculates and returns key classification metrics.
    
    Args:
        y_true (list/array): True labels.
        y_pred (list/array): Predicted labels.
        
    Returns:
        dict: A dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def generate_confusion_matrix_plot(y_true, y_pred, path='static/confusion_matrix.png'):
    """
    Generates a confusion matrix plot and saves it as an image.
    
    Args:
        y_true (list/array): True labels.
        y_pred (list/array): Predicted labels.
        path (str): The path to save the image.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save the plot to the specified path
    plt.savefig(path)
    plt.close()

def generate_metrics_bar_chart(metrics, path='static/metrics_bar_chart.png'):
    """
    Generates a bar chart of the evaluation metrics and saves it as an image.
    
    Args:
        metrics (dict): A dictionary of metrics.
        path (str): The path to save the image.
    """
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    # Save the plot to the specified path
    plt.savefig(path)
    plt.close()
