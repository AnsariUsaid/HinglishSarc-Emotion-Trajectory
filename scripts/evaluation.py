"""
Evaluation Metrics Module for HinglishSarc
Handles evaluation metrics, reporting, and visualization
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


class SarcasmEvaluator:
    """Evaluator for sarcasm detection models"""
    
    def __init__(self, model_name="Model"):
        """
        Initialize evaluator
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.results = {}
    
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
        }
        
        # Add ROC-AUC if probabilities provided
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate per-class precision and recall
        tn, fp, fn, tp = cm.ravel()
        metrics['class_0_precision'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['class_0_recall'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['class_1_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['class_1_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        self.results = metrics
        return metrics
    
    def print_metrics(self, metrics=None):
        """Print metrics in a formatted way"""
        if metrics is None:
            metrics = self.results
        
        print(f"\n{'='*60}")
        print(f"  {self.model_name} - Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"Macro-F1:         {metrics['macro_f1']:.4f} ⭐")
        print(f"Macro-Precision:  {metrics['macro_precision']:.4f}")
        print(f"Macro-Recall:     {metrics['macro_recall']:.4f}")
        print(f"\nBinary Metrics (Class 1 - Sarcastic):")
        print(f"F1:               {metrics['f1']:.4f}")
        print(f"Precision:        {metrics['precision']:.4f}")
        print(f"Recall:           {metrics['recall']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"\nROC-AUC:          {metrics['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        print(f"{'='*60}\n")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Sarcastic', 'Sarcastic'],
                    yticklabels=['Non-Sarcastic', 'Sarcastic'])
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, save_path):
        """
        Save results to JSON file
        
        Args:
            save_path: Path to save JSON file
        """
        output = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.results
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to {save_path}")


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        save_path: Path to save comparison plot
    """
    import pandas as pd
    
    # Extract key metrics
    metrics_to_compare = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'f1']
    
    comparison_data = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        for metric in metrics_to_compare:
            if metric in metrics:
                row[metric.replace('_', ' ').title()] = metrics[metric]
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n" + "="*80)
    print("  MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df.set_index('Model')
    df_plot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Model Comparison - Key Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    
    evaluator = SarcasmEvaluator(model_name="Test Model")
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    evaluator.print_metrics()
    
    print("✓ Evaluation module test complete!")
