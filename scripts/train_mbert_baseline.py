"""
mBERT Baseline for Sarcasm Detection
Fine-tune multilingual BERT on Hinglish sarcasm data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import sys
sys.path.append('.')
from scripts.evaluation import SarcasmEvaluator
import os
import json
from datetime import datetime


class SarcasmDataset(Dataset):
    """Dataset for sarcasm detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class mBERTBaseline:
    """mBERT baseline model for sarcasm detection"""
    
    def __init__(self, model_name='bert-base-multilingual-cased', 
                 max_length=128, batch_size=16, learning_rate=2e-5, 
                 num_epochs=3, device=None):
        """
        Initialize mBERT baseline
        
        Args:
            model_name: Pretrained model name
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        self.model.to(self.device)
        
        self.train_history = []
        self.val_history = []
    
    def create_data_loaders(self, train_texts, train_labels, 
                           val_texts, val_labels):
        """Create data loaders for training and validation"""
        train_dataset = SarcasmDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = SarcasmDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader):
        """Train the model"""
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            progress_bar = tqdm(train_loader, desc='Training')
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                preds = torch.argmax(outputs.logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_f1 = SarcasmEvaluator().compute_metrics(
                train_labels, train_preds
            )['macro_f1']
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics['macro_f1']
            
            print(f"Val Loss: {val_metrics.get('loss', 0):.4f}, Val F1: {val_f1:.4f}")
            
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1
            })
            self.val_history.append({
                'epoch': epoch + 1,
                'val_f1': val_f1,
                **val_metrics
            })
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model('models/checkpoints/mbert_baseline_best.pt')
                print(f"✓ Best model saved (Val F1: {val_f1:.4f})")
        
        return self.train_history, self.val_history
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        evaluator = SarcasmEvaluator(model_name="mBERT Baseline")
        metrics = evaluator.compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"✓ Model loaded from {path}")


# Example usage / training script
if __name__ == "__main__":
    print("mBERT Baseline Training Script")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    
    # Initialize model
    print("\n2. Initializing mBERT baseline...")
    model = mBERTBaseline(
        model_name='bert-base-multilingual-cased',
        max_length=128,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3
    )
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader = model.create_data_loaders(
        train_df['text_cleaned'].values,
        train_df['label'].values,
        val_df['text_cleaned'].values,
        val_df['label'].values
    )
    
    # Train
    print("\n4. Training model...")
    train_history, val_history = model.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_dataset = SarcasmDataset(
        test_df['text_cleaned'].values,
        test_df['label'].values,
        model.tokenizer,
        model.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    test_metrics = model.evaluate(test_loader)
    evaluator = SarcasmEvaluator(model_name="mBERT Baseline")
    evaluator.results = test_metrics
    evaluator.print_metrics()
    
    # Save results
    evaluator.save_results('outputs/results/mbert_baseline_results.json')
    
    print("\n✅ Training complete!")
