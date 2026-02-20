"""
Error Analysis for mBERT Baseline
Analyze the 69 misclassified samples to determine if emotion trajectory could help
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from collections import Counter
import json

# Dataset class
class SarcasmDataset(Dataset):
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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float32),
            'text': text
        }

# Model class
class mBERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits.squeeze()

print("="*80)
print("ERROR ANALYSIS - mBERT Baseline")
print("="*80)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv('data/processed/test.csv')
print(f"   Test samples: {len(test_df)}")

# Load model
print("\n2. Loading best mBERT model...")
device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = mBERTClassifier().to(device)

checkpoint = torch.load('models/checkpoints/mbert_baseline_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("   ✓ Model loaded")

# Get predictions
print("\n3. Getting predictions on test set...")
test_dataset = SarcasmDataset(
    test_df['text_cleaned'].values,
    test_df['label'].values,
    tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_preds = []
all_probs = []
all_labels = []
all_texts = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Find errors
errors_mask = all_preds != all_labels
error_indices = np.where(errors_mask)[0]
num_errors = len(error_indices)

print(f"   Total errors: {num_errors}")

# Analyze errors
print("\n" + "="*80)
print("ERROR ANALYSIS RESULTS")
print("="*80)

# Get error samples
error_df = test_df.iloc[error_indices].copy()
error_df['predicted'] = all_preds[error_indices]
error_df['probability'] = all_probs[error_indices]
error_df['true_label'] = all_labels[error_indices]

# Error types
false_positives = error_df[error_df['predicted'] == 1]  # Predicted sarcastic, actually not
false_negatives = error_df[error_df['predicted'] == 0]  # Predicted not sarcastic, actually is

print(f"\n📊 ERROR BREAKDOWN:")
print(f"   False Positives (predicted sarcastic, actually not): {len(false_positives)}")
print(f"   False Negatives (predicted not sarcastic, actually is): {len(false_negatives)}")

# Sentence count analysis
print(f"\n📝 SENTENCE COUNT ANALYSIS:")
print(f"   Avg sentences in ALL test samples: {test_df['sentence_count'].mean():.2f}")
print(f"   Avg sentences in ERRORS: {error_df['sentence_count'].mean():.2f}")
print(f"   Avg sentences in CORRECT: {test_df.iloc[~errors_mask]['sentence_count'].mean():.2f}")

errors_multi_sentence = error_df[error_df['sentence_count'] > 1]
print(f"   Errors with multiple sentences: {len(errors_multi_sentence)} ({len(errors_multi_sentence)/len(error_df)*100:.1f}%)")

# Length analysis
print(f"\n📏 LENGTH ANALYSIS:")
print(f"   Avg length in ALL test samples: {test_df['cleaned_length'].mean():.1f} chars")
print(f"   Avg length in ERRORS: {error_df['cleaned_length'].mean():.1f} chars")
print(f"   Avg length in CORRECT: {test_df.iloc[~errors_mask]['cleaned_length'].mean():.1f} chars")

# Confidence analysis
print(f"\n🎯 CONFIDENCE ANALYSIS:")
uncertain_errors = error_df[(error_df['probability'] > 0.4) & (error_df['probability'] < 0.6)]
confident_errors = error_df[(error_df['probability'] < 0.2) | (error_df['probability'] > 0.8)]
print(f"   Uncertain errors (prob 0.4-0.6): {len(uncertain_errors)} ({len(uncertain_errors)/len(error_df)*100:.1f}%)")
print(f"   Confident errors (prob <0.2 or >0.8): {len(confident_errors)} ({len(confident_errors)/len(error_df)*100:.1f}%)")

# Show examples
print(f"\n" + "="*80)
print("EXAMPLE ERRORS (Could emotion trajectory help?)")
print("="*80)

# Show multi-sentence errors (most promising for trajectory)
print("\n🔍 MULTI-SENTENCE ERRORS (Trajectory could help here):")
multi_sent_errors = error_df[error_df['sentence_count'] > 1].head(5)
for idx, row in multi_sent_errors.iterrows():
    print(f"\n   Example {idx}:")
    print(f"   Text: {row['text'][:150]}...")
    print(f"   True: {'Sarcastic' if row['true_label'] == 1 else 'Not Sarcastic'}")
    print(f"   Predicted: {'Sarcastic' if row['predicted'] == 1 else 'Not Sarcastic'}")
    print(f"   Probability: {row['probability']:.3f}")
    print(f"   Sentences: {row['sentence_count']}")

# Show uncertain errors
print(f"\n🤔 UNCERTAIN ERRORS (Model was confused):")
uncertain = error_df[(error_df['probability'] > 0.45) & (error_df['probability'] < 0.55)].head(5)
for idx, row in uncertain.iterrows():
    print(f"\n   Example {idx}:")
    print(f"   Text: {row['text'][:150]}...")
    print(f"   True: {'Sarcastic' if row['true_label'] == 1 else 'Not Sarcastic'}")
    print(f"   Predicted: {'Sarcastic' if row['predicted'] == 1 else 'Not Sarcastic'}")
    print(f"   Probability: {row['probability']:.3f}")
    print(f"   Sentences: {row['sentence_count']}")

# Calculate potential
print(f"\n" + "="*80)
print("🎯 TRAJECTORY MODELING POTENTIAL")
print("="*80)

# How many errors have multiple sentences?
multi_sent_error_count = len(error_df[error_df['sentence_count'] > 1])
multi_sent_error_pct = multi_sent_error_count / len(error_df) * 100

print(f"\n📊 Can trajectory help?")
print(f"   Errors with 2+ sentences: {multi_sent_error_count}/{num_errors} ({multi_sent_error_pct:.1f}%)")
print(f"   Uncertain predictions: {len(uncertain_errors)}/{num_errors} ({len(uncertain_errors)/num_errors*100:.1f}%)")

# Realistic improvement estimate
if multi_sent_error_count > len(error_df) * 0.3:  # 30%+ are multi-sentence
    print(f"\n✅ VERDICT: Trajectory modeling has GOOD potential")
    print(f"   - Many errors involve multiple sentences")
    print(f"   - Emotion shifts could provide useful signal")
    
    # Calculate potential F1 improvement
    if_we_fix_half = (num_errors - multi_sent_error_count * 0.5)
    new_correct = len(test_df) - if_we_fix_half
    potential_accuracy = new_correct / len(test_df)
    
    print(f"\n📈 Potential improvement:")
    print(f"   If trajectory fixes 50% of multi-sentence errors:")
    print(f"   Accuracy: 95.21% → {potential_accuracy*100:.2f}%")
    print(f"   Expected F1: ~{potential_accuracy*100:.2f}%")
else:
    print(f"\n⚠️ VERDICT: Trajectory modeling has LIMITED potential")
    print(f"   - Most errors are single-sentence")
    print(f"   - Emotion trajectory unlikely to help significantly")
    print(f"   - Consider: Interpretability focus or different approach")

# Save error analysis
error_analysis = {
    'total_errors': num_errors,
    'false_positives': len(false_positives),
    'false_negatives': len(false_negatives),
    'multi_sentence_errors': multi_sent_error_count,
    'multi_sentence_error_percentage': multi_sent_error_pct,
    'uncertain_errors': len(uncertain_errors),
    'avg_sentence_count_errors': float(error_df['sentence_count'].mean()),
    'avg_sentence_count_correct': float(test_df.iloc[~errors_mask]['sentence_count'].mean()),
}

with open('outputs/results/error_analysis.json', 'w') as f:
    json.dump(error_analysis, f, indent=2)

error_df.to_csv('outputs/results/error_samples.csv', index=False)

print(f"\n✓ Error analysis saved to outputs/results/error_analysis.json")
print(f"✓ Error samples saved to outputs/results/error_samples.csv")
print("\n" + "="*80)
