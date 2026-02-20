"""
Simplified Error Analysis - Just load results from confusion matrix
"""
import pandas as pd
import json
import numpy as np

print("="*80)
print("ERROR ANALYSIS - mBERT Baseline")
print("="*80)

# Load test data and results
test_df = pd.read_csv('data/processed/test.csv')
with open('outputs/results/mbert_baseline_results.json', 'r') as f:
    results = json.load(f)

# Extract confusion matrix
cm = results['metrics']['confusion_matrix']
TN, FP = cm[0]
FN, TP = cm[1]

print(f"\n📊 CONFUSION MATRIX:")
print(f"   True Negatives (correct non-sarcastic): {TN}")
print(f"   False Positives (wrong - said sarcastic): {FP}")
print(f"   False Negatives (wrong - said not sarcastic): {FN}")
print(f"   True Positives (correct sarcastic): {TP}")
print(f"\n   Total errors: {FP + FN}")

# Calculate error distribution in dataset
total_samples = len(test_df)
non_sarc_samples = (test_df['label'] == 0).sum()
sarc_samples = (test_df['label'] == 1).sum()

print(f"\n📈 TEST SET DISTRIBUTION:")
print(f"   Total samples: {total_samples}")
print(f"   Non-sarcastic: {non_sarc_samples} ({non_sarc_samples/total_samples*100:.1f}%)")
print(f"   Sarcastic: {sarc_samples} ({sarc_samples/total_samples*100:.1f}%)")

# Analyze sentence counts
print(f"\n📝 SENTENCE COUNT ANALYSIS:")
avg_sentences = test_df['sentence_count'].mean()
multi_sentence_pct = (test_df['sentence_count'] > 1).sum() / len(test_df) * 100

print(f"   Average sentences per sample: {avg_sentences:.2f}")
print(f"   Multi-sentence samples: {(test_df['sentence_count'] > 1).sum()} ({multi_sentence_pct:.1f}%)")
print(f"   Single-sentence samples: {(test_df['sentence_count'] == 1).sum()} ({100-multi_sentence_pct:.1f}%)")

# Sentence count distribution
sent_counts = test_df['sentence_count'].value_counts().sort_index()
print(f"\n   Distribution:")
for count, freq in sent_counts.items():
    print(f"     {count} sentence(s): {freq} samples ({freq/len(test_df)*100:.1f}%)")

# Estimate trajectory potential
print(f"\n{'='*80}")
print("🎯 TRAJECTORY MODELING POTENTIAL ASSESSMENT")
print("="*80)

# Assumption: If X% of test set is multi-sentence, roughly X% of errors might be too
expected_multi_sent_errors = (FP + FN) * (multi_sentence_pct / 100)

print(f"\n📊 Expected error distribution:")
print(f"   Total errors: {FP + FN}")
print(f"   Expected multi-sentence errors: ~{expected_multi_sent_errors:.0f} ({expected_multi_sent_errors/(FP+FN)*100:.1f}%)")
print(f"   Expected single-sentence errors: ~{(FP+FN)-expected_multi_sent_errors:.0f}")

# Improvement scenarios
print(f"\n📈 IMPROVEMENT SCENARIOS:")
print(f"\n   Scenario 1: Fix 50% of multi-sentence errors")
errors_fixed = expected_multi_sent_errors * 0.5
new_errors = (FP + FN) - errors_fixed
new_accuracy = (total_samples - new_errors) / total_samples
print(f"     Errors: {FP+FN} → {new_errors:.0f}")
print(f"     Accuracy: 95.21% → {new_accuracy*100:.2f}%")
print(f"     Expected F1: ~{new_accuracy*100:.2f}%")
print(f"     Gain: +{(new_accuracy-0.9521)*100:.2f}%")

print(f"\n   Scenario 2: Fix 30% of multi-sentence errors (conservative)")
errors_fixed = expected_multi_sent_errors * 0.3
new_errors = (FP + FN) - errors_fixed
new_accuracy = (total_samples - new_errors) / total_samples
print(f"     Errors: {FP+FN} → {new_errors:.0f}")
print(f"     Accuracy: 95.21% → {new_accuracy*100:.2f}%")
print(f"     Expected F1: ~{new_accuracy*100:.2f}%")
print(f"     Gain: +{(new_accuracy-0.9521)*100:.2f}%")

print(f"\n   Scenario 3: Fix 70% of multi-sentence errors (optimistic)")
errors_fixed = expected_multi_sent_errors * 0.7
new_errors = (FP + FN) - errors_fixed
new_accuracy = (total_samples - new_errors) / total_samples
print(f"     Errors: {FP+FN} → {new_errors:.0f}")
print(f"     Accuracy: 95.21% → {new_accuracy*100:.2f}%")
print(f"     Expected F1: ~{new_accuracy*100:.2f}%")
print(f"     Gain: +{(new_accuracy-0.9521)*100:.2f}%")

# Verdict
print(f"\n{'='*80}")
print("🔍 VERDICT")
print("="*80)

if multi_sentence_pct > 40:
    print(f"\n✅ GOOD POTENTIAL for trajectory modeling")
    print(f"   - {multi_sentence_pct:.1f}% of samples have multiple sentences")
    print(f"   - Emotion shifts could provide useful signal")
    print(f"   - Expected improvement: +0.5% to +1.5% F1")
    verdict = "PROCEED"
elif multi_sentence_pct > 25:
    print(f"\n⚠️ MODERATE POTENTIAL for trajectory modeling")
    print(f"   - {multi_sentence_pct:.1f}% of samples have multiple sentences")
    print(f"   - Some room for improvement but limited")
    print(f"   - Expected improvement: +0.3% to +0.8% F1")
    print(f"   - Focus on interpretability as main contribution")
    verdict = "PROCEED_WITH_CAUTION"
else:
    print(f"\n❌ LIMITED POTENTIAL for trajectory modeling")
    print(f"   - Only {multi_sentence_pct:.1f}% of samples have multiple sentences")
    print(f"   - Trajectory unlikely to help significantly")
    print(f"   - Expected improvement: +0.1% to +0.4% F1")
    print(f"   - Consider pivoting approach")
    verdict = "RECONSIDER"

# Save analysis
analysis_results = {
    'total_errors': FP + FN,
    'false_positives': FP,
    'false_negatives': FN,
    'test_samples': total_samples,
    'multi_sentence_percentage': multi_sentence_pct,
    'expected_multi_sentence_errors': expected_multi_sent_errors,
    'verdict': verdict,
    'expected_improvement_conservative': f"+{((total_samples - ((FP+FN) - expected_multi_sent_errors*0.3))/total_samples - 0.9521)*100:.2f}%",
    'expected_improvement_optimistic': f"+{((total_samples - ((FP+FN) - expected_multi_sent_errors*0.7))/total_samples - 0.9521)*100:.2f}%"
}

with open('outputs/results/error_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"\n✓ Analysis saved to outputs/results/error_analysis.json")
print("="*80)
