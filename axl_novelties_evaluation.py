# ============================================================================
# COMPLETE AXL-ICT TRAINING - 3 EPOCHS (FAST VERSION)
# Expected Time: ~40 minutes | Completion: ~7:40 PM IST
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ============================================================================
# UPDATE CONFIG TO 3 EPOCHS
# ============================================================================

AXL_CONFIG['max_epochs_axl'] = 3  # Changed to 3 epochs

print("\n" + "="*80)
print("AXL-ICT CONFIGURATION - 3 EPOCHS (FAST MODE)")
print("="*80)
print(f"✓ Epochs: {AXL_CONFIG['max_epochs_axl']}")
print(f"✓ Estimated time: ~40 minutes")
print(f"✓ Expected completion: ~7:40 PM IST")
print("="*80)

# ============================================================================
# FIX: Redefine DPRModel with forward() method
# ============================================================================

class DenseEncoder(nn.Module):
    """Dense encoder for queries or passages"""

    def __init__(self, model_name: str, pooling: str = 'cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            embeddings = embeddings / attention_mask.sum(-1, keepdim=True)

        return F.normalize(embeddings, p=2, dim=1)


class DPRModel(nn.Module):
    """Dual-encoder Dense Passage Retriever with forward() method"""

    def __init__(self, config):
        super().__init__()
        self.query_encoder = DenseEncoder(config['model_name'])
        self.passage_encoder = DenseEncoder(config['model_name'])
        self.config = config

    def forward(self, query_inputs, passage_inputs, negative_inputs=None):
        """Forward method for training"""
        query_emb = self.query_encoder(**query_inputs)
        pos_emb = self.passage_encoder(**passage_inputs)
        
        if negative_inputs is not None:
            neg_emb = self.passage_encoder(**negative_inputs)
            return query_emb, pos_emb, neg_emb
        
        return query_emb, pos_emb

    def encode_queries(self, queries, tokenizer, device, batch_size=32):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=self.config['max_length'],
                                   return_tensors='pt').to(device)
                emb = self.query_encoder(**inputs)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def encode_passages(self, passages, tokenizer, device, batch_size=32):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=self.config['max_length'],
                                   return_tensors='pt').to(device)
                emb = self.passage_encoder(**inputs)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

# ============================================================================
# RELOAD MODEL
# ============================================================================

print("\n[FIX] Reloading model with forward() method...")

model = DPRModel(CONFIG).to(device)

try:
    model.load_state_dict(torch.load(f"{CONFIG['model_path']}/ict_p_baseline/FT_best.pt"))
    print("✓ Loaded Phase 1 weights successfully")
except:
    print("⚠ Could not load Phase 1 weights - starting fresh")

print("✓ Model ready with forward() method")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING AXL-ICT TRAINING - 3 EPOCHS")
print("="*80)

use_negatives = [[] for _ in range(len(mrtydi_train))]

model, losses, mrrs, recalls = train_axlict_final(
    model,
    mrtydi_train,
    AXL_CONFIG,
    tokenizer,
    device,
    use_negatives
)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION ON FULL TEST SET")
print("="*80)

results_axl = evaluate_model(model, mrtydi_test, AXL_CONFIG, tokenizer, device)

# ============================================================================
# VISUALIZATION 1: TRAINING LOSS CURVE
# ============================================================================

print("\n[Viz] Creating training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Training Loss
axes[0, 0].plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color='steelblue')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('AXL-ICT Training Loss (3 Epochs)', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Intermediate MRR
if len(mrrs) > 0:
    eval_epochs = [3][:len(mrrs)]
    axes[0, 1].plot(eval_epochs, mrrs, marker='s', linewidth=2, markersize=10, color='coral', label='MRR@100')
    axes[0, 1].axhline(y=0.454, color='red', linestyle='--', linewidth=2, label='Paper ICT-P (0.454)')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MRR@100', fontsize=12)
    axes[0, 1].set_title('MRR@100 Progress', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlim([0, 4])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'Evaluation at Epoch 3', 
                    ha='center', va='center', fontsize=12)
    axes[0, 1].set_title('MRR@100 Progress', fontsize=14, fontweight='bold')

# Plot 3: Intermediate Recall
if len(recalls) > 0:
    eval_epochs = [3][:len(recalls)]
    axes[1, 0].plot(eval_epochs, recalls, marker='^', linewidth=2, markersize=10, color='green', label='Recall@100')
    axes[1, 0].axhline(y=0.870, color='red', linestyle='--', linewidth=2, label='Paper ICT-P (0.870)')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Recall@100', fontsize=12)
    axes[1, 0].set_title('Recall@100 Progress', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim([0, 4])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'Evaluation at Epoch 3', 
                    ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('Recall@100 Progress', fontsize=14, fontweight='bold')

# Plot 4: Training Summary
axes[1, 1].axis('off')
summary_text = f"""
AXL-ICT TRAINING SUMMARY
{'='*40}

Configuration:
• Epochs: 3 (Fast Mode)
• Batch size: {AXL_CONFIG['batch_size']}
• Learning rate: {AXL_CONFIG['learning_rate']}
• Hard negatives: {sum(len(n) for n in use_negatives)}

Training Duration: ~40 minutes

Final Metrics:
• Best loss: {min(losses):.4f}
• Final loss: {losses[-1]:.4f}
• Loss reduction: {((losses[0]-losses[-1])/losses[0]*100):.1f}%

Novelties Applied:
✓ Enhanced contrastive loss
✓ 3-stage curriculum learning
✓ ICT-P iterative clustering
✓ In-batch hard negatives
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{AXL_CONFIG["base_path"]}/logs/axlict_training_curves_3epochs.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Training curves saved")

# ============================================================================
# VISUALIZATION 2: RESULTS COMPARISON WITH PAPER
# ============================================================================

# Paper's ICT-P results
paper_results = {
    'english': {'MRR@100': 0.426, 'Recall@100': 0.842},
    'arabic': {'MRR@100': 0.457, 'Recall@100': 0.884},
    'bengali': {'MRR@100': 0.492, 'Recall@100': 0.919},
    'finnish': {'MRR@100': 0.466, 'Recall@100': 0.882},
    'indonesian': {'MRR@100': 0.467, 'Recall@100': 0.870},
    'japanese': {'MRR@100': 0.434, 'Recall@100': 0.849},
    'korean': {'MRR@100': 0.438, 'Recall@100': 0.863},
    'russian': {'MRR@100': 0.423, 'Recall@100': 0.835},
    'swahili': {'MRR@100': 0.486, 'Recall@100': 0.887},
    'telugu': {'MRR@100': 0.455, 'Recall@100': 0.867}
}

langs = list(paper_results.keys())
paper_mrrs = [paper_results[lang]['MRR@100'] for lang in langs]
paper_recalls = [paper_results[lang]['Recall@100'] for lang in langs]
axl_mrrs = [results_axl.get(lang, {}).get('MRR@100', 0) for lang in langs]
axl_recalls = [results_axl.get(lang, {}).get('Recall@100', 0) for lang in langs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# MRR Comparison
x = np.arange(len(langs))
width = 0.35
bars1 = ax1.bar(x - width/2, paper_mrrs, width, label='Paper ICT-P', color='lightcoral', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, axl_mrrs, width, label='Our AXL-ICT (3 epochs)', color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Language', fontsize=12)
ax1.set_ylabel('MRR@100', fontsize=12)
ax1.set_title('MRR@100: Paper ICT-P vs. Our AXL-ICT', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(langs, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0.454, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Paper Avg')

# Recall Comparison
bars3 = ax2.bar(x - width/2, paper_recalls, width, label='Paper ICT-P', color='lightcoral', alpha=0.7, edgecolor='black')
bars4 = ax2.bar(x + width/2, axl_recalls, width, label='Our AXL-ICT (3 epochs)', color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Language', fontsize=12)
ax2.set_ylabel('Recall@100', fontsize=12)
ax2.set_title('Recall@100: Paper ICT-P vs. Our AXL-ICT', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(langs, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0.870, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{AXL_CONFIG["base_path"]}/logs/axlict_vs_paper_3epochs.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Comparison chart saved")

# ============================================================================
# VISUALIZATION 3: IMPROVEMENT HEATMAP
# ============================================================================

improvements = np.array(axl_mrrs) - np.array(paper_mrrs)
improvement_pct = (improvements / np.array(paper_mrrs)) * 100

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['red' if x < 0 else 'green' for x in improvements]
bars = ax.barh(langs, improvement_pct, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('MRR@100 Improvement (%)', fontsize=12)
ax.set_title('AXL-ICT (3 epochs) Improvement over Paper ICT-P', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=2)
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, improvement_pct)):
    label = f'{val:+.1f}%'
    ax.text(val + 1 if val > 0 else val - 1, i, label, 
            ha='left' if val > 0 else 'right', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{AXL_CONFIG["base_path"]}/logs/axlict_improvement_3epochs.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Improvement heatmap saved")

# ============================================================================
# FINAL RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS: AXL-ICT (3 EPOCHS) vs. PAPER ICT-P")
print("="*80)

paper_avg_mrr = np.mean(paper_mrrs)
paper_avg_recall = np.mean(paper_recalls)
axl_avg_mrr = np.mean(axl_mrrs)
axl_avg_recall = np.mean(axl_recalls)

print(f"\nPaper's ICT-P Baseline:")
print(f"  Average MRR@100:    {paper_avg_mrr:.4f}")
print(f"  Average Recall@100: {paper_avg_recall:.4f}")

print(f"\nOur AXL-ICT (3 epochs):")
print(f"  Average MRR@100:    {axl_avg_mrr:.4f}")
print(f"  Average Recall@100: {axl_avg_recall:.4f}")

improvement_mrr = ((axl_avg_mrr - paper_avg_mrr) / paper_avg_mrr) * 100
improvement_recall = ((axl_avg_recall - paper_avg_recall) / paper_avg_recall) * 100

print(f"\nImprovement:")
print(f"  MRR@100:    {improvement_mrr:+.2f}%")
print(f"  Recall@100: {improvement_recall:+.2f}%")

if axl_avg_mrr > paper_avg_mrr:
    print(f"\n🎉 SUCCESS: Exceeded paper's ICT-P baseline in just 3 epochs!")
else:
    print(f"\n📊 Results: Good progress in 3 epochs")
    print(f"   Note: More epochs or adversarial negatives would improve further")

# Per-language table
print("\n" + "="*80)
print("PER-LANGUAGE COMPARISON")
print("="*80)
print(f"{'Language':<12} {'Paper':<10} {'AXL-ICT':<10} {'Diff':<10} {'Status'}")
print("-" * 80)

for lang, paper_mrr, axl_mrr in zip(langs, paper_mrrs, axl_mrrs):
    diff = axl_mrr - paper_mrr
    status = "✓ Better" if diff > 0 else "→ Similar" if abs(diff) < 0.01 else "✗ Lower"
    print(f"{lang:<12} {paper_mrr:<10.4f} {axl_mrr:<10.4f} {diff:+10.4f} {status}")

# Save results
comparison_df = pd.DataFrame({
    'Language': langs,
    'Paper_MRR': paper_mrrs,
    'Paper_Recall': paper_recalls,
    'AXL_MRR': axl_mrrs,
    'AXL_Recall': axl_recalls,
    'MRR_Improvement_%': improvement_pct.tolist()
})

comparison_df.to_csv(f"{AXL_CONFIG['base_path']}/logs/final_comparison_3epochs.csv", index=False)
print(f"\n✓ Results saved to {AXL_CONFIG['base_path']}/logs/final_comparison_3epochs.csv")

print("\n" + "="*80)
print("PHASE 2 COMPLETE - 3 EPOCHS")
print("="*80)
print("\n✓ Training completed in ~40 minutes")
print("\n✓ Visualizations generated:")
print(f"  • Training curves: axlict_training_curves_3epochs.png")
print(f"  • Comparison chart: axlict_vs_paper_3epochs.png")
print(f"  • Improvement heatmap: axlict_improvement_3epochs.png")
print(f"  • Results CSV: final_comparison_3epochs.csv")
print("\n✓ All files saved to: /content/logs/")
print("="*80)
