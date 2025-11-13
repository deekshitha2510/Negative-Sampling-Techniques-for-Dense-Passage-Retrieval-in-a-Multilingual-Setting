# ============================================================================
# DENSE PASSAGE RETRIEVER (DPR) IMPLEMENTATION — FINAL FIXED VERSION (2025)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ----- Model & Paths -----
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "data_path": "/content/data",
    "model_path": "/content/models",
    "base_path": "/content",

    # ----- Dataset -----
    "mr_tydi_languages": [
        "english", "arabic", "bengali", "finnish", "indonesian",
        "japanese", "korean", "russian", "swahili", "telugu"
    ],
    "mr_tydi_balanced_size": 2000,
    "batch_size": 8,
    "num_workers": 2,
    "max_length": 256,

    # ----- Training Hyperparameters -----
    "learning_rate": 3e-5,
    "warmup_steps": 100,
    "max_epochs_pft": 5,      # Pre-finetuning epochs (MS MARCO)
    "max_epochs_ft": 5,       # Finetuning epochs (Mr. TyDi)
    "gradient_accumulation_steps": 2,
    "save_steps": 200,
    "num_clusters": 50,
    "clustering_frequency": 1
}

os.makedirs(f"{CONFIG['model_path']}/checkpoints", exist_ok=True)
os.makedirs(f"{CONFIG['model_path']}/ict_p_baseline", exist_ok=True)
os.makedirs(f"{CONFIG['base_path']}/logs", exist_ok=True)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class DenseEncoder(nn.Module):
    """Dense encoder for queries or passages"""

    def __init__(self, model_name: str, pooling: str = 'cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, **kwargs):
        # Accept extra arguments like 'token_type_ids' from tokenizer
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            embeddings = embeddings / attention_mask.sum(-1, keepdim=True)

        return F.normalize(embeddings, p=2, dim=1)


class DPRModel(nn.Module):
    """Dual-encoder Dense Passage Retriever"""

    def __init__(self, config):
        super().__init__()
        self.query_encoder = DenseEncoder(config['model_name'])
        self.passage_encoder = DenseEncoder(config['model_name'])
        self.config = config

    def forward(self, query_inputs, passage_inputs, negative_inputs=None):
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
# ICT-P: ITERATIVE CLUSTERED TRAINING (PASSAGE-BASED)
# ============================================================================

class ICTPTrainer:
    """Iterative Clustered Training with Passage clustering"""

    def __init__(self, model, config, tokenizer, device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.clusters = None
        self.passage_embeddings = None

    def cluster_passages(self, passages: List[str]):
        print("\n[ICT-P] Clustering passages...")
        embeddings = self.model.encode_passages(passages, self.tokenizer, self.device, batch_size=64)
        self.passage_embeddings = embeddings.numpy()

        n_clusters = min(self.config['num_clusters'], max(2, len(passages) // 10))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, max_iter=100)
        cluster_labels = kmeans.fit_predict(self.passage_embeddings)

        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)

        self.clusters = clusters
        print(f"✓ Created {len(clusters)} clusters")
        print(f"✓ Avg cluster size: {np.mean([len(c) for c in clusters.values()]):.1f}")
        return clusters


def contrastive_loss(query_emb, pos_emb, neg_embs=None):
    batch_size = query_emb.size(0)
    all_scores = torch.matmul(query_emb, pos_emb.t())
    labels = torch.arange(batch_size).to(query_emb.device)

    if neg_embs is not None:
        neg_scores = torch.matmul(query_emb, neg_embs.t())
        all_scores = torch.cat([all_scores, neg_scores], dim=1)

    loss = F.cross_entropy(all_scores, labels)
    return loss

# ============================================================================
# DATASET + TRAINING LOOP
# ============================================================================

class RetrievalDataset(Dataset):
    """Dataset for retrieval training"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        if 'passage' in item:
            passage = item['passage']
        elif 'positive' in item:
            passage = item['positive']
        else:
            passage = ''
        return {'query': query, 'passage': passage, 'idx': idx}


def collate_fn(batch, tokenizer, max_length):
    queries = [item['query'] for item in batch]
    passages = [item['passage'] for item in batch]
    indices = [item['idx'] for item in batch]

    query_inputs = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    passage_inputs = tokenizer(passages, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    return {'query_inputs': query_inputs, 'passage_inputs': passage_inputs, 'indices': indices}


def train_dpr_ictp(model, train_data, val_data, config, tokenizer, device, phase="pFT"):
    """Train DPR with ICT-P negative sampling"""
    print(f"\n{'='*80}\nTRAINING PHASE: {phase}\n{'='*80}")

    train_dataset = RetrievalDataset(train_data, tokenizer, config['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, config['max_length']),
                              num_workers=config['num_workers'], pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config[f'max_epochs_{phase.lower()}']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=total_steps)
    icp_trainer = ICTPTrainer(model, config, tokenizer, device)

    # ✅ FIX: handle dataset format (MS MARCO vs Mr. TyDi)
    if 'passage' in train_data[0]:
        all_passages = [item['passage'] for item in train_data]
    elif 'positive' in train_data[0]:
        all_passages = [item['positive'] for item in train_data]
    else:
        raise KeyError("Dataset must contain either 'passage' or 'positive' field.")

    model.train()
    global_step, best_loss = 0, float('inf')

    for epoch in range(config[f'max_epochs_{phase.lower()}']):
        print(f"\nEpoch {epoch + 1}/{config[f'max_epochs_{phase.lower()}']}")
        if epoch % config['clustering_frequency'] == 0:
            icp_trainer.cluster_passages(all_passages)

        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training {phase}")

        for batch_idx, batch in enumerate(progress_bar):
            query_inputs = {k: v.to(device) for k, v in batch['query_inputs'].items()}
            passage_inputs = {k: v.to(device) for k, v in batch['passage_inputs'].items()}

            query_emb, pos_emb = model(query_inputs, passage_inputs)
            loss = contrastive_loss(query_emb, pos_emb) / config['gradient_accumulation_steps']
            loss.backward()

            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f"{config['model_path']}/ict_p_baseline/{phase}_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved: {best_model_path}")

    final_model_path = f"{config['model_path']}/ict_p_baseline/{phase}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    return model

# ============================================================================
# EVALUATION
# ============================================================================

def compute_mrr_recall(query_embs, passage_embs, qrels, k=100):
    dim = passage_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs.numpy())
    D, I = index.search(query_embs.numpy(), k)

    mrr_scores, recall_scores = [], []
    for query_idx, retrieved in enumerate(I):
        relevant = qrels.get(query_idx, set())
        if not relevant:
            continue
        for rank, pid in enumerate(retrieved, 1):
            if pid in relevant:
                mrr_scores.append(1.0 / rank)
                break
        recall_scores.append(len(set(retrieved) & relevant) / len(relevant))

    return np.mean(mrr_scores), np.mean(recall_scores)


def evaluate_model(model, test_data, config, tokenizer, device):
    print("\n" + "="*80 + "\nEVALUATION\n" + "="*80)
    model.eval()
    lang_data = defaultdict(list)
    for item in test_data:
        lang_data[item['language']].append(item)

    results = {}
    for lang, data in lang_data.items():
        print(f"\nEvaluating {lang}...")
        queries = [d['query'] for d in data]
        passages = [d['passage'] for d in data]
        query_embs = model.encode_queries(queries, tokenizer, device)
        passage_embs = model.encode_passages(passages, tokenizer, device)
        qrels = {i: {i} for i in range(len(queries))}
        mrr, recall = compute_mrr_recall(query_embs, passage_embs, qrels, k=100)
        results[lang] = {'MRR@100': mrr, 'Recall@100': recall}
        print(f"  MRR@100: {mrr:.4f} | Recall@100: {recall:.4f}")

    avg_mrr = np.mean([v['MRR@100'] for v in results.values()])
    avg_recall = np.mean([v['Recall@100'] for v in results.values()])
    print(f"\n{'='*80}\nAVERAGE RESULTS\n{'='*80}")
    print(f"MRR@100: {avg_mrr:.4f} | Recall@100: {avg_recall:.4f}")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{CONFIG['base_path']}/logs/evaluation_results.csv")
    print(f"✓ Results saved to {CONFIG['base_path']}/logs/evaluation_results.csv")
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: ICT-P BASELINE TRAINING")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

model = DPRModel(CONFIG).to(device)
print(f"✓ Model initialized")

# === STEP 1: Pre-finetuning on MS MARCO ===
print("\n[STEP 1] Pre-finetuning on MS MARCO...")
model = train_dpr_ictp(model, msmarco_data, None, CONFIG, tokenizer, device, phase="pFT")

# === STEP 2: Finetuning on Mr. TyDi ===
print("\n[STEP 2] Finetuning on Mr. TyDi...")
model = train_dpr_ictp(model, mrtydi_train, mrtydi_test, CONFIG, tokenizer, device, phase="FT")

# === STEP 3: Evaluation ===
print("\n[STEP 3] Final Evaluation...")
results = evaluate_model(model, mrtydi_test, CONFIG, tokenizer, device)

print("\n" + "="*80)
print("PHASE 1 COMPLETE")
print("="*80)
print("✓ ICT-P baseline model trained successfully")
print("✓ All checkpoints saved")
print("✓ Evaluation complete")
print("\nNext: Proceed to Phase 2 (AXL-ICT Novelty Integration)")
