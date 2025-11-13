# ============================================================================
# PHASE 2: AXL-ICT - EXCEED PAPER'S ICT-P BASELINE (0.454 MRR@100)
# Target: MRR@100 > 0.50, Recall@100 > 0.90
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import os
import random

print("\n" + "="*80)
print("PHASE 2: AXL-ICT - SURPASS PAPER'S ICT-P BASELINE")
print(f"Paper's ICT-P: MRR@100 = 0.454 | Target: > 0.50")
print("="*80)

# ============================================================================
# ENHANCED CONFIGURATION (Paper-aligned + Improvements)
# ============================================================================

AXL_CONFIG = {
    **CONFIG,
    
    # ----- Model Architecture (from paper Section 4.4) -----
    "model_name": "bert-base-multilingual-cased",  # mBERT like paper
    "pooling": "cls",  # CLS pooling (paper standard)
    
    # ----- Training (aligned with paper: 40 epochs each phase) -----
    "max_epochs_pft": 40,  # Pre-finetuning on MS MARCO
    "max_epochs_ft": 40,   # Finetuning on Mr. TyDi
    "max_epochs_axl": 10,  # AXL-ICT additional epochs
    "learning_rate": 1e-5,  # Paper's LR
    "batch_size": 16,       # Paper's batch size
    "warmup_steps": 1000,   # More warmup
    
    # ----- ICT-P Clustering (from paper Section 3.3) -----
    "num_clusters": 50,
    "clustering_frequency": 10,  # Re-cluster every 10 epochs (paper)
    
    # ----- NOVELTY 1: Generative Adversarial Negatives -----
    "mt5_model": "google/mt5-base",  # Larger model for better quality
    "num_synthetic_negatives": 3,    # More negatives per query
    "generation_temperature": 1.2,   # Higher diversity
    "generation_top_p": 0.92,
    
    # ----- NOVELTY 2: Cross-Lingual Curriculum -----
    "language_families": {
        "Indo-European": ["english", "russian"],
        "Afro-Asiatic": ["arabic"],
        "Indo-Aryan": ["bengali", "telugu"],
        "Uralic": ["finnish"],
        "Austronesian": ["indonesian"],
        "Koreanic": ["korean"],
        "Japonic": ["japanese"],
        "Niger-Congo": ["swahili"]
    },
    "curriculum_stages": 3,  # 3-stage curriculum
    "stage1_ratio": 0.9,  # Stage 1: 90% high-resource
    "stage2_ratio": 0.5,  # Stage 2: 50% mixed
    "stage3_ratio": 0.2,  # Stage 3: 20% high-resource (full mix)
    
    # ----- NOVELTY 3: Adversarial Filtering -----
    "adversarial_filter_ratio": 0.6,  # Keep only top 60% hardest
    "similarity_threshold": 0.6,      # Higher threshold = harder negatives
    "use_margin_loss": True,          # Add margin-based adversarial loss
    "margin": 0.2,                    # Margin for triplet loss
    
    # ----- Enhanced Loss Function -----
    "lambda_adversarial": 0.5,  # 50% weight on adversarial
    "lambda_margin": 0.2,       # 20% weight on margin loss
    "temperature": 0.05,        # Temperature for contrastive loss (lower = harder)
}

print(f"✓ Enhanced configuration loaded")
print(f"  • Target: Beat paper's ICT-P (MRR@100 = 0.454)")
print(f"  • Training: {AXL_CONFIG['max_epochs_axl']} AXL-ICT epochs")
print(f"  • Synthetic negatives: {AXL_CONFIG['num_synthetic_negatives']} per query")

# ============================================================================
# ENHANCED LOSS FUNCTIONS
# ============================================================================

def enhanced_contrastive_loss(query_emb, pos_emb, neg_emb=None, config=None):
    """
    Enhanced contrastive loss with temperature scaling and hard negative focus
    """
    batch_size = query_emb.size(0)
    temperature = config.get('temperature', 0.05)
    
    # Standard contrastive loss with temperature
    scores = torch.matmul(query_emb, pos_emb.t()) / temperature
    labels = torch.arange(batch_size).to(query_emb.device)
    standard_loss = F.cross_entropy(scores, labels)
    
    if neg_emb is None:
        return standard_loss
    
    # Adversarial hard negative loss
    neg_scores = torch.matmul(query_emb, neg_emb.t()) / temperature
    combined_scores = torch.cat([scores, neg_scores], dim=1)
    adversarial_loss = F.cross_entropy(combined_scores, labels)
    
    # Margin-based triplet loss (additional novelty)
    if config.get('use_margin_loss', False):
        pos_similarity = torch.sum(query_emb * pos_emb, dim=1)
        neg_similarity = torch.matmul(query_emb, neg_emb.t()).max(dim=1)[0]
        margin = config.get('margin', 0.2)
        margin_loss = torch.clamp(margin - pos_similarity + neg_similarity, min=0).mean()
        
        # Weighted combination
        lambda_adv = config['lambda_adversarial']
        lambda_margin = config['lambda_margin']
        
        total_loss = (1 - lambda_adv - lambda_margin) * standard_loss + \
                     lambda_adv * adversarial_loss + \
                     lambda_margin * margin_loss
        return total_loss
    
    # Without margin loss
    lambda_adv = config['lambda_adversarial']
    return (1 - lambda_adv) * standard_loss + lambda_adv * adversarial_loss

# ============================================================================
# NOVELTY 1: IMPROVED MT5 GENERATOR
# ============================================================================

class EnhancedMT5Generator:
    """
    Enhanced mT5 generator with better prompting and quality control
    """
    
    def __init__(self, model_name="google/mt5-base", device="cuda"):
        print(f"\n[Novelty 1] Loading enhanced mT5 generator...")
        self.device = device
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model.eval()
        print(f"✓ mT5-base loaded for high-quality adversarial generation")
    
    def generate_adversarial_negatives(
        self,
        queries: List[str],
        positive_passages: List[str],
        num_negatives: int = 5
    ) -> List[List[str]]:
        """
        Generate high-quality adversarial negatives with improved prompting
        """
        all_negatives = []
        
        print(f"  Generating {num_negatives} adversarial negatives...")
        
        for query, pos in tqdm(zip(queries, positive_passages), total=len(queries)):
            # Multi-strategy prompting for better quality
            prompts = [
                f"generate confusing passage for query: {query[:80]}",
                f"create misleading answer about: {query[:80]}",
                f"write wrong information for: {query[:80]}"
            ]
            
            negatives = []
            
            with torch.no_grad():
                for prompt in prompts[:min(3, num_negatives)]:
                    inputs = self.tokenizer(
                        prompt,
                        max_length=128,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # High-quality generation
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_return_sequences=2,
                        do_sample=True,
                        top_k=40,
                        top_p=0.92,
                        temperature=1.2,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    
                    for out in outputs:
                        text = self.tokenizer.decode(out, skip_special_tokens=True)
                        if len(text.split()) >= 15:  # Quality filter
                            negatives.append(text)
            
            # Ensure we have enough negatives
            while len(negatives) < num_negatives:
                negatives.append(f"misleading content about {query[:40]}")
            
            all_negatives.append(negatives[:num_negatives])
        
        return all_negatives

# ============================================================================
# NOVELTY 2: STAGED CURRICULUM SAMPLER
# ============================================================================

class StagedCurriculumSampler:
    """
    3-stage curriculum: High-resource → Mixed → Full diversity
    """
    
    def __init__(self, data: List[Dict], language_families: Dict, config: Dict):
        print(f"\n[Novelty 2] Initializing staged curriculum...")
        self.data = data
        self.language_families = language_families
        self.config = config
        self.current_epoch = 0
        self.total_epochs = config['max_epochs_axl']
        
        # Categorize by resource level
        self.high_resource_langs = ["english", "russian", "arabic"]
        self.medium_resource_langs = ["japanese", "korean", "indonesian", "finnish"]
        self.low_resource_langs = ["bengali", "telugu", "swahili"]
        
        self.high_resource_data = [d for d in data if d['language'] in self.high_resource_langs]
        self.medium_resource_data = [d for d in data if d['language'] in self.medium_resource_langs]
        self.low_resource_data = [d for d in data if d['language'] in self.low_resource_langs]
        
        print(f"✓ Staged curriculum ready:")
        print(f"  • High-resource: {len(self.high_resource_data)} samples")
        print(f"  • Medium-resource: {len(self.medium_resource_data)} samples")
        print(f"  • Low-resource: {len(self.low_resource_data)} samples")
    
    def get_stage(self) -> int:
        """Determine current curriculum stage"""
        progress = self.current_epoch / self.total_epochs
        if progress < 0.33:
            return 1  # Stage 1: Focus on high-resource
        elif progress < 0.67:
            return 2  # Stage 2: Balanced mix
        else:
            return 3  # Stage 3: Full diversity
    
    def sample_curriculum_batch(self, batch_size: int) -> List[Dict]:
        """Sample based on current curriculum stage"""
        stage = self.get_stage()
        
        if stage == 1:
            # Stage 1: 90% high-resource, 10% others
            n_high = int(batch_size * 0.9)
            n_medium = int(batch_size * 0.07)
            n_low = batch_size - n_high - n_medium
        elif stage == 2:
            # Stage 2: 50% high, 30% medium, 20% low
            n_high = int(batch_size * 0.5)
            n_medium = int(batch_size * 0.3)
            n_low = batch_size - n_high - n_medium
        else:
            # Stage 3: 20% high, 40% medium, 40% low (focus on challenging)
            n_high = int(batch_size * 0.2)
            n_medium = int(batch_size * 0.4)
            n_low = batch_size - n_high - n_medium
        
        batch = []
        if n_high > 0 and len(self.high_resource_data) > 0:
            batch.extend(random.choices(self.high_resource_data, k=n_high))
        if n_medium > 0 and len(self.medium_resource_data) > 0:
            batch.extend(random.choices(self.medium_resource_data, k=n_medium))
        if n_low > 0 and len(self.low_resource_data) > 0:
            batch.extend(random.choices(self.low_resource_data, k=n_low))
        
        # Fill if needed
        while len(batch) < batch_size:
            batch.append(random.choice(self.data))
        
        return batch[:batch_size]
    
    def next_epoch(self):
        self.current_epoch += 1
        print(f"  → Curriculum Stage {self.get_stage()}/3")

# ============================================================================
# NOVELTY 3: STRICTER ADVERSARIAL FILTER
# ============================================================================

class StrictAdversarialFilter:
    """
    Stricter filtering: only keep negatives that are truly adversarial
    """
    
    def __init__(self, model, config, tokenizer, device):
        print(f"\n[Novelty 3] Initializing strict adversarial filter...")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.filter_ratio = config['adversarial_filter_ratio']
        self.threshold = config['similarity_threshold']
        print(f"✓ Strict filter: top {self.filter_ratio*100:.0f}%, threshold={self.threshold}")
    
    def filter_hard_negatives(
        self,
        queries: List[str],
        all_negatives: List[List[str]],
        positives: List[str]
    ) -> List[List[str]]:
        """
        Keep only truly adversarial negatives (high similarity but wrong)
        """
        filtered = []
        
        print(f"  Applying strict adversarial filtering...")
        
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), 8), desc="Filtering"):
                batch_queries = queries[i:i+8]
                batch_negs = all_negatives[i:i+8]
                batch_pos = positives[i:i+8]
                
                # Encode queries and positives
                q_inputs = self.tokenizer(batch_queries, padding=True, truncation=True,
                                         max_length=self.config['max_length'], return_tensors='pt').to(self.device)
                q_embs = self.model.query_encoder(**q_inputs)
                
                p_inputs = self.tokenizer(batch_pos, padding=True, truncation=True,
                                         max_length=self.config['max_length'], return_tensors='pt').to(self.device)
                pos_embs = self.model.passage_encoder(**p_inputs)
                
                # Filter each query's negatives
                for j, (q_emb, pos_emb, negs) in enumerate(zip(q_embs, pos_embs, batch_negs)):
                    if len(negs) == 0:
                        filtered.append([])
                        continue
                    
                    # Encode negatives
                    neg_inputs = self.tokenizer(negs, padding=True, truncation=True,
                                               max_length=self.config['max_length'], return_tensors='pt').to(self.device)
                    neg_embs = self.model.passage_encoder(**neg_inputs)
                    
                    # Score negatives
                    scores = torch.matmul(q_emb.unsqueeze(0), neg_embs.t()).squeeze(0)
                    pos_score = torch.matmul(q_emb.unsqueeze(0), pos_emb.unsqueeze(0).t()).item()
                    
                    # Keep only adversarial: similarity > threshold AND < positive score
                    valid_mask = (scores >= self.threshold) & (scores < pos_score)
                    
                    if valid_mask.sum() == 0:
                        filtered.append([])
                        continue
                    
                    valid_indices = torch.where(valid_mask)[0]
                    valid_scores = scores[valid_indices]
                    k = max(1, int(len(valid_indices) * self.filter_ratio))
                    
                    _, top_k = torch.topk(valid_scores, k=min(k, len(valid_scores)))
                    final_indices = valid_indices[top_k]
                    
                    hard_negs = [negs[idx] for idx in final_indices.cpu().tolist()]
                    filtered.append(hard_negs)
        
        print(f"  ✓ Filtered to {sum(len(f) for f in filtered)} high-quality adversarial negatives")
        return filtered

# ============================================================================
# REST OF IMPLEMENTATION (Dataset, Training Loop) - Same as before but with:
# - enhanced_contrastive_loss instead of axlict_loss
# - EnhancedMT5Generator instead of MT5AdversarialGenerator
# - StagedCurriculumSampler instead of CrossLingualCurriculumSampler
# - StrictAdversarialFilter instead of AdversarialNegativeFilter
# ============================================================================

# [Include previous AXLICTDataset, axlict_collate_fn, AXLICTTrainer classes here]
# [Include train_axlict function with enhanced_contrastive_loss]

# ============================================================================
# EXECUTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 2 EXECUTION: TARGET > 0.50 MRR@100")
print("="*80)

# Load Phase 1 best model
model.load_state_dict(torch.load(f"{CONFIG['model_path']}/ict_p_baseline/FT_best.pt"))
model.to(device)

# Initialize enhanced components
mt5_generator = EnhancedMT5Generator(AXL_CONFIG['mt5_model'], device)
curriculum_sampler = StagedCurriculumSampler(mrtydi_train, AXL_CONFIG['language_families'], AXL_CONFIG)
adversarial_filter = StrictAdversarialFilter(model, AXL_CONFIG, tokenizer, device)
icp_trainer = AXLICTTrainer(model, AXL_CONFIG, tokenizer, device)

# Train
model = train_axlict(model, mrtydi_train, mrtydi_test, AXL_CONFIG, tokenizer, device,
                     mt5_generator, curriculum_sampler, adversarial_filter, icp_trainer)

# Evaluate
results_axl = evaluate_model(model, mrtydi_test, AXL_CONFIG, tokenizer, device)

# Compare with paper
print("\n" + "="*80)
print("RESULTS vs. PAPER'S ICT-P")
print("="*80)
print(f"Paper's ICT-P: MRR@100 = 0.454, Recall@100 = 0.870")
print(f"Our AXL-ICT:   MRR@100 = {np.mean([v['MRR@100'] for v in results_axl.values()]):.4f}, " +
      f"Recall@100 = {np.mean([v['Recall@100'] for v in results_axl.values()]):.4f}")
print("="*80)
