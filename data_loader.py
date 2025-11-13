

# ============================================================================
# DATA ACQUISITION AND PREPROCESSING
# ============================================================================
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
import random
import numpy as np

print("\n" + "="*80)
print("DATA ACQUISITION AND PREPROCESSING")
print("="*80)

# Create directories
os.makedirs(f"{CONFIG['data_path']}/msmarco", exist_ok=True)
os.makedirs(f"{CONFIG['data_path']}/mrtydi", exist_ok=True)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print(f"✓ Loaded tokenizer: {CONFIG['model_name']}")

# ============================================================================
# MS MARCO DATA LOADING (Pre-finetuning)
# ============================================================================
class MSMARCODataLoader:
    """Efficient MS MARCO data loader using streaming and chunking"""

    def __init__(self, config, max_samples=100000):
        self.config = config
        self.max_samples = max_samples
        self.data_path = f"{config['data_path']}/msmarco"

    def load_triples(self):
        """Load MS MARCO triples dataset efficiently"""
        print("\n[MS MARCO] Loading training triples...")
        try:
            dataset = load_dataset(
                "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
                "triplet-hard",
                split="train",
                streaming=True
            )

            triples = []
            for i, example in enumerate(tqdm(dataset, desc="Loading MS MARCO", total=self.max_samples)):
                if i >= self.max_samples:
                    break
                triples.append({
                    'query': example['query'],
                    'positive': example['positive'],
                    'negative': example['negative']
                })

            print(f"✓ Loaded {len(triples)} MS MARCO triples")
            df = pd.DataFrame(triples)
            df.to_csv(f"{self.data_path}/msmarco_triples.csv", index=False)
            print(f"✓ Saved to {self.data_path}/msmarco_triples.csv")
            return triples

        except Exception as e:
            print(f"✗ Error loading MS MARCO: {e}")
            raise

# Load MS MARCO
msmarco_loader = MSMARCODataLoader(CONFIG, max_samples=100000)
msmarco_data = msmarco_loader.load_triples()

# ============================================================================
# MR. TYDI DATA LOADING (Finetuning) WITH PROPORTIONAL BALANCING
# ============================================================================
class MrTyDiDataLoader:
    """Mr. TyDi data loader using castorini/mr-tydi from Hugging Face Hub"""

    def __init__(self, config):
        self.config = config
        self.languages = config['mr_tydi_languages']
        self.balanced_size = config['mr_tydi_balanced_size']
        self.data_path = f"{config['data_path']}/mrtydi"

    def load_and_balance(self):
        """Load and balance Mr. TyDi across all languages"""
        print("\n[Mr. TyDi] Loading and balancing multilingual data...")
        all_train_data = []
        all_test_data = []

        for lang in tqdm(self.languages, desc="Loading languages"):
            try:
                # Load the dataset for the given language
                ds_train = load_dataset("castorini/mr-tydi", lang, split="train")
                ds_test = load_dataset("castorini/mr-tydi", lang, split="dev")

                # Extract positive passages for training
                lang_data = []
                for ex in ds_train:
                    for pos in ex.get("positive_passages", []):
                        lang_data.append({
                            "query": ex["query"],
                            "passage": pos["text"],
                            "language": lang
                        })

                # Balance per language
                if len(lang_data) > self.balanced_size:
                    lang_data = random.sample(lang_data, self.balanced_size)
                elif len(lang_data) < self.balanced_size and len(lang_data) > 0:
                    extra = random.choices(lang_data, k=self.balanced_size - len(lang_data))
                    lang_data.extend(extra)

                all_train_data.extend(lang_data)
                print(f"  ✓ {lang}: {len(lang_data)} samples (balanced)")

                # Prepare test set (use same fields)
                test_data = []
                for ex in ds_test:
                    for pos in ex.get("positive_passages", []):
                        test_data.append({
                            "query": ex["query"],
                            "passage": pos["text"],
                            "language": lang
                        })
                all_test_data.extend(test_data)
                print(f"  ✓ {lang}: {len(test_data)} test samples")

            except Exception as e:
                print(f"  ✗ Error loading {lang}: {e}")
                continue

        print(f"\n✓ Total training samples: {len(all_train_data)}")
        print(f"✓ Total test samples: {len(all_test_data)}")

        # Save results
        pd.DataFrame(all_train_data).to_csv(f"{self.data_path}/mrtydi_train_balanced.csv", index=False)
        pd.DataFrame(all_test_data).to_csv(f"{self.data_path}/mrtydi_test.csv", index=False)
        print(f"✓ Saved balanced data to {self.data_path}/")

        return all_train_data, all_test_data

# Load Mr. TyDi
mrtydi_loader = MrTyDiDataLoader(CONFIG)
mrtydi_train, mrtydi_test = mrtydi_loader.load_and_balance()

print("\n✓ DATA LOADING COMPLETE")
print(f"  MS MARCO: {len(msmarco_data)} samples")
print(f"  Mr. TyDi Train: {len(mrtydi_train)} samples ({len(CONFIG['mr_tydi_languages'])} languages)")
print(f"  Mr. TyDi Test: {len(mrtydi_test)} samples")