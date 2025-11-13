# ============================================================================
# PHASE 1: setup
# ============================================================================

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Check environment
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB = 'google.colab' in sys.modules

if IS_KAGGLE:
    BASE_PATH = '/kaggle/working'
    DATA_PATH = '/kaggle/working/data'
    MODEL_PATH = '/kaggle/working/models'
elif IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content'
    DATA_PATH = '/content/data'
    MODEL_PATH = '/content/models'
else:
    BASE_PATH = '.'
    DATA_PATH = './data'
    MODEL_PATH = './models'

print(f"✓ Running on: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")
print(f"✓ Base Path: {BASE_PATH}")

# Create directory structure
os.makedirs(f"{DATA_PATH}/msmarco", exist_ok=True)
os.makedirs(f"{DATA_PATH}/mrtydi", exist_ok=True)
os.makedirs(f"{DATA_PATH}/processed", exist_ok=True)
os.makedirs(f"{MODEL_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{MODEL_PATH}/ict_p_baseline", exist_ok=True)
os.makedirs(f"{BASE_PATH}/logs", exist_ok=True)

print("✓ Directory structure created")

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================

print("\n" + "="*80)
print("INSTALLING DEPENDENCIES")
print("="*80)
!pip install faiss-cpu
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q datasets==2.14.0 accelerate==0.24.0
!pip install -q faiss-gpu sentence-transformers
!pip install -q ir-datasets pyserini scikit-learn tqdm pandas numpy
!pip install -q tensorboard wandb
!pip install -U transformers==4.44.2 huggingface-hub==0.24.6


print("✓ All dependencies installed successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model Configuration
    "model_name": "bert-base-multilingual-cased",
    "embedding_dim": 768,
    "max_length": 512,

    # Training Configuration
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-5,
    "warmup_steps": 1000,
    "max_epochs_pft": 40,  # Pre-finetuning epochs
    "max_epochs_ft": 40,   # Finetuning epochs
    "eval_steps": 500,
    "save_steps": 1000,
    "clustering_frequency": 10,  # Re-cluster every 10 epochs for ICT-P

    # ICT-P Configuration
    "num_clusters": 1000,
    "num_hard_negatives": 7,  # In-batch + hard negatives

    # Data Configuration
    "mr_tydi_languages": ["ar", "bn", "en", "fi", "id", "ja", "ko", "ru", "sw", "te", "th"],
    "mr_tydi_balanced_size": 2019,  # Smallest language (Korean)

    # Language Families for Curriculum Learning (Phase 2)
    "language_families": {
        "indo_european": ["en", "ru"],
        "afro_asiatic": ["ar"],
        "indo_aryan": ["bn", "te"],
        "uralic": ["fi"],
        "austronesian": ["id"],
        "japonic": ["ja"],
        "koreanic": ["ko"],
        "niger_congo": ["sw"],
        "tai_kadai": ["th"]
    },

    # Paths
    "data_path": DATA_PATH,
    "model_path": MODEL_PATH,
    "base_path": BASE_PATH
}

# Save configuration
with open(f"{BASE_PATH}/config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("\n✓ Configuration saved to config.json")
print(f"✓ Training on {len(CONFIG['mr_tydi_languages'])} languages")
print(f"✓ Balanced dataset size: {CONFIG['mr_tydi_balanced_size']} samples per language")

# ============================================================================
# DEPENDENCY VERIFICATION
# ============================================================================

import numpy, pandas, pyarrow, datasets
print("\n" + "="*80)
print("DEPENDENCY VERSIONS")
print("="*80)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("PyArrow:", pyarrow.__version__)
print("Datasets:", datasets.__version__)


