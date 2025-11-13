# ============================================================================
# CONFIGURATION (FINAL FIXED VERSION)
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
    "max_epochs_pft": 1,      # Pre-finetuning epochs (MS MARCO)
    "max_epochs_ft": 1,       # Finetuning epochs (Mr. TyDi)
    "gradient_accumulation_steps": 2,
    "save_steps": 200,
    "num_clusters": 50,       # For ICT-P clustering
    "clustering_frequency": 1
}
