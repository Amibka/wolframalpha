import os
from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

for d in [DATA_DIR, CHECKPOINTS_DIR, TOKENIZER_DIR]:
    d.mkdir(exist_ok=True)

TRAINING_DATA_PATH = DATA_DIR / "training_data.json"
VOCAB_PATH = TOKENIZER_DIR / "vocab.json"

# Гиперпараметры модели
MODEL_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "n_decoder_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_seq_length": 128,
    "vocab_size": 10000,
}

# Обучение
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "warmup_steps": 4000,
    "save_every": 5,
    "train_split": 0.9,
}

# Датасет
DATASET_CONFIG = {
    "num_samples": 100000,
    "random_seed": 42,
}

# Inference
INFERENCE_CONFIG = {
    "beam_size": 5,
    "max_decode_length": 128,
}

# Спец. токены
SPECIAL_TOKENS = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"