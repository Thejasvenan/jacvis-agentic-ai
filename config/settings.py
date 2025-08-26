import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.0-pro"  
GEMINI_API_VERSION = "v1"

API_RATE_LIMITS = {
    "gemini": {
        "requests_per_minute": 15,  
        "tokens_per_minute": 15000,
        "daily_limit": 1500,  
        "request_delay": 4.0  
    }
}

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINE_TUNED_MODEL_PATH = "./models/finetuned_rpg_generator"

DATASET_PATH = "./data/processed/rpg_dataset.json"
MAX_DATASET_SIZE = 1000  
MIN_DATASET_SIZE = 100

TRAINING_SETTINGS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_steps": 1000,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "optim": "paged_adamw_32bit"
}

RPG_SETTINGS = {
    "themes": ["fantasy", "sci-fi", "post-apocalyptic", "medieval", "cyberpunk"],
    "difficulty_levels": ["easy", "medium", "hard"],
    "level_sizes": ["small", "medium", "large"]
}