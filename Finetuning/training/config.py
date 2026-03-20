#!/usr/bin/env python3
"""
Configuration for fine-tuning Llama-4-Maverick on task results.
Updated for Google Colab: finetuning folder uploaded in Colab, images in Drive.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Detect if running in Google Colab
RUNNING_IN_COLAB = os.path.exists("/content") or os.environ.get("COLAB_GPU", "").strip() != ""

# Model configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",  # Using Qwen3-VL-8B model
    "local_model_path": None,  # Set to local path if model is downloaded locally
    "use_local": False,
    "trust_remote_code": True,
    "hf_token": os.environ.get("HF_TOKEN", ""),  # Set HF_TOKEN environment variable
    "load_in_8bit": False,  # Enable 8-bit quantization to save memory (requires bitsandbytes)
    "load_in_4bit": True,  # Enable 4-bit quantization for even more memory savings (ENABLED for OOM prevention)
}

# Training hyperparameters
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 32,  # Increased to 32 for better memory management (OOM prevention)
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "save_strategy": "epoch",  # Save after each epoch
    "eval_steps": 500,
    "evaluation_strategy": "no",  # Disable evaluation during training to prevent OOM (we'll do manual evaluation)
    "max_steps": -1,  # -1 means use num_train_epochs
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True,  # Enable gradient checkpointing to save memory
    "dataloader_num_workers": 0,  # Set to 0 to avoid multiprocessing issues with images
    "dataloader_pin_memory": False,  # Disable pin memory to save RAM
    "max_length": 768,  # Reduced from 1024 to 768 for even more aggressive memory savings (OOM prevention for 40GB)
    "remove_unused_columns": False,
    "report_to": "tensorboard",  # Changed to "tensorboard" for logging (can be "none", "wandb", etc.)
}

# PEFT (LoRA) configuration
PEFT_CONFIG = {
    "r": 2,  # Reduced from 4 to 2 for aggressive memory savings (OOM prevention)
    "lora_alpha": 4,  # Scaled down proportionally
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Data configuration
DATA_CONFIG = {
    "csv_path": "task_results/task_results.csv",
    "test_csv_path": "task_results_test/task_results_test.csv",  # Test dataset CSV
    "task_start": None,  # None = use all tasks
    "task_end": None,    # None = use all tasks
    "test_task_start": None,  # None = process all tasks
    "test_task_end": None,    # None = process all tasks
    "image_base_path": None,  # Will be resolved relative to CSV location
    "test_image_base_path": None,  # Base path for test images
    "val_split": 0.1,  # 10% for validation
}

# Path configuration
# finetuning folder is uploaded in Colab, so use relative paths for CSV files
PROJECT_ROOT = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent

# Update paths to point to finetuning directory (relative paths work in Colab)
DATA_CONFIG["csv_path"] = str(FINETUNING_ROOT / "task_results.csv")
DATA_CONFIG["test_csv_path"] = str(FINETUNING_ROOT / "task_results_clean.csv")  # Updated to use cleaned CSV

# Image paths - conditionally set based on environment
# In Colab: training images are in Google Drive at /content/drive/MyDrive/images
# In Colab: test images are in Google Drive at /content/drive/MyDrive/images_test
# Locally: images are in finetuning/images/
if RUNNING_IN_COLAB:
    # Google Colab: training images are mounted from Drive
    # CSV Image_id format: "task_results/images/3DSlicer_endtoend049_step1.png"
    # Code extracts filename: "3DSlicer_endtoend049_step1.png"
    # Looks for it in: /content/drive/MyDrive/images/filename.png
    DRIVE_IMAGES_PATH = Path("/content/drive/MyDrive/images")
    if DATA_CONFIG["image_base_path"] is None:
        DATA_CONFIG["image_base_path"] = str(DRIVE_IMAGES_PATH)
    
    # Google Colab: test images are in images_test folder
    # CSV Image_id format: "test_CVPR/images/Orthanc_endtoend001_step1.png"
    # Code extracts filename: "Orthanc_endtoend001_step1.png"
    # Looks for it in: /content/drive/MyDrive/images_test/filename.png
    DRIVE_TEST_IMAGES_PATH = Path("/content/drive/MyDrive/images_test")
    if DATA_CONFIG["test_image_base_path"] is None:
        DATA_CONFIG["test_image_base_path"] = str(DRIVE_TEST_IMAGES_PATH)
    
    # Also update for local paths if needed (for consistency)
    # In Colab, test images are always in /content/drive/MyDrive/images_test
else:
    # Local: images are in finetuning/images/
    if DATA_CONFIG["image_base_path"] is None:
        DATA_CONFIG["image_base_path"] = str(FINETUNING_ROOT / "images")
    if DATA_CONFIG["test_image_base_path"] is None:
        DATA_CONFIG["test_image_base_path"] = str(FINETUNING_ROOT / "images")

# Output configuration
OUTPUT_CONFIG = {
    "output_dir": "finetuning/outputs",
    "checkpoint_dir": "finetuning/checkpoints",
    "logs_dir": "finetuning/logs",
    "logging_dir": "finetuning/logs",  # For TrainingArguments logging_dir
}

# Create output directories relative to project root
for key, path in OUTPUT_CONFIG.items():
    OUTPUT_CONFIG[key] = str(PROJECT_ROOT / path)
    os.makedirs(OUTPUT_CONFIG[key], exist_ok=True)

# Inference configuration - OPTIMIZED FOR MAXIMUM SPEED (40GB GPU limit)
INFERENCE_CONFIG = {
    "max_new_tokens": 256,      # Reduced from 512 - JSON outputs are typically 100-300 tokens (2x faster)
    "temperature": 0.1,          # Lower temperature for faster, more deterministic output
    "top_p": 0.9,
    "do_sample": False,          # Use greedy decoding (2-3x faster!)
    "num_return_sequences": 1,
    "early_stopping": True,      # Stop early if possible
    "use_cache": True,           # Enable KV cache for faster generation (10-20% speedup)
    "repetition_penalty": 1.0,   # No repetition penalty (faster, no effect on quality)
    "clear_cache_frequency": 20,  # Clear cache every N steps (increased from 10 for speed)
    "skip_special_tokens": True,  # Skip special tokens during decoding (faster)
}

# Memory extraction configuration
MEMORY_CONFIG = {
    "short_term_key": "Short Term",
    "long_term_key": "Long Term",
    "predicted_action_key": "Predicted.predicted_action",
}

# Ablation study configuration
# Set to False to remove specific components from training/inference
ABLATION_CONFIG = {
    "use_grounding": True,           # Include Grounding in prompt
    "use_short_term_memory": True,   # Include Short Term Memory in prompt
    "use_long_term_memory": True,    # Include Long Term Memory in prompt
}

def get_config(config_key: str = None) -> Dict[str, Any]:
    """
    Get configuration dictionary.
    
    Args:
        config_key: Optional key to get specific config section
        
    Returns:
        Configuration dictionary or specific section
    """
    full_config = {
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "peft": PEFT_CONFIG,
        "data": DATA_CONFIG,
        "output": OUTPUT_CONFIG,
        "inference": INFERENCE_CONFIG,
        "memory": MEMORY_CONFIG,
        "ablation": ABLATION_CONFIG,
    }
    
    if config_key:
        return full_config.get(config_key, {})
    return full_config

