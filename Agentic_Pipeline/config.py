#!/usr/bin/env python3
"""
Configuration module for the Medical Visual Agent system.
This file contains all configurable parameters for the system.
"""

import os
from typing import Dict, Any, Optional

# General system configuration
SYSTEM_CONFIG = {
    "max_total_steps": 15,              # Maximum number of steps for a single task
    "max_retries_per_step": 3,          # Maximum number of retries per step
    "verbose": True,                    # Enable verbose logging
    "debug_mode": False,                # Enable debug mode
}

# HuggingFace dataset configuration
DATASET_CONFIG = {
    "hf_dataset_name": "rishuKumar404/test_cvpr_dataset",  # CVPR test dataset for JSON generation
    # Alternative datasets:
    # "rishuKumar404/train_cvpr_dataset"  # CVPR training dataset with 3 software types
    # "rishuKumar404/3dslicer-tabular-benchmark"  # Original 3DSlicer dataset
    "hf_token": os.environ.get("HF_TOKEN", ""),  # Set HF_TOKEN environment variable
    "dataset_cache_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"),
}

# Agent configuration
AGENT_CONFIG = {
    "target_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # Target agent model (Llama 4 Maverick)
    "critic_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # Critic agent model (Llama 4 Maverick)
    "max_tokens": 4096,                 # Maximum tokens for model response
    "temperature": 0.2,                 # Model temperature
    # Deep Infra API configuration
    "deepinfra_api_key": os.environ.get("DEEPINFRA_TOKEN", ""),  # Set DEEPINFRA_TOKEN environment variable
    "deepinfra_base_url": "https://api.deepinfra.com/v1/openai",  # Deep Infra base URL
    "deepinfra_api_key_env": "DEEPINFRA_TOKEN",  # Environment variable name for API key
}

# Decision thresholds
DECISION_THRESHOLDS = {
    "action_accept_threshold": 0.80,    # Threshold to accept an action
    "action_fail_threshold": 0.40,      # Threshold to fail an action
    "semantic_min": 0.60,               # Minimum semantic score for partial success
    "terminate_confidence": 0.90,       # Confidence threshold for task completion
    "grounding_confidence_trust": 0.80, # Threshold to trust grounding over model prediction
}

# Tool configuration
TOOL_CONFIG = {
    "visual_grounding_confidence": 0.70, # Confidence threshold for visual grounding
    "ocr_confidence": 0.60,             # Confidence threshold for OCR
    "medical_tool_confidence": 0.75,    # Confidence threshold for medical tools
}

# Memory configuration
MEMORY_CONFIG = {
    "short_term_memory_size": 5,        # Number of recent steps to keep in short-term memory
    "long_term_memory_max_tokens": 2048, # Maximum tokens for long-term memory
}

# Synthetic dataset creation
DATASET_CREATION_CONFIG = {
    "synthetic_positive_filter": 0.85,  # Minimum score to include in synthetic dataset
    "max_examples_per_task": 100,       # Maximum number of examples to generate per task
    "output_dir": "synthetic_dataset",  # Directory to save synthetic dataset
}

def get_config(config_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the configuration dictionary.
    
    Args:
        config_key: Optional key to get a specific part of the configuration.
                   If None, returns the full configuration.
    
    Returns:
        Configuration dictionary
    """
    full_config = {
        "system": SYSTEM_CONFIG,
        "dataset": DATASET_CONFIG,
        "agent": AGENT_CONFIG,
        "thresholds": DECISION_THRESHOLDS,
        "tool": TOOL_CONFIG,
        "memory": MEMORY_CONFIG,
        "dataset_creation": DATASET_CREATION_CONFIG,
    }
    
    if config_key is not None:
        if config_key not in full_config:
            raise ValueError(f"Unknown configuration key: {config_key}")
        return full_config[config_key]
    
    return full_config

def update_config(config_key: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration values.
    
    Args:
        config_key: Key of the configuration section to update
        updates: Dictionary of values to update
    """
    if config_key == "system":
        SYSTEM_CONFIG.update(updates)
    elif config_key == "dataset":
        DATASET_CONFIG.update(updates)
    elif config_key == "agent":
        AGENT_CONFIG.update(updates)
    elif config_key == "thresholds":
        DECISION_THRESHOLDS.update(updates)
    elif config_key == "tool":
        TOOL_CONFIG.update(updates)
    elif config_key == "memory":
        MEMORY_CONFIG.update(updates)
    elif config_key == "dataset_creation":
        DATASET_CREATION_CONFIG.update(updates)
    else:
        raise ValueError(f"Unknown configuration key: {config_key}")

# Environment-specific overrides (from environment variables)
def load_env_overrides() -> None:
    """Load configuration overrides from environment variables."""
    # Dataset configuration
    if os.environ.get("HF_DATASET_NAME"):
        DATASET_CONFIG["hf_dataset_name"] = os.environ.get("HF_DATASET_NAME")
    
    if os.environ.get("HF_TOKEN"):
        DATASET_CONFIG["hf_token"] = os.environ.get("HF_TOKEN")
    
    # Agent configuration
    if os.environ.get("TARGET_MODEL"):
        AGENT_CONFIG["target_model"] = os.environ.get("TARGET_MODEL")
    
    if os.environ.get("CRITIC_MODEL"):
        AGENT_CONFIG["critic_model"] = os.environ.get("CRITIC_MODEL")
    
    # Deep Infra API key - check environment variable first, then use default
    if os.environ.get(AGENT_CONFIG.get("deepinfra_api_key_env", "DEEPINFRA_TOKEN")):
        AGENT_CONFIG["deepinfra_api_key"] = os.environ.get(AGENT_CONFIG["deepinfra_api_key_env"])
    
    # System configuration
    if os.environ.get("DEBUG_MODE"):
        SYSTEM_CONFIG["debug_mode"] = os.environ.get("DEBUG_MODE").lower() in ("true", "1", "yes")
    
    if os.environ.get("VERBOSE"):
        SYSTEM_CONFIG["verbose"] = os.environ.get("VERBOSE").lower() in ("true", "1", "yes")

# Load environment overrides at import time
load_env_overrides()

if __name__ == "__main__":
    # Print the configuration when run directly
    import json
    print(json.dumps(get_config(), indent=2))
