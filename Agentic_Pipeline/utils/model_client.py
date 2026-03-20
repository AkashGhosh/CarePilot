#!/usr/bin/env python3
"""
Model client utility for the Medical Visual Agent system.
This module creates and manages API clients for model inference.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from config import get_config

logger = logging.getLogger(__name__)

def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Any:
    """
    Create an OpenAI-compatible client for Deep Infra.
    
    Args:
        api_key: API key for Deep Infra (if None, uses config)
        base_url: Base URL for Deep Infra API (if None, uses config)
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install it with: pip install openai")
    
    # Get configuration
    agent_config = get_config("agent")
    
    # Use provided values or fall back to config
    api_key = api_key or agent_config.get("deepinfra_api_key")
    base_url = base_url or agent_config.get("deepinfra_base_url")
    
    # Check environment variable if not provided
    if not api_key:
        api_key_env = agent_config.get("deepinfra_api_key_env", "DEEPINFRA_TOKEN")
        api_key = os.getenv(api_key_env, api_key)
    
    if not api_key:
        raise ValueError("Deep Infra API key not found. Set DEEPINFRA_TOKEN environment variable or configure in config.py")
    
    if not base_url:
        raise ValueError("Deep Infra base URL not configured")
    
    # Create OpenAI client configured for Deep Infra
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    logger.info(f"Created OpenAI client for Deep Infra (base_url: {base_url})")
    
    return client

def get_model_client() -> Any:
    """
    Get or create the default model client.
    
    Returns:
        OpenAI client instance for Deep Infra
    """
    return create_openai_client()

