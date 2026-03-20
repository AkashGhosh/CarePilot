#!/usr/bin/env python3
"""
Inference module for fine-tuned Llama-4-Maverick model.
Handles sequential step prediction with memory propagation.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
from tqdm import tqdm

# Optimize PyTorch for inference speed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Faster for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Faster (slight non-determinism acceptable for inference)

# Try to import Qwen VL generation classes (Qwen2, Qwen2.5, Qwen3)
QWEN2VL_AVAILABLE = False
QWEN2_5VL_AVAILABLE = False
QWEN3VL_AVAILABLE = False
Qwen2VLForConditionalGeneration = None
Qwen2_5VLForConditionalGeneration = None
Qwen3VLForConditionalGeneration = None

# Try different import patterns for Qwen2-VL
try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN2VL_AVAILABLE = True
except ImportError:
    try:
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
        QWEN2VL_AVAILABLE = True
    except ImportError:
        pass

# Try different import patterns for Qwen2.5-VL
try:
    from transformers import Qwen2_5VLForConditionalGeneration
    QWEN2_5VL_AVAILABLE = True
except ImportError:
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5VLForConditionalGeneration
        QWEN2_5VL_AVAILABLE = True
    except ImportError:
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForConditionalGeneration
            QWEN2_5VL_AVAILABLE = True
        except ImportError:
            pass

# Try different import patterns for Qwen3-VL
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3VL_AVAILABLE = True
except ImportError:
    try:
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        QWEN3VL_AVAILABLE = True
    except ImportError:
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            QWEN3VL_AVAILABLE = True
        except ImportError:
            pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from data_preprocessor import format_training_prompt, load_image, resolve_image_path

class StepPredictor:
    """
    Predictor for sequential step prediction with memory propagation.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to fine-tuned model (PEFT adapter or full model)
            base_model_name: Base model name if using PEFT adapter
            trust_remote_code: Whether to trust remote code
            hf_token: HuggingFace token for authentication
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.trust_remote_code = trust_remote_code
        
        # Get token from config if not provided
        if hf_token is None:
            config = get_config("model")
            hf_token = config.get("hf_token")
        
        self.hf_token = hf_token
        
        # Image cache for faster repeated image loading
        self._image_cache = {}
        
        print(f"Loading model from {model_path}...")
        self._load_model()
        
        # Memory state for sequential prediction
        self.current_task_id = None
        self.memory_state = {
            "short_term": {},
            "long_term": {}
        }
    
    def _load_model(self):
        """Load model and tokenizer/processor."""
        # Normalize model path first if it's a local path
        if not self.base_model_name:
            # Normalize the path early
            normalized_path = os.path.abspath(os.path.expanduser(self.model_path))
            if os.path.exists(normalized_path) or os.path.isdir(normalized_path):
                self.model_path = normalized_path
        
        # Determine model path for checking
        if self.base_model_name:
            check_model_name = self.base_model_name
            # Check if this is a Qwen2.5-VL model from name
            is_qwen_vl = "qwen" in check_model_name.lower() and ("vl" in check_model_name.lower() or "vision" in check_model_name.lower())
        else:
            # For local model paths, check the config file to determine model type
            check_model_name = self.model_path
            is_qwen_vl = False
            
            # Try to load config from model directory to determine type
            config_path = os.path.join(self.model_path, "config.json")
            print(f"Checking for config file at: {config_path}")
            
            if os.path.exists(config_path):
                try:
                    from transformers import AutoConfig
                    print(f"Loading config from {self.model_path}...")
                    config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
                    # Check config class name
                    config_class_name = config.__class__.__name__
                    print(f"Config class name: {config_class_name}")
                    
                    if "qwen" in config_class_name.lower() and ("vl" in config_class_name.lower() or "vision" in config_class_name.lower()):
                        is_qwen_vl = True
                        print(f"✓ Detected Qwen VL model from config: {config_class_name}")
                    else:
                        print(f"Config class {config_class_name} does not match Qwen VL pattern")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load config to detect model type: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: check path string
                    is_qwen_vl = "qwen" in check_model_name.lower() and ("vl" in check_model_name.lower() or "vision" in check_model_name.lower())
            else:
                print(f"⚠️  Config file not found at {config_path}. Trying fallback detection...")
                # Fallback: check path string
                is_qwen_vl = "qwen" in check_model_name.lower() and ("vl" in check_model_name.lower() or "vision" in check_model_name.lower())
        
        # Store this for use in predict_step
        self.is_qwen_vl = is_qwen_vl
        
        if is_qwen_vl:
            print("✓ Detected Qwen2.5-VL vision-language model")
        else:
            print("⚠️  Model type detection: Not detected as Qwen VL model")
        
        # Load tokenizer/processor
        if self.base_model_name:
            tokenizer_path = self.base_model_name
            token = self.hf_token  # Need token for base model
        else:
            tokenizer_path = self.model_path
            token = None  # Fine-tuned model shouldn't need token
            # Check if this is a local path (absolute path starting with / or exists locally)
            # Normalize the path to ensure it's treated as local
            normalized_path = os.path.abspath(os.path.expanduser(tokenizer_path))
            if os.path.isabs(tokenizer_path):
                # If it's an absolute path, check if it exists
                if os.path.exists(normalized_path) or os.path.isdir(normalized_path):
                    tokenizer_path = normalized_path
                    # Also ensure model_path is normalized for later use
                    self.model_path = tokenizer_path
                else:
                    # Path doesn't exist - provide helpful error
                    raise FileNotFoundError(
                        f"Model path does not exist: {tokenizer_path}\n"
                        f"Normalized path: {normalized_path}\n"
                        f"Please check that the path is correct. Current working directory: {os.getcwd()}"
                    )
            elif os.path.exists(tokenizer_path) or os.path.isdir(tokenizer_path):
                # Relative path that exists
                tokenizer_path = normalized_path
                self.model_path = tokenizer_path
        
        print(f"Loading tokenizer/processor from {tokenizer_path}...")
        if token:
            print("Using HuggingFace token for authentication...")
            try:
                from huggingface_hub import login as hf_login, whoami
                hf_login(token=token, add_to_git_credential=False)
                try:
                    user_info = whoami()
                    print(f"✓ Authenticated as: {user_info.get('name', 'Unknown user')}")
                except Exception:
                    pass
                print("✓ Authentication successful!")
            except Exception as e:
                print(f"⚠️  Warning: Could not login: {e}")
        
        if is_qwen_vl:
            print("Loading processor for Qwen2.5-VL model...")
            load_kwargs = {
                "trust_remote_code": self.trust_remote_code,
            }
            if token:
                load_kwargs["token"] = token
            
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_path,
                **load_kwargs
            )
            self.tokenizer = self.processor  # For compatibility
        else:
            load_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "padding_side": "right"
            }
            if token:
                load_kwargs["token"] = token
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                **load_kwargs
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model if PEFT
        if self.base_model_name:
            print(f"Loading base model: {self.base_model_name}...")
            if self.hf_token:
                print("Using HuggingFace token for base model authentication...")
            
            if is_qwen_vl:
                # Determine which Qwen VL model we're using
                is_qwen3_vl = "qwen3" in self.base_model_name.lower() or ("3" in self.base_model_name.lower() and "2.5" not in self.base_model_name.lower() and "2_5" not in self.base_model_name.lower())
                is_qwen2_5_vl = "2.5" in self.base_model_name or "2_5" in self.base_model_name
                
                # Try Qwen3-VL first
                if is_qwen3_vl:
                    # Safely get Qwen3VLForConditionalGeneration from module or import it
                    qwen3_model_class = globals().get('Qwen3VLForConditionalGeneration')
                    if qwen3_model_class is None and QWEN3VL_AVAILABLE:
                        # Try to get from module-level import
                        import sys
                        qwen3_model_class = sys.modules[__name__].__dict__.get('Qwen3VLForConditionalGeneration')
                    
                    if qwen3_model_class is not None:
                        print("Using Qwen3VLForConditionalGeneration for Qwen3-VL model...")
                        model_config = get_config("model")
                        load_in_4bit = model_config.get("load_in_4bit", True)
                        load_in_8bit = model_config.get("load_in_8bit", False)
                        
                        # Use BitsAndBytesConfig for Qwen3-VL (doesn't accept direct load_in_4bit/8bit)
                        from transformers import BitsAndBytesConfig
                        quantization_config = None
                        if load_in_4bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                        elif load_in_8bit:
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        
                        base_model = qwen3_model_class.from_pretrained(
                            self.base_model_name,
                            token=self.hf_token,
                            trust_remote_code=self.trust_remote_code,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True,
                            max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                        )
                    else:
                        print("Qwen3VLForConditionalGeneration not directly available, trying direct import...")
                        try:
                            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                            print("Successfully imported Qwen3VLForConditionalGeneration from modeling module")
                            model_config = get_config("model")
                            load_in_4bit = model_config.get("load_in_4bit", True)
                            load_in_8bit = model_config.get("load_in_8bit", False)
                            
                            # Use BitsAndBytesConfig for Qwen3-VL
                            from transformers import BitsAndBytesConfig
                            quantization_config = None
                            if load_in_4bit:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4"
                                )
                            elif load_in_8bit:
                                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            
                            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                                self.base_model_name,
                                token=self.hf_token,
                                trust_remote_code=self.trust_remote_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto",
                                quantization_config=quantization_config,
                                low_cpu_mem_usage=True,
                                max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                            )
                        except ImportError as e:
                            print(f"Failed to import Qwen3VLForConditionalGeneration: {e}")
                            print("Please ensure transformers>=4.57.0 is installed with Qwen3-VL support.")
                            print("Try: pip install git+https://github.com/huggingface/transformers")
                            raise
                # Then Qwen2.5-VL
                elif is_qwen2_5_vl:
                    if QWEN2_5VL_AVAILABLE:
                        print("Using Qwen2_5VLForConditionalGeneration for Qwen2.5-VL model...")
                        # Get model config for memory optimizations
                        model_config = get_config("model")
                        load_in_4bit = model_config.get("load_in_4bit", True)
                        load_in_8bit = model_config.get("load_in_8bit", False)
                        
                        base_model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                            self.base_model_name,
                            token=self.hf_token,
                            trust_remote_code=self.trust_remote_code,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            load_in_8bit=load_in_8bit,
                            load_in_4bit=load_in_4bit,
                            low_cpu_mem_usage=True,  # OOM prevention
                            max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                        )
                    else:
                        print("Qwen2_5VLForConditionalGeneration not directly available, trying direct import...")
                        try:
                            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForConditionalGeneration
                            print("Successfully imported Qwen2_5VLForConditionalGeneration from modeling module")
                            # Get model config for memory optimizations
                            model_config = get_config("model")
                            load_in_4bit = model_config.get("load_in_4bit", True)
                            load_in_8bit = model_config.get("load_in_8bit", False)
                            
                            base_model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                                self.base_model_name,
                                token=self.hf_token,
                                trust_remote_code=self.trust_remote_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto",
                                load_in_8bit=load_in_8bit,
                                load_in_4bit=load_in_4bit,
                                low_cpu_mem_usage=True,  # OOM prevention
                                max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                            )
                        except ImportError as e:
                            # Fallback: Use AutoConfig to find the correct architecture name
                            print("Direct import failed, using AutoConfig to find correct class...")
                            try:
                                from transformers import AutoConfig
                                import importlib
                                
                                # Load config to get architecture name
                                config = AutoConfig.from_pretrained(
                                    self.base_model_name,
                                    token=self.hf_token,
                                    trust_remote_code=self.trust_remote_code
                                )
                                
                                # Get architecture name from config
                                if hasattr(config, 'architectures') and config.architectures:
                                    arch_name = config.architectures[0]
                                    print(f"Found model architecture: {arch_name}")
                                    
                                    # Try to dynamically import the class
                                    try:
                                        module_name = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
                                        module = importlib.import_module(module_name)
                                        model_class = getattr(module, arch_name, None)
                                        
                                        if model_class:
                                            print(f"Successfully loaded class: {arch_name}")
                                            # Get model config for memory optimizations
                                            model_config = get_config("model")
                                            load_in_4bit = model_config.get("load_in_4bit", True)
                                            load_in_8bit = model_config.get("load_in_8bit", False)
                                            
                                            base_model = model_class.from_pretrained(
                                                self.base_model_name,
                                                token=self.hf_token,
                                                trust_remote_code=self.trust_remote_code,
                                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                device_map="auto",
                                                load_in_8bit=load_in_8bit,
                                                load_in_4bit=load_in_4bit,
                                                low_cpu_mem_usage=True,  # OOM prevention
                                                max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                                            )
                                        else:
                                            raise ImportError(f"Class {arch_name} not found in {module_name}")
                                    except Exception as e2:
                                        print(f"Dynamic import failed: {e2}")
                                        # Last resort: try AutoModel with trust_remote_code
                                        print("Trying AutoModel with trust_remote_code as last resort...")
                                        from transformers import AutoModel
                                        # Get model config for memory optimizations
                                        model_config = get_config("model")
                                        load_in_4bit = model_config.get("load_in_4bit", True)
                                        load_in_8bit = model_config.get("load_in_8bit", False)
                                        
                                        base_model = AutoModel.from_pretrained(
                                            self.base_model_name,
                                            token=self.hf_token,
                                            trust_remote_code=True,
                                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                            device_map="auto",
                                            load_in_8bit=load_in_8bit,
                                            load_in_4bit=load_in_4bit,
                                            low_cpu_mem_usage=True,  # OOM prevention
                                            max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                                        )
                                        print(f"⚠️  WARNING: Loaded model using AutoModel fallback. Model type: {type(base_model)}")
                                else:
                                    raise ValueError("Config does not specify model architecture")
                            except Exception as e3:
                                raise ImportError(
                                    f"Failed to load Qwen2.5-VL model: {e3}. "
                                    f"Please ensure transformers is up to date: pip install --upgrade transformers"
                                )
                else:
                    if QWEN2VL_AVAILABLE:
                        print("Using Qwen2VLForConditionalGeneration for Qwen2-VL model...")
                        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.base_model_name,
                            token=self.hf_token,
                            trust_remote_code=self.trust_remote_code,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                        )
                    else:
                        raise ImportError("Qwen2VLForConditionalGeneration not available. Please update transformers: pip install --upgrade transformers")
            else:
                print("Using AutoModelForCausalLM for causal LM model...")
                # Get model config for memory optimizations
                model_config = get_config("model")
                load_in_4bit = model_config.get("load_in_4bit", True)
                load_in_8bit = model_config.get("load_in_8bit", False)
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    token=self.hf_token,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    low_cpu_mem_usage=True,  # OOM prevention
                    max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                )
            
            # Load PEFT adapter
            print(f"Loading PEFT adapter from {self.model_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # CRITICAL FIX: Ensure the PEFT model has generate method
            if not hasattr(self.model, 'generate'):
                print("⚠️  PEFT model missing 'generate' method. Binding from base_model...")
                import types
                if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'generate'):
                    self.model.generate = types.MethodType(self.model.base_model.generate.__func__, self.model)
                    print("✓ Bound 'generate' method to PEFT model")
                else:
                    raise AttributeError(f"Base model missing 'generate'. Base type: {type(base_model)}")
            else:
                print("✓ PEFT model has 'generate' method")
        else:
            # Check if this is a PEFT adapter (has adapter_model.safetensors or adapter_model.bin)
            adapter_path = os.path.join(self.model_path, "adapter_model.safetensors")
            adapter_bin_path = os.path.join(self.model_path, "adapter_model.bin")
            is_peft_adapter = os.path.exists(adapter_path) or os.path.exists(adapter_bin_path)
            
            if is_peft_adapter:
                print(f"Detected PEFT adapter at {self.model_path}. Loading as adapter...")
                # Load base model first
                # Get base model name from adapter_config.json or try to detect from config
                adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
                base_model_name_from_config = None
                
                if os.path.exists(adapter_config_path):
                    try:
                        import json
                        with open(adapter_config_path, 'r') as f:
                            adapter_config = json.load(f)
                            base_model_name_from_config = adapter_config.get("base_model_name_or_path")
                            if base_model_name_from_config:
                                print(f"Found base model name in adapter config: {base_model_name_from_config}")
                                # Override with detected base model
                                self.base_model_name = base_model_name_from_config
                                # Re-detect model type from base model name
                                check_model_name = self.base_model_name
                                is_qwen_vl = "qwen" in check_model_name.lower() and ("vl" in check_model_name.lower() or "vision" in check_model_name.lower())
                                print(f"Re-detected model type from base model: is_qwen_vl={is_qwen_vl}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read adapter config: {e}")
                
                # Load base model if we have the name
                if self.base_model_name or base_model_name_from_config:
                    base_model_to_use = self.base_model_name or base_model_name_from_config
                    print(f"Loading base model: {base_model_to_use}...")
                    
                    if is_qwen_vl:
                        # Determine which Qwen VL model we're using
                        is_qwen3_vl = "qwen3" in base_model_to_use.lower() or ("3" in base_model_to_use.lower() and "2.5" not in base_model_to_use.lower() and "2_5" not in base_model_to_use.lower())
                        is_qwen2_5_vl = "2.5" in base_model_to_use or "2_5" in base_model_to_use
                        
                        # Get model config for memory optimizations
                        model_config = get_config("model")
                        
                        # Try Qwen3-VL first
                        if is_qwen3_vl:
                            # Safely get Qwen3VLForConditionalGeneration from module or import it
                            qwen3_model_class = globals().get('Qwen3VLForConditionalGeneration')
                            if qwen3_model_class is None and QWEN3VL_AVAILABLE:
                                # Try to get from module-level import
                                import sys
                                qwen3_model_class = sys.modules[__name__].__dict__.get('Qwen3VLForConditionalGeneration')
                            
                            if qwen3_model_class is not None:
                                print("Using Qwen3VLForConditionalGeneration for Qwen3-VL model...")
                                # Use BitsAndBytesConfig for Qwen3-VL (doesn't accept direct load_in_4bit/8bit)
                                from transformers import BitsAndBytesConfig
                                quantization_config = None
                                if model_config.get("load_in_4bit", True):
                                    quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4"
                                    )
                                elif model_config.get("load_in_8bit", False):
                                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                                
                                base_model = qwen3_model_class.from_pretrained(
                                    base_model_to_use,
                                    token=self.hf_token,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                    quantization_config=quantization_config,
                                    low_cpu_mem_usage=True,
                                    max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                                )
                            else:
                                print("Qwen3VLForConditionalGeneration not available, trying direct import...")
                                try:
                                    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                                    # Use BitsAndBytesConfig for Qwen3-VL
                                    from transformers import BitsAndBytesConfig
                                    quantization_config = None
                                    if model_config.get("load_in_4bit", True):
                                        quantization_config = BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4"
                                        )
                                    elif model_config.get("load_in_8bit", False):
                                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                                    
                                    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                                        base_model_to_use,
                                        token=self.hf_token,
                                        trust_remote_code=self.trust_remote_code,
                                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                        device_map="auto",
                                        quantization_config=quantization_config,
                                        low_cpu_mem_usage=True,
                                        max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                                    )
                                except ImportError as e:
                                    print(f"Failed to import Qwen3VLForConditionalGeneration: {e}")
                                    print("Please ensure transformers>=4.57.0 is installed with Qwen3-VL support.")
                                    raise
                        # Then Qwen2.5-VL
                        elif is_qwen2_5_vl:
                            if QWEN2_5VL_AVAILABLE:
                                base_model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                                    base_model_to_use,
                                    token=self.hf_token,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                )
                            else:
                                # Fallback: Use AutoConfig to find the correct architecture name
                                print("Direct import failed, using AutoConfig to find correct class...")
                                try:
                                    from transformers import AutoConfig
                                    import importlib
                                    
                                    # Load config to get architecture name
                                    config = AutoConfig.from_pretrained(
                                        base_model_to_use,
                                        token=self.hf_token,
                                        trust_remote_code=self.trust_remote_code
                                    )
                                    
                                    # Get architecture name from config
                                    if hasattr(config, 'architectures') and config.architectures:
                                        arch_name = config.architectures[0]
                                        print(f"Found model architecture: {arch_name}")
                                        
                                        # Try to dynamically import the class
                                        try:
                                            module_name = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
                                            module = importlib.import_module(module_name)
                                            model_class = getattr(module, arch_name, None)
                                            
                                            if model_class:
                                                print(f"Successfully loaded class: {arch_name}")
                                                base_model = model_class.from_pretrained(
                                                    base_model_to_use,
                                                    token=self.hf_token,
                                                    trust_remote_code=self.trust_remote_code,
                                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                    device_map="auto",
                                                )
                                            else:
                                                raise ImportError(f"Class {arch_name} not found in {module_name}")
                                        except Exception as e2:
                                            print(f"Dynamic import failed: {e2}")
                                            # Last resort: try AutoModel with trust_remote_code
                                            print("Trying AutoModel with trust_remote_code as last resort...")
                                            from transformers import AutoModel
                                            # Get model config for memory optimizations
                                            model_config = get_config("model")
                                            load_in_4bit = model_config.get("load_in_4bit", True)
                                            load_in_8bit = model_config.get("load_in_8bit", False)
                                            
                                            base_model = AutoModel.from_pretrained(
                                                base_model_to_use,
                                                token=self.hf_token,
                                                trust_remote_code=True,
                                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                device_map="auto",
                                                load_in_8bit=load_in_8bit,
                                                load_in_4bit=load_in_4bit,
                                                low_cpu_mem_usage=True,  # OOM prevention
                                                max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                                            )
                                            print(f"⚠️  WARNING: Loaded model using AutoModel fallback. Model type: {type(base_model)}")
                                    else:
                                        raise ValueError("Config does not specify model architecture")
                                except Exception as e3:
                                    raise ImportError(
                                        f"Failed to load Qwen2.5-VL model: {e3}. "
                                        f"Please ensure transformers is up to date: pip install --upgrade transformers"
                                    )
                        else:
                            if QWEN2VL_AVAILABLE:
                                base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                                    base_model_to_use,
                                    token=self.hf_token,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                )
                            else:
                                raise ImportError("Qwen2VLForConditionalGeneration not available")
                    else:
                        # Get model config for memory optimizations
                        model_config = get_config("model")
                        load_in_4bit = model_config.get("load_in_4bit", True)
                        load_in_8bit = model_config.get("load_in_8bit", False)
                        
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_to_use,
                            token=self.hf_token,
                            trust_remote_code=self.trust_remote_code,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            load_in_8bit=load_in_8bit,
                            load_in_4bit=load_in_4bit,
                            low_cpu_mem_usage=True,  # OOM prevention
                            max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                        )
                    
                    # CRITICAL: Verify the base model has 'generate' method and is a generation class
                    model_class_name = type(base_model).__name__
                    print(f"Loaded model class: {model_class_name}")
                    
                    # Check if it's a generation class (should have 'ForConditionalGeneration' or 'ForCausalLM' in name)
                    is_generation_class = (
                        'ForConditionalGeneration' in model_class_name or 
                        'ForCausalLM' in model_class_name or
                        'ForGeneration' in model_class_name
                    )
                    
                    if not is_generation_class:
                        print(f"⚠️  WARNING: Model class {model_class_name} doesn't look like a generation class!")
                        print(f"Expected class name to contain 'ForConditionalGeneration' or 'ForCausalLM'")
                    
                    if not hasattr(base_model, 'generate'):
                        print(f"❌ ERROR: Loaded model class {model_class_name} does not have 'generate' method!")
                        print(f"This happens when the base model class is loaded instead of the generation class.")
                        raise ValueError(
                            f"Model loaded as {model_class_name} which doesn't support generation. "
                            f"Expected Qwen2_5VLForConditionalGeneration or similar generation class. "
                            f"Model was loaded from: {base_model_to_use}"
                        )
                    else:
                        print(f"✓ Base model class {model_class_name} has 'generate' method")
                        
                        # Test that generate method is actually callable
                        try:
                            # Just check if it's callable, don't actually call it
                            if not callable(base_model.generate):
                                raise ValueError("generate attribute exists but is not callable")
                            print(f"✓ 'generate' method is callable")
                        except Exception as e:
                            raise ValueError(f"'generate' method exists but is not usable: {e}")
                    
                    # Ensure base model has prepare_inputs_for_generation for PEFT compatibility
                    if not hasattr(base_model, 'prepare_inputs_for_generation') or not callable(getattr(base_model, 'prepare_inputs_for_generation', None)):
                        print("Base model missing prepare_inputs_for_generation, searching parent classes...")
                        import types
                        method_found = False
                        # Check parent classes for the method
                        for cls in type(base_model).__mro__:
                            if hasattr(cls, 'prepare_inputs_for_generation'):
                                print(f"Found prepare_inputs_for_generation in parent class: {cls.__name__}")
                                # Bind the method to the instance
                                method = getattr(cls, 'prepare_inputs_for_generation')
                                if callable(method):
                                    base_model.prepare_inputs_for_generation = types.MethodType(method, base_model)
                                    method_found = True
                                    print("✓ Successfully bound prepare_inputs_for_generation to base model")
                                    break
                        
                        # If still not found, check if the model wraps another model
                        if not method_found and hasattr(base_model, 'model') and hasattr(base_model.model, 'prepare_inputs_for_generation'):
                            print("Binding prepare_inputs_for_generation from model.model...")
                            method = base_model.model.prepare_inputs_for_generation
                            if callable(method):
                                base_model.prepare_inputs_for_generation = types.MethodType(method.__func__, base_model)
                                method_found = True
                                print("✓ Successfully bound prepare_inputs_for_generation from model.model")
                        
                        if not method_found:
                            print("⚠️  Warning: Could not find prepare_inputs_for_generation. Trying GenerationMixin...")
                            # Last resort: try to get it from GenerationMixin
                            try:
                                from transformers.generation.utils import GenerationMixin
                                if hasattr(GenerationMixin, 'prepare_inputs_for_generation'):
                                    method = GenerationMixin.prepare_inputs_for_generation
                                    base_model.prepare_inputs_for_generation = types.MethodType(method, base_model)
                                    print("✓ Bound prepare_inputs_for_generation from GenerationMixin")
                            except Exception as e:
                                print(f"⚠️  Warning: Could not bind from GenerationMixin: {e}")
                    else:
                        print("✓ Base model has prepare_inputs_for_generation")
                    
                    # Load PEFT adapter
                    print(f"Loading PEFT adapter from {self.model_path}...")
                    
                    # Verify prepare_inputs_for_generation exists and is callable before PEFT loading
                    if not hasattr(base_model, 'prepare_inputs_for_generation') or not callable(base_model.prepare_inputs_for_generation):
                        print("⚠️  CRITICAL: Base model still missing prepare_inputs_for_generation after binding attempt!")
                        print(f"Model type: {type(base_model)}")
                        print(f"Has attr: {hasattr(base_model, 'prepare_inputs_for_generation')}")
                        if hasattr(base_model, 'prepare_inputs_for_generation'):
                            print(f"Method is callable: {callable(base_model.prepare_inputs_for_generation)}")
                        # Last resort: try to get it from GenerationMixin
                        try:
                            from transformers.generation.utils import GenerationMixin
                            import types
                            if hasattr(GenerationMixin, 'prepare_inputs_for_generation'):
                                method = GenerationMixin.prepare_inputs_for_generation
                                base_model.prepare_inputs_for_generation = types.MethodType(method, base_model)
                                print("✓ Manually bound prepare_inputs_for_generation from GenerationMixin")
                        except Exception as e:
                            print(f"Failed to bind from GenerationMixin: {e}")
                            raise RuntimeError(
                                "Cannot load PEFT adapter: base model must have prepare_inputs_for_generation. "
                                "This usually means the model was not loaded as a generation class. "
                                f"Model type: {type(base_model)}"
                            )
                    
                    # Final verification: ensure the method exists and is accessible
                    if hasattr(base_model, 'prepare_inputs_for_generation'):
                        try:
                            # Test if method is accessible
                            _ = base_model.prepare_inputs_for_generation
                            print(f"✓ Verified prepare_inputs_for_generation is accessible")
                        except Exception as e:
                            print(f"⚠️  Warning: prepare_inputs_for_generation exists but not accessible: {e}")
                    
                    try:
                        self.model = PeftModel.from_pretrained(base_model, self.model_path)
                        print("✓ PEFT adapter loaded successfully")
                        
                        # CRITICAL FIX: Ensure the PEFT model has generate method
                        # PEFT wraps the model, but sometimes it accesses base_model.model which is the inner model
                        if not hasattr(self.model, 'generate'):
                            print("⚠️  PEFT model missing 'generate' method. Checking base_model...")
                            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'generate'):
                                print("✓ Found 'generate' in base_model, binding to PEFT model...")
                                import types
                                self.model.generate = types.MethodType(self.model.base_model.generate.__func__, self.model)
                            elif hasattr(self.model, 'get_base_model') and hasattr(self.model.get_base_model(), 'generate'):
                                base = self.model.get_base_model()
                                import types
                                self.model.generate = types.MethodType(base.generate.__func__, self.model)
                            else:
                                raise AttributeError(
                                    "PEFT model and base model both missing 'generate' method. "
                                    f"Base model type: {type(base_model)}"
                                )
                        else:
                            print("✓ PEFT model has 'generate' method")
                    except AttributeError as e:
                        if "prepare_inputs_for_generation" in str(e):
                            print(f"❌ PEFT loading failed: {e}")
                            print(f"Base model type: {type(base_model)}")
                            print(f"Model has prepare_inputs_for_generation: {hasattr(base_model, 'prepare_inputs_for_generation')}")
                            # Try one more time with explicit method access
                            if hasattr(base_model, 'model') and hasattr(base_model.model, 'prepare_inputs_for_generation'):
                                print("Attempting to bind from nested model...")
                                import types
                                base_model.prepare_inputs_for_generation = base_model.model.prepare_inputs_for_generation
                                try:
                                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
                                    print("✓ PEFT adapter loaded after nested model binding")
                                except Exception as e2:
                                    raise RuntimeError(
                                        f"Failed to load PEFT adapter even after binding method. "
                                        f"Error: {e2}"
                                    ) from e
                            else:
                                raise RuntimeError(
                                    f"PEFT requires prepare_inputs_for_generation but base model doesn't have it. "
                                    f"This may indicate the model was loaded incorrectly. "
                                    f"Model type: {type(base_model)}, Error: {e}"
                                ) from e
                        else:
                            raise
                else:
                    raise ValueError(
                        f"PEFT adapter detected at {self.model_path}, but no base model name found. "
                        f"Please provide --base_model_name argument or ensure adapter_config.json contains base_model_name_or_path."
                    )
            else:
                # Load full model
                print(f"Loading full model from {self.model_path}...")
                if is_qwen_vl:
                    # Determine which Qwen VL model we're using
                    # Check config if available, otherwise check path
                    is_qwen3_vl = False
                    is_qwen2_5_vl = False
                    config_path = os.path.join(self.model_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            from transformers import AutoConfig
                            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
                            config_class_name = config.__class__.__name__
                            print(f"Checking Qwen version from config class: {config_class_name}")
                            is_qwen3_vl = "qwen3" in config_class_name.lower() or ("3" in config_class_name.lower() and "2.5" not in config_class_name.lower() and "2_5" not in config_class_name.lower())
                            is_qwen2_5_vl = "2.5" in config_class_name or "2_5" in config_class_name
                            if is_qwen3_vl:
                                print(f"✓ Detected Qwen3-VL from config")
                            elif is_qwen2_5_vl:
                                print(f"✓ Detected Qwen2.5-VL from config")
                            else:
                                print(f"Detected Qwen2-VL (not 2.5 or 3) from config")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not load config to determine Qwen version: {e}")
                            is_qwen3_vl = "qwen3" in self.model_path.lower() or ("3" in self.model_path.lower() and "2.5" not in self.model_path.lower() and "2_5" not in self.model_path.lower())
                            is_qwen2_5_vl = "2.5" in self.model_path or "2_5" in self.model_path
                    else:
                        print(f"⚠️  Config file not found at {config_path}, checking path string...")
                        is_qwen3_vl = "qwen3" in self.model_path.lower() or ("3" in self.model_path.lower() and "2.5" not in self.model_path.lower() and "2_5" not in self.model_path.lower())
                        is_qwen2_5_vl = "2.5" in self.model_path or "2_5" in self.model_path
                
                    # Try Qwen3-VL first
                    if is_qwen3_vl:
                        # Safely get Qwen3VLForConditionalGeneration from module or import it
                        qwen3_model_class = globals().get('Qwen3VLForConditionalGeneration')
                        if qwen3_model_class is None and QWEN3VL_AVAILABLE:
                            # Try to get from module-level import
                            import sys
                            qwen3_model_class = sys.modules[__name__].__dict__.get('Qwen3VLForConditionalGeneration')
                        
                        if qwen3_model_class is not None:
                            print("Using Qwen3VLForConditionalGeneration for Qwen3-VL model...")
                            # Use BitsAndBytesConfig for Qwen3-VL (doesn't accept direct load_in_4bit/8bit)
                            from transformers import BitsAndBytesConfig
                            quantization_config = None
                            if model_config.get("load_in_4bit", True):
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4"
                                )
                            elif model_config.get("load_in_8bit", False):
                                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            
                            self.model = qwen3_model_class.from_pretrained(
                                self.model_path,
                                token=token,
                                trust_remote_code=self.trust_remote_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto",
                                quantization_config=quantization_config,
                                low_cpu_mem_usage=True,
                                max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                            )
                        else:
                            print("Qwen3VLForConditionalGeneration not directly available, trying direct import...")
                            try:
                                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                                print("Successfully imported Qwen3VLForConditionalGeneration from modeling module")
                                # Use BitsAndBytesConfig for Qwen3-VL
                                from transformers import BitsAndBytesConfig
                                quantization_config = None
                                if model_config.get("load_in_4bit", True):
                                    quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4"
                                    )
                                elif model_config.get("load_in_8bit", False):
                                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                                
                                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                                    self.model_path,
                                    token=token,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                    quantization_config=quantization_config,
                                    low_cpu_mem_usage=True,
                                    max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,
                                )
                            except ImportError as e:
                                print(f"Failed to import Qwen3VLForConditionalGeneration: {e}")
                                print("Please ensure transformers>=4.57.0 is installed with Qwen3-VL support.")
                                print("Try: pip install git+https://github.com/huggingface/transformers")
                                raise
                    # Then Qwen2.5-VL
                    elif is_qwen2_5_vl:
                        if QWEN2_5VL_AVAILABLE:
                            print("Using Qwen2_5VLForConditionalGeneration for Qwen2.5-VL model...")
                            self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                                self.model_path,
                                token=token,
                                trust_remote_code=self.trust_remote_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto",
                            )
                        else:
                            print("Qwen2_5VLForConditionalGeneration not directly available, trying direct import...")
                            try:
                                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForConditionalGeneration
                                print("Successfully imported Qwen2_5VLForConditionalGeneration from modeling module")
                                self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                                    self.model_path,
                                    token=token,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                )
                            except ImportError as e:
                                raise ImportError(
                                    f"Failed to import Qwen2_5VLForConditionalGeneration: {e}. "
                                    f"Please ensure transformers is up to date: pip install --upgrade transformers"
                                )
                    else:
                        if QWEN2VL_AVAILABLE:
                            print("Using Qwen2VLForConditionalGeneration for Qwen2-VL model...")
                            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                                self.model_path,
                                token=token,
                                trust_remote_code=self.trust_remote_code,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto",
                            )
                        else:
                            raise ImportError("Qwen2VLForConditionalGeneration not available. Please update transformers: pip install --upgrade transformers")
                else:
                    print("Using AutoModelForCausalLM for causal LM model...")
                    # Get model config for memory optimizations
                    model_config = get_config("model")
                    load_in_4bit = model_config.get("load_in_4bit", True)
                    load_in_8bit = model_config.get("load_in_8bit", False)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        token=token,
                        trust_remote_code=self.trust_remote_code,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        low_cpu_mem_usage=True,  # OOM prevention
                        max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                    )
        
        self.model.eval()
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("Compiling model for faster inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Model compilation not available or failed: {e}")
            print("   Continuing without compilation (slightly slower)")
        
        print("Model loaded successfully")
        print(f"Using GPU: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            # Clear CUDA cache after model loading
            torch.cuda.empty_cache()
            import gc
            gc.collect()  # Force garbage collection
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    def reset_memory(self, task_id: str = None):
        """
        Reset memory state for new task.
        
        Args:
            task_id: New task ID
        """
        self.current_task_id = task_id
        self.memory_state = {
            "short_term": {},
            "long_term": {}
        }
    
    def extract_memory_from_output(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Short Term and Long Term memory from output JSON.
        
        Respects ablation config: if long_term_memory is disabled, it won't be extracted.
        
        Args:
            output_json: Output JSON dictionary
            
        Returns:
            Dictionary with short_term and long_term memory
        """
        memory_config = get_config("memory")
        ablation_config = get_config("ablation")
        
        short_term = output_json.get(memory_config["short_term_key"], {})
        long_term = output_json.get(memory_config["long_term_key"], {})
        
        # Respect ablation config: if LTM is disabled, don't extract it
        if not ablation_config.get("use_long_term_memory", True):
            long_term = {}
        
        # Respect ablation config: if STM is disabled, don't extract it
        if not ablation_config.get("use_short_term_memory", True):
            short_term = {}
        
        return {
            "short_term": short_term if short_term else {},
            "long_term": long_term if long_term else {}
        }
    
    def predict_step(
        self,
        task_id: str,
        slide_number: str,
        grounding: str,
        image_id: Optional[str] = None,
        image_base_path: Optional[str] = None,
        short_term_memory: Optional[str] = None,
        long_term_memory: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict output for a single step.
        
        IMPORTANT: During test/evaluation, short_term_memory and long_term_memory
        should be None - memory comes from previous step predictions (self.memory_state),
        NOT from test CSV. Using memory from test CSV would cause data leakage.
        
        Args:
            task_id: Task identifier
            slide_number: Step number
            grounding: Grounding JSON string
            image_id: Optional image path
            image_base_path: Base path for images
            short_term_memory: Optional short term memory JSON string (IGNORED during test)
            long_term_memory: Optional long term memory JSON string (IGNORED during test)
            inference_config: Inference configuration
            
        Returns:
            Dictionary with predicted output and extracted memory
        """
        # CRITICAL: Always use memory from state (from previous predictions)
        # Never use provided memory during test/evaluation to prevent data leakage
        # Memory should come from model's own predictions, not from ground truth
        # Try using orjson for faster JSON serialization if available
        try:
            import orjson
            use_orjson_memory = True
        except ImportError:
            use_orjson_memory = False
        
        # Get ablation config BEFORE extracting memory (to respect ablation settings)
        ablation_config = get_config("ablation")
        
        # Respect ablation config: only extract memory if it's enabled
        use_stm = ablation_config.get("use_short_term_memory", True)
        use_ltm = ablation_config.get("use_long_term_memory", True)
        
        if use_orjson_memory:
            short_term_memory = orjson.dumps(self.memory_state["short_term"]).decode('utf-8') if (use_stm and self.memory_state["short_term"]) else "{}"
            long_term_memory = orjson.dumps(self.memory_state["long_term"]).decode('utf-8') if (use_ltm and self.memory_state["long_term"]) else "{}"
        else:
            short_term_memory = json.dumps(self.memory_state["short_term"], ensure_ascii=False) if (use_stm and self.memory_state["short_term"]) else "{}"
            long_term_memory = json.dumps(self.memory_state["long_term"], ensure_ascii=False) if (use_ltm and self.memory_state["long_term"]) else "{}"
        
        # Get inference config
        if inference_config is None:
            inference_config = get_config("inference")
        
        # Handle vision-language models (Qwen2.5-VL) vs text-only models
        if self.is_qwen_vl:
            # For vision-language models, use processor with images and chat template
            # Format prompt text
            # Check if function accepts ablation_config (handle cached bytecode issues)
            import inspect
            sig = inspect.signature(format_training_prompt)
            if 'ablation_config' in sig.parameters:
                prompt_text = format_training_prompt(
                    task_id=task_id,
                    slide_number=slide_number,
                    grounding=grounding or "{}",
                    short_term_memory=short_term_memory or "{}",
                    long_term_memory=long_term_memory or "{}",
                    ablation_config=ablation_config
                )
            else:
                # Fallback: function doesn't have ablation_config parameter yet
                prompt_text = format_training_prompt(
                    task_id=task_id,
                    slide_number=slide_number,
                    grounding=grounding or "{}",
                    short_term_memory=short_term_memory or "{}",
                    long_term_memory=long_term_memory or "{}"
                )
            
            # Load image if provided (with caching for speed)
            image = None
            if image_id:
                image_path = resolve_image_path(image_id, image_base_path)
                if image_path and image_path.exists():
                    # Use cached image if available (faster)
                    cache_key = str(image_path)
                    if cache_key in self._image_cache:
                        image = self._image_cache[cache_key]
                    else:
                        image = load_image(image_path)
                        if image is not None:
                            # Cache the image (limit cache size to prevent OOM)
                            if len(self._image_cache) < 100:  # Cache up to 100 images
                                self._image_cache[cache_key] = image
                        else:
                            print(f"Warning: Could not load image from {image_path}")
            
            # Use chat template for Qwen2.5-VL
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},  # Image placeholder
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
            
            # Apply chat template and process with processor
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Add generation prompt for inference
            )
            
            # Process with processor (handles both text and images)
            if image is not None:
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                )
            
            # Ensure all inputs are on the correct device (GPU)
            device = next(self.model.parameters()).device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            # Verify GPU usage
            if torch.cuda.is_available() and device.type == "cuda":
                pass  # Good, using GPU
            else:
                print(f"⚠️  WARNING: Model is on {device}, not GPU! This will be very slow.")
            
            # Get generation parameters
            max_new_tokens = inference_config.get("max_new_tokens", 512)
            do_sample = inference_config.get("do_sample", False)
            temperature = inference_config.get("temperature", 0.2) if do_sample else None
            top_p = inference_config.get("top_p", 0.9) if do_sample else None
            
            # Get tokenizer for pad/eos tokens
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.tokenizer
            
            # Prepare generation kwargs (optimized for speed)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": inference_config.get("num_return_sequences", 1),
                "num_beams": 1,  # Explicitly set for greedy decoding (faster)
                "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else None,
                "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None,
                "use_cache": inference_config.get("use_cache", True),  # Enable KV cache for speed
                "output_scores": False,  # Don't compute scores (faster)
                "return_dict_in_generate": False,  # Don't return dict (faster)
                "early_stopping": inference_config.get("early_stopping", True),  # Stop early if EOS found (faster)
            }
            
            # Add repetition penalty if specified (default 1.0 = no penalty)
            if "repetition_penalty" in inference_config:
                generation_kwargs["repetition_penalty"] = inference_config["repetition_penalty"]
            
            # Add sampling parameters only if sampling is enabled
            if do_sample:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            
            # Generate with optimizations
            # Use inference_mode for faster inference (slightly faster than no_grad)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            
            # Decode using processor tokenizer (skip special tokens for faster decoding)
            skip_special = inference_config.get("skip_special_tokens", True)
            if hasattr(self.processor, 'tokenizer'):
                generated_text = self.processor.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=skip_special
                )
            else:
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=skip_special
                )
        else:
            # For text-only models, use standard tokenization
            # Format prompt
            # Check if function accepts ablation_config (handle cached bytecode issues)
            import inspect
            sig = inspect.signature(format_training_prompt)
            if 'ablation_config' in sig.parameters:
                prompt = format_training_prompt(
                    task_id=task_id,
                    slide_number=slide_number,
                    grounding=grounding or "{}",
                    short_term_memory=short_term_memory or "{}",
                    long_term_memory=long_term_memory or "{}",
                    ablation_config=ablation_config
                )
            else:
                # Fallback: function doesn't have ablation_config parameter yet
                prompt = format_training_prompt(
                    task_id=task_id,
                    slide_number=slide_number,
                    grounding=grounding or "{}",
                    short_term_memory=short_term_memory or "{}",
                    long_term_memory=long_term_memory or "{}"
                )
            
            # Tokenize (optimized for speed - reduce max_length)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Reduced from 2048 for faster processing
            )
            
            # Ensure inputs are on the correct device (GPU)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Verify GPU usage
            if torch.cuda.is_available() and device.type == "cuda":
                pass  # Good, using GPU
            else:
                print(f"⚠️  WARNING: Model is on {device}, not GPU! This will be very slow.")
            
            # Get generation parameters
            max_new_tokens = inference_config.get("max_new_tokens", 512)
            do_sample = inference_config.get("do_sample", False)
            temperature = inference_config.get("temperature", 0.2) if do_sample else None
            top_p = inference_config.get("top_p", 0.9) if do_sample else None
            
            # Prepare generation kwargs (optimized for speed)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": inference_config.get("num_return_sequences", 1),
                "num_beams": 1,  # Explicitly set for greedy decoding (faster)
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else None,
                "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None,
                "use_cache": inference_config.get("use_cache", True),  # Enable KV cache for speed
                "output_scores": False,  # Don't compute scores (faster)
                "return_dict_in_generate": False,  # Don't return dict (faster)
                "early_stopping": inference_config.get("early_stopping", True),  # Stop early if EOS found (faster)
            }
            
            # Add repetition penalty if specified (default 1.0 = no penalty)
            if "repetition_penalty" in inference_config:
                generation_kwargs["repetition_penalty"] = inference_config["repetition_penalty"]
            
            # Add sampling parameters only if sampling is enabled
            if do_sample:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            
            # Generate with optimizations
            # Use inference_mode for faster inference (slightly faster than no_grad)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            
            # Decode (skip special tokens for faster decoding)
            skip_special = inference_config.get("skip_special_tokens", True)
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=skip_special
            )
        
        # Extract output JSON (after "### Response")
        if "### Response" in generated_text:
            response_part = generated_text.split("### Response")[-1].strip()
        else:
            response_part = generated_text[len(prompt):].strip()
        
        # Parse JSON with improved error handling (optimized for speed)
        output_json = {}
        try:
            # Try using orjson for faster JSON parsing if available
            try:
                import orjson
                use_orjson = True
            except ImportError:
                use_orjson = False
            # Try to extract complete JSON (FULL output, not just Predicted section)
            # The model should generate the full output with Grounding, Short Term, Long Term, Predicted, etc.
            # We need to parse the FULL output to extract memory sections
            response_part = response_part.strip()
            
            # Find the first opening brace (start of JSON)
            first_brace = response_part.find('{')
            if first_brace != -1:
                response_part = response_part[first_brace:]
            
            # Remove any trailing text after closing brace
            if "}" in response_part:
                # Find the last complete JSON object
                # Count braces to find matching closing brace
                brace_count = 0
                last_valid_brace = -1
                for i, char in enumerate(response_part):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid_brace = i
                            break
                
                if last_valid_brace != -1:
                    response_part = response_part[:last_valid_brace + 1]
                else:
                    # Fallback: use last closing brace
                    last_brace = response_part.rfind("}")
                    if last_brace != -1:
                        response_part = response_part[:last_brace + 1]
            
            # Try to parse JSON (use orjson if available for speed)
            if use_orjson:
                output_json = orjson.loads(response_part)
            else:
                output_json = json.loads(response_part)
            
            # Debug: Check what keys are in the output
            output_keys = list(output_json.keys()) if isinstance(output_json, dict) else []
            if "Short Term" not in output_keys or "Long Term" not in output_keys:
                # Model didn't generate memory sections - log for debugging
                print(f"⚠️  Model output missing memory sections. Keys: {output_keys}")
                print(f"   Response length: {len(response_part)} chars")
                if len(response_part) < 500:
                    print(f"   Full response: {response_part}")
            
            # If JSON contains Grounding but not Predicted, try to extract from it
            if "Predicted" not in output_json and "Grounding" in output_json:
                # Model copied input format - this is an error, but try to extract
                # Sometimes the model generates Grounding instead of Predicted
                # We'll mark this but still try to extract what we can
                print(f"⚠️  Model generated Grounding instead of Predicted - this will be counted as incorrect")
                # Keep the output_json as is - evaluation code will handle it
                
        except json.JSONDecodeError as e:
            # Try to repair common JSON errors
            try:
                # Try to fix common issues like missing commas, quotes, etc.
                # Remove trailing incomplete structures
                if "Expecting ',' delimiter" in str(e):
                    # Try to find where the error occurred and fix it
                    error_pos = int(str(e).split("char ")[-1].split(")")[0])
                    # Truncate at error position and try to close JSON
                    repaired = response_part[:error_pos].rstrip()
                    # Remove incomplete last key-value pair
                    if ":" in repaired:
                        last_colon = repaired.rfind(":")
                        repaired = repaired[:last_colon].rstrip()
                        # Remove trailing comma
                        if repaired.endswith(","):
                            repaired = repaired[:-1]
                        # Try to close JSON structure
                        open_braces = repaired.count("{") - repaired.count("}")
                        repaired += "}" * open_braces
                        if use_orjson:
                            output_json = orjson.loads(repaired)
                        else:
                            output_json = json.loads(repaired)
                    else:
                        output_json = {}
                else:
                    # Try to extract action and target even from incomplete JSON
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response_part)
                    target_match = re.search(r'"target"\s*:\s*"([^"]+)"', response_part)
                    
                    if action_match and target_match:
                        # Create minimal valid JSON structure with Predicted format
                        output_json = {
                            "Predicted": {
                                "predicted_action": {
                                    "action": action_match.group(1),
                                    "target": target_match.group(1)
                                }
                            }
                        }
                        print(f"⚠️  Extracted action/target from incomplete JSON: {action_match.group(1)}, {target_match.group(1)}")
                    else:
                        # Check if we have Grounding format
                        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response_part)
                        target_match = re.search(r'"target"\s*:\s*"([^"]+)"', response_part)
                        
                        if action_match and target_match:
                            # Create JSON in Grounding format (will be handled by evaluation)
                            output_json = {
                                "Grounding": {
                                    "ground_truth": {
                                        "action": action_match.group(1),
                                        "target": target_match.group(1)
                                    }
                                }
                            }
                            print(f"⚠️  Extracted action/target from incomplete JSON (Grounding format): {action_match.group(1)}, {target_match.group(1)}")
                        else:
                            print(f"Warning: Could not parse JSON from response: {e}")
                            print(f"Response part (first 500 chars): {response_part[:500]}...")
                            output_json = {}
            except Exception as repair_error:
                # Final fallback: try to extract action/target using regex
                action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response_part)
                target_match = re.search(r'"target"\s*:\s*"([^"]+)"', response_part)
                
                if action_match and target_match:
                    output_json = {
                        "Predicted": {
                            "predicted_action": {
                                "action": action_match.group(1),
                                "target": target_match.group(1)
                            }
                        }
                    }
                    print(f"⚠️  Extracted action/target from malformed JSON: {action_match.group(1)}, {target_match.group(1)}")
                else:
                    print(f"Warning: Could not parse or repair JSON from response: {e}")
                    print(f"Repair error: {repair_error}")
                    print(f"Response part (first 500 chars): {response_part[:500]}...")
                    output_json = {}
        
        # Extract memory for next step
        extracted_memory = self.extract_memory_from_output(output_json)
        
        # Get ablation config to respect memory updates
        ablation_config = get_config("ablation")
        
        # Update memory state, respecting ablation config
        # If ablation disables a memory type, don't update it (keep existing or set to empty)
        if ablation_config.get("use_short_term_memory", True):
            self.memory_state["short_term"] = extracted_memory.get("short_term", {})
        else:
            # If STM is disabled, clear it
            self.memory_state["short_term"] = {}
        
        if ablation_config.get("use_long_term_memory", True):
            self.memory_state["long_term"] = extracted_memory.get("long_term", {})
        else:
            # If LTM is disabled, clear it
            self.memory_state["long_term"] = {}
        
        # Clear CUDA cache periodically to prevent OOM (less frequent = faster)
        if torch.cuda.is_available():
            inference_config = get_config("inference")
            clear_freq = inference_config.get("clear_cache_frequency", 20)  # Increased from 10 to 20 for speed
            # Only clear cache every N steps to reduce overhead
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1
            if self._step_count % clear_freq == 0:
                torch.cuda.empty_cache()
        
        # Use orjson for faster JSON serialization if available
        try:
            import orjson
            use_orjson_serialize = True
        except ImportError:
            use_orjson_serialize = False
        
        if use_orjson_serialize:
            output_json_str = orjson.dumps(output_json).decode('utf-8')
        else:
            output_json_str = json.dumps(output_json, ensure_ascii=False)
        
        return {
            "output": output_json,
            "output_json_str": output_json_str,
            "extracted_memory": extracted_memory,
            "full_response": generated_text,
        }
    
    def predict_sequence(
        self,
        steps: List[Dict[str, Any]],
        task_id: str,
        image_base_path: Optional[str] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict outputs for a sequence of steps.
        
        IMPORTANT: Memory comes from model predictions, NOT from test CSV.
        This prevents data leakage during evaluation.
        
        Args:
            steps: List of step dictionaries with keys: slide_number, grounding, image_id
            task_id: Task identifier
            image_base_path: Base path for images
            show_progress: Show progress bar
            
        Returns:
            List of prediction results
        """
        # Reset memory for new task
        self.reset_memory(task_id=task_id)
        
        predictions = []
        
        # Progress bar for steps
        steps_iter = tqdm(steps, desc=f"Task {task_id} steps", unit="step") if show_progress else steps
        
        for i, step in enumerate(steps_iter):
            slide_number = str(step.get("slide_number", i + 1))
            grounding = step.get("grounding", "{}")
            image_id = step.get("image_id")
            
            if show_progress:
                steps_iter.set_description(f"Task {task_id} step {slide_number}")
            
            # CRITICAL: Do NOT use Short_Term_Memory or Long_term_Memory from step dict
            # Memory comes from self.memory_state (from previous predictions)
            # This prevents data leakage
            
            # Predict
            result = self.predict_step(
                task_id=task_id,
                slide_number=slide_number,
                grounding=grounding,
                image_id=image_id,
                image_base_path=image_base_path,
                # NOTE: short_term_memory and long_term_memory are NOT passed
                # They come from self.memory_state (from previous predictions)
            )
            
            predictions.append({
                "step_number": slide_number,
                "prediction": result,
            })
        
        return predictions

def load_model_for_inference(
    model_path: str,
    base_model_name: Optional[str] = None,
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None
) -> StepPredictor:
    """
    Load model for inference.
    
    Args:
        model_path: Path to fine-tuned model
        base_model_name: Base model name if using PEFT
        trust_remote_code: Whether to trust remote code
        hf_token: HuggingFace token for authentication
        
    Returns:
        StepPredictor instance
    """
    # Get token from config if not provided
    if hf_token is None:
        config = get_config("model")
        hf_token = config.get("hf_token")
    
    return StepPredictor(
        model_path=model_path,
        base_model_name=base_model_name,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token
    )

if __name__ == "__main__":
    # Test inference
    config = get_config()
    
    # Example usage
    predictor = load_model_for_inference(
        model_path=config["output"]["output_dir"],
        base_model_name=config["model"]["model_name"] if config["model"]["use_local"] else None,
    )
    
    # Example step prediction
    result = predictor.predict_step(
        task_id="1",
        slide_number="1",
        grounding='{"ground_truth": {"action": "CLICK", "target": "Load Data"}}',
    )
    
    print("\nPrediction result:")
    print(json.dumps(result, indent=2))

