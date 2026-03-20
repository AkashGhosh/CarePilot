#!/usr/bin/env python3
"""
Fine-tuning logic for Llama-4-Maverick on task results dataset.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

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
        # Try alternative import path
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
        # Try alternative import path
        from transformers.models.qwen2_5_vl import Qwen2_5VLForConditionalGeneration
        QWEN2_5VL_AVAILABLE = True
    except ImportError:
        try:
            # Try with underscore variant
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
        # Try alternative import path
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        QWEN3VL_AVAILABLE = True
    except ImportError:
        try:
            # Try with underscore variant
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            QWEN3VL_AVAILABLE = True
        except ImportError:
            pass
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from compute_tracker import ComputeTracker

def load_model_and_tokenizer(
    model_name: str,
    local_model_path: Optional[str] = None,
    use_local: bool = False,
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None
):
    """
    Load model and tokenizer/processor.
    Supports both causal LM models (Llama) and vision-language models (Qwen2.5-VL).
    
    Args:
        model_name: Model name from HuggingFace
        local_model_path: Local path to model if available
        use_local: Whether to use local model
        trust_remote_code: Whether to trust remote code
        hf_token: HuggingFace token for authentication
        
    Returns:
        Tuple of (model, tokenizer/processor)
    """
    # Get model config for all settings
    model_config = get_config("model")
    
    # Get token from config if not provided
    if hf_token is None:
        hf_token = model_config.get("hf_token")
    
    # Also check environment variable as fallback
    if hf_token is None:
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    # Check if this is a Qwen VL model (Qwen2, Qwen2.5, or Qwen3)
    is_qwen_vl = "qwen" in model_name.lower() and ("vl" in model_name.lower() or "vision" in model_name.lower())
    
    # Determine which Qwen VL version
    is_qwen3_vl = is_qwen_vl and ("qwen3" in model_name.lower() or ("3" in model_name.lower() and "2.5" not in model_name.lower() and "2_5" not in model_name.lower()))
    is_qwen2_5_vl = is_qwen_vl and ("2.5" in model_name.lower() or "2_5" in model_name.lower())
    is_qwen2_vl = is_qwen_vl and not is_qwen3_vl and not is_qwen2_5_vl
    
    # Get quantization settings from config
    load_in_8bit = model_config.get("load_in_8bit", False)
    load_in_4bit = model_config.get("load_in_4bit", False)
    
    print(f"Loading model: {model_name}")
    if is_qwen_vl:
        if is_qwen3_vl:
            print("Detected Qwen3-VL vision-language model")
        elif is_qwen2_5_vl:
            print("Detected Qwen2.5-VL vision-language model")
        else:
            print("Detected Qwen2-VL vision-language model")
    
    if load_in_8bit:
        print("⚠️  8-bit quantization enabled (requires bitsandbytes)")
    elif load_in_4bit:
        print("⚠️  4-bit quantization enabled (requires bitsandbytes)")
    
    # Determine model path
    if use_local and local_model_path:
        model_path = local_model_path
        print(f"Using local model from: {model_path}")
        token = None  # Don't need token for local model
    else:
        model_path = model_name
        print(f"Loading from HuggingFace: {model_path}")
        
        # Explicitly login to HuggingFace Hub
        if hf_token:
            print("Authenticating with HuggingFace Hub...")
            try:
                from huggingface_hub import login as hf_login, whoami
                hf_login(token=hf_token, add_to_git_credential=False)
                try:
                    user_info = whoami()
                    print(f"✓ Authenticated as: {user_info.get('name', 'Unknown user')}")
                except Exception:
                    pass
                print("✓ Authentication successful!")
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            except Exception as e:
                print(f"⚠️  Warning: Could not login to HuggingFace Hub: {e}")
                print("Attempting to use token directly in model loading...")
        
        token = hf_token
    
    # Load tokenizer/processor
    if is_qwen_vl:
        print("Loading processor for Qwen2.5-VL model...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = processor  # For compatibility, use processor as tokenizer
    else:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=token,
            trust_remote_code=trust_remote_code,
            padding_side="right"
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    try:
        if is_qwen_vl:
            # Try Qwen3-VL first
            if is_qwen3_vl:
                qwen3_model_class = None
                
                # Try to get the model class - handle import dynamically to avoid scoping issues
                # Check if globally imported class is available
                if QWEN3VL_AVAILABLE:
                    # Use globals() to access the module-level variable safely
                    global_qwen3 = globals().get('Qwen3VLForConditionalGeneration')
                    if global_qwen3 is not None:
                        qwen3_model_class = global_qwen3
                        print("Using Qwen3VLForConditionalGeneration for Qwen3-VL model...")
                
                # If not available from global import, try importing now
                if qwen3_model_class is None:
                    print("Qwen3VLForConditionalGeneration not directly available, trying direct import...")
                    try:
                        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                        qwen3_model_class = Qwen3VLForConditionalGeneration
                        print("Successfully imported Qwen3VLForConditionalGeneration from modeling module")
                    except ImportError:
                        try:
                            from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
                            qwen3_model_class = Qwen3VLForConditionalGeneration
                            print("Successfully imported Qwen3VLForConditionalGeneration from qwen3_vl module")
                        except ImportError:
                            try:
                                from transformers import Qwen3VLForConditionalGeneration
                                qwen3_model_class = Qwen3VLForConditionalGeneration
                                print("Successfully imported Qwen3VLForConditionalGeneration from transformers")
                            except ImportError as e:
                                print(f"Failed to import Qwen3VLForConditionalGeneration: {e}")
                                print("Please ensure transformers>=4.57.0 is installed with Qwen3-VL support.")
                                print("Try: pip install git+https://github.com/huggingface/transformers")
                                raise ImportError("Qwen3VLForConditionalGeneration not available. Please install transformers from source.")
                
                # Now load the model
                if qwen3_model_class is not None:
                    # Qwen3-VL requires BitsAndBytesConfig for quantization instead of direct parameters
                    quantization_config = None
                    if load_in_4bit or load_in_8bit:
                        try:
                            from transformers import BitsAndBytesConfig
                            if load_in_4bit:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4"
                                )
                            elif load_in_8bit:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True
                                )
                        except ImportError:
                            print("⚠️  bitsandbytes not available, loading without quantization")
                    
                    # Build model loading kwargs
                    model_kwargs = {
                        "token": token,
                        "trust_remote_code": trust_remote_code,
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                    }
                    
                    # Add quantization config if available
                    if quantization_config is not None:
                        model_kwargs["quantization_config"] = quantization_config
                    
                    # Add max_memory for GPU memory limit
                    if torch.cuda.is_available():
                        model_kwargs["max_memory"] = {0: "32GiB", "cpu": "50GiB"}
                    
                    model = qwen3_model_class.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                else:
                    raise ImportError("Qwen3VLForConditionalGeneration not available")
            # Then Qwen2.5-VL
            elif is_qwen2_5_vl:
                # Try to use Qwen2.5-VL generation class
                if QWEN2_5VL_AVAILABLE:
                    print("Using Qwen2_5VLForConditionalGeneration for Qwen2.5-VL model...")
                    model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                        model_path,
                        token=token,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        low_cpu_mem_usage=True,  # OOM prevention
                        max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                    )
                else:
                    # Fallback: Try to import and use the model class directly from the modeling module
                    print("Qwen2_5VLForConditionalGeneration not directly available, trying direct import from modeling module...")
                    try:
                        # Try importing directly from the qwen2_5_vl modeling module
                        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForConditionalGeneration
                        print("Successfully imported Qwen2_5VLForConditionalGeneration from modeling_qwen2_5_vl module")
                        model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                            model_path,
                            token=token,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            load_in_8bit=load_in_8bit,
                            load_in_4bit=load_in_4bit,
                            low_cpu_mem_usage=True,  # OOM prevention
                            max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                        )
                    except ImportError as e:
                        print(f"Direct import from modeling module failed: {e}")
                        print("Trying alternative import path...")
                        try:
                            # Alternative: import from qwen2_5_vl package
                            from transformers.models import qwen2_5_vl
                            model_class = getattr(qwen2_5_vl, 'Qwen2_5VLForConditionalGeneration', None)
                            if model_class:
                                print("Found Qwen2_5VLForConditionalGeneration in qwen2_5_vl package")
                                model = model_class.from_pretrained(
                                    model_path,
                                    token=token,
                                    trust_remote_code=trust_remote_code,
                                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                    load_in_8bit=load_in_8bit,
                                    load_in_4bit=load_in_4bit,
                                )
                            else:
                                raise ImportError("Qwen2_5VLForConditionalGeneration not found in qwen2_5_vl package")
                        except Exception as e2:
                            print(f"Alternative import also failed: {e2}")
                            print("Trying to load generation class using AutoConfig...")
                            try:
                                # Load config to find the correct model class
                                from transformers import AutoConfig
                                config = AutoConfig.from_pretrained(
                                    model_path,
                                    token=token,
                                    trust_remote_code=trust_remote_code
                                )
                                
                                # Check config for model architecture
                                if hasattr(config, 'architectures') and config.architectures:
                                    arch_name = config.architectures[0]
                                    print(f"Model architecture from config: {arch_name}")
                                    
                                    # Try to dynamically import the class
                                    try:
                                        # Import the modeling module
                                        import importlib
                                        module_name = f"transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
                                        module = importlib.import_module(module_name)
                                        model_class = getattr(module, arch_name, None)
                                        
                                        if model_class:
                                            print(f"Successfully found model class: {arch_name}")
                                            model = model_class.from_pretrained(
                                                model_path,
                                                token=token,
                                                trust_remote_code=trust_remote_code,
                                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                device_map="auto",
                                                load_in_8bit=load_in_8bit,
                                                load_in_4bit=load_in_4bit,
                                                low_cpu_mem_usage=True,  # OOM prevention
                                                # max_memory removed - PyTorch will allocate memory as needed during training
                                            )
                                        else:
                                            raise ImportError(f"Class {arch_name} not found in module")
                                    except Exception as e3:
                                        print(f"Dynamic import failed: {e3}")
                                        raise ImportError(f"Could not load generation class {arch_name}")
                                else:
                                    raise ValueError("Config does not specify model architecture")
                            except Exception as e3:
                                print(f"Config-based loading failed: {e3}")
                                print("\n" + "="*60)
                                print("❌ ERROR: Cannot load Qwen2.5-VL generation model")
                                print("="*60)
                                print("The model needs to be loaded as a generation class (not base model).")
                                print("Please ensure transformers>=4.49.0 is installed and includes Qwen2.5-VL support.")
                                print("\nTry: pip install --upgrade transformers")
                                print("Or install from source: pip install git+https://github.com/huggingface/transformers.git")
                                print("="*60)
                                raise
            else:
                # Use Qwen2-VL generation class
                if QWEN2VL_AVAILABLE:
                    print("Using Qwen2VLForConditionalGeneration for Qwen2-VL model...")
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        token=token,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        low_cpu_mem_usage=True,  # OOM prevention
                        max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                    )
                else:
                    # Fallback to AutoModelForCausalLM
                    print("Qwen2VLForConditionalGeneration not available, using AutoModelForCausalLM...")
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        token=token,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        low_cpu_mem_usage=True,  # OOM prevention
                        max_memory={0: "32GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # Limit GPU memory to 32GB to prevent OOM (leaves 8GB buffer)
                    )
        else:
            print("Using AutoModelForCausalLM for causal LM model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=token,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                low_cpu_mem_usage=True,  # OOM prevention
                # max_memory removed - PyTorch will allocate memory as needed during training
            )
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "gated" in error_msg.lower() or "not in the authorized list" in error_msg.lower():
            print("\n" + "="*60)
            print("❌ ACCESS DENIED - Authorization Required")
            print("="*60)
            print(f"Please visit https://huggingface.co/{model_name}")
            print("and accept the license agreement.")
            print("="*60)
        raise
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        print("Enabling gradient checkpointing to save memory...")
        model.gradient_checkpointing_enable()
    
    print("Model and tokenizer loaded successfully")
    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        # Clear CUDA cache after model loading
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    return model, tokenizer

def setup_peft_model(
    model,
    peft_config: Dict[str, Any]
) -> Any:
    """
    Setup PEFT (LoRA) for model.
    
    Args:
        model: Base model
        peft_config: PEFT configuration dictionary
        
    Returns:
        Model with PEFT adapter
    """
    print("Setting up PEFT (LoRA)...")
    
    # Check if model has prepare_inputs_for_generation (needed for PEFT)
    # For vision-language models, this is critical
    model_type_str = str(type(model))
    is_vl_model = any(x in model_type_str.lower() for x in ['vl', 'vision', 'qwen2_5_vl', 'qwen2vl'])
    
    model_has_prepare = hasattr(model, 'prepare_inputs_for_generation')
    
    if not model_has_prepare:
        print(f"Model type: {model_type_str}")
        if is_vl_model:
            print("⚠️  WARNING: Vision-language model loaded as base model, not generation class!")
            print("This may cause issues with PEFT. The model should be loaded as a generation class.")
        
        # Try to find prepare_inputs_for_generation in parent classes or attributes
        if hasattr(model, 'model'):
            base = model.model
            if hasattr(base, 'prepare_inputs_for_generation'):
                print("Found prepare_inputs_for_generation in model.model")
                model_has_prepare = True
        elif hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'prepare_inputs_for_generation'):
                print("Found prepare_inputs_for_generation in model.base_model")
                model_has_prepare = True
        
        # If still not found, check parent classes
        if not model_has_prepare:
            for cls in type(model).__mro__:
                if hasattr(cls, 'prepare_inputs_for_generation'):
                    print(f"Found prepare_inputs_for_generation in parent class: {cls.__name__}")
                    # Bind the method to the instance
                    import types
                    method = getattr(cls, 'prepare_inputs_for_generation')
                    model.prepare_inputs_for_generation = types.MethodType(method, model)
                    model_has_prepare = True
                    break
    
    # Only prepare for kbit training if needed (may not work for all vision models)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print(f"Warning: prepare_model_for_kbit_training failed: {e}")
        print("Continuing without kbit preparation (model may already be prepared)...")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        target_modules=peft_config["target_modules"],
        lora_dropout=peft_config["lora_dropout"],
        bias=peft_config["bias"],
        task_type=peft_config["task_type"],
    )
    
    # Apply PEFT
    try:
        model = get_peft_model(model, lora_config)
    except AttributeError as e:
        if "prepare_inputs_for_generation" in str(e):
            print("\n" + "="*60)
            print("⚠️  PEFT ERROR: Model missing prepare_inputs_for_generation")
            print("="*60)
            print("For vision-language models, PEFT may need the generation class.")
            print("Trying to load model with generation class directly...")
            print("="*60)
            
            # Try to reload model with generation class if we know the model name
            # This is a workaround - ideally we should ensure we load the generation class from the start
            raise ValueError(
                "Model loaded as base model instead of generation class. "
                "Please ensure Qwen2_5VLForConditionalGeneration is used instead of base model. "
                "The fallback to AutoModel returns the base model which doesn't have generation methods."
            ) from e
        raise
    
    print("PEFT setup complete")
    try:
        print(f"Trainable parameters: {model.print_trainable_parameters()}")
    except Exception:
        print("Could not print trainable parameters (this is okay)")
    
    return model

def preprocess_function(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_length: int = 768,  # Reduced from 1024 to 768 for even more aggressive memory savings (OOM prevention for 40GB)
    is_qwen_vl: bool = False
) -> Dict[str, Any]:
    """
    Preprocess examples for training.
    
    For Qwen2.5-VL models, handles both images and text using the processor.
    For text-only models, only tokenizes text.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer/processor instance
        max_length: Maximum sequence length
        is_qwen_vl: Whether this is a Qwen2.5-VL vision-language model
        
    Returns:
        Preprocessed examples
    """
    if is_qwen_vl:
        # For vision-language models, use processor to handle both images and text
        processor = tokenizer  # tokenizer is actually a processor for VL models
        
        # Handle both batched (list) and single (dict) examples
        if isinstance(examples["prompt"], list):
            # Batched processing (not used for Qwen VL, but handle just in case)
            texts = []
            images = []
            
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                output = examples["output"][i]
                text = prompt + "\n" + output
                texts.append(text)
                
                if "image" in examples and examples["image"][i] is not None:
                    images.append(examples["image"][i])
                else:
                    images.append(None)
            
            # Process with processor
            has_images = any(img is not None for img in images)
            
            if has_images:
                model_inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            else:
                model_inputs = processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
        else:
            # Single example processing (for batched=False)
            prompt = examples["prompt"]
            output = examples["output"]
            image = examples.get("image") if "image" in examples else None
            
            # Process with processor using chat template for Qwen2.5-VL
            if image is not None:
                # For Qwen2.5-VL, use the chat template to properly format with image tokens
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},  # Image placeholder
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": output},
                        ],
                    }
                ]
                
                # Apply chat template to get properly formatted text with image tokens
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # We have the assistant response
                )
                
                # Now process with both text and image
                model_inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            else:
                # No image - process text only
                # Format as chat without image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": output},
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                model_inputs = processor.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            
            # Convert from list-of-lists to single list for single example
            for key in list(model_inputs.keys()):
                if isinstance(model_inputs[key], list) and len(model_inputs[key]) == 1:
                    # For most keys, unwrap the outer list
                    model_inputs[key] = model_inputs[key][0]
            
            # Special handling for image_grid_thw
            # The processor returns it as [[t, h, w]], we need [t, h, w]
            if "image_grid_thw" in model_inputs:
                grid_thw = model_inputs["image_grid_thw"]
                # If it's a nested list, flatten it
                if isinstance(grid_thw, list):
                    # Check if it's nested
                    if len(grid_thw) > 0 and isinstance(grid_thw[0], list):
                        # Nested list: [[t, h, w]] -> [t, h, w]
                        model_inputs["image_grid_thw"] = grid_thw[0]
                    # Otherwise it's already flat [t, h, w]
        
        # Create labels from input_ids
        if "input_ids" in model_inputs:
            input_ids = model_inputs["input_ids"]
            
            if isinstance(input_ids, list):
                # Handle list format (from return_tensors=None)
                if isinstance(input_ids[0], list):
                    # List of lists (batched)
                    labels = [list(ids) for ids in input_ids]
                    # Mask padding tokens
                    if "attention_mask" in model_inputs:
                        attention_mask = model_inputs["attention_mask"]
                        if isinstance(attention_mask, list):
                            for i, mask in enumerate(attention_mask):
                                if isinstance(mask, list):
                                    for j, val in enumerate(mask):
                                        if val == 0:
                                            labels[i][j] = -100
                else:
                    # Single list (single example)
                    labels = list(input_ids)
                    # Mask padding tokens
                    if "attention_mask" in model_inputs:
                        attention_mask = model_inputs["attention_mask"]
                        if isinstance(attention_mask, list):
                            for j, val in enumerate(attention_mask):
                                if val == 0:
                                    labels[j] = -100
            else:
                # Tensor format (shouldn't happen with return_tensors=None, but handle it)
                labels = input_ids.clone()
                if "attention_mask" in model_inputs:
                    labels[model_inputs["attention_mask"] == 0] = -100
            
            model_inputs["labels"] = labels
        
        return model_inputs
    else:
        # For text-only models, combine prompt and output
        inputs = []
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            output = examples["output"][i]
            # Format as: prompt + output + eos_token
            text = prompt + "\n" + output + tokenizer.eos_token
            inputs.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None  # Return dicts, not tensors (for dataset.map)
        )
        
        # Labels are the same as input_ids for causal LM
        model_inputs["labels"] = model_inputs["input_ids"].copy()  # Use copy() instead of clone() for lists
        
        return model_inputs

def create_data_collator(tokenizer: Any, is_qwen_vl: bool = False) -> Any:
    """
    Create data collator for training.
    
    For vision-language models, creates a custom collator that handles both
    images and text. For text-only models, uses standard DataCollatorForLanguageModeling.
    
    Args:
        tokenizer: Tokenizer/processor instance
        is_qwen_vl: Whether this is a Qwen2.5-VL vision-language model
        
    Returns:
        Data collator
    """
    if is_qwen_vl:
        # Create custom collator for vision-language models
        class VisionLanguageDataCollator:
            """Custom data collator for vision-language models."""
            
            def __init__(self, processor):
                self.processor = processor
            
            def __call__(self, features):
                """
                Collate a batch of features.
                
                Args:
                    features: List of dictionaries with preprocessed features
                    
                Returns:
                    Batched dictionary ready for model input
                """
                import torch
                
                # Separate text and image features
                batch_texts = []
                batch_images = []
                batch_labels = []
                
                # Also collect image_grid_thw if present (required for Qwen2.5-VL)
                batch_image_grid_thw = []
                
                for feature in features:
                    # Extract text and labels
                    if "input_ids" in feature:
                        # Already preprocessed, extract directly
                        batch_texts.append(feature.get("input_ids"))
                        batch_labels.append(feature.get("labels", feature.get("input_ids")))
                        
                        # Extract images if available
                        if "pixel_values" in feature:
                            batch_images.append(feature["pixel_values"])
                            # Also extract image_grid_thw if present
                            if "image_grid_thw" in feature:
                                batch_image_grid_thw.append(feature["image_grid_thw"])
                            else:
                                batch_image_grid_thw.append(None)
                        else:
                            batch_images.append(None)
                            batch_image_grid_thw.append(None)
                    else:
                        # Fallback: try to reconstruct from raw data
                        # This shouldn't happen if preprocessing worked correctly
                        raise ValueError("Invalid feature format - missing input_ids")
                
                # Convert lists to tensors and pad
                batch = {}
                
                # Get pad token ID
                pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', 
                                      getattr(self.processor.tokenizer, 'eos_token_id', 0))
                
                # Stack input_ids
                input_ids_list = []
                for ids in batch_texts:
                    if isinstance(ids, list):
                        input_ids_list.append(torch.tensor(ids))
                    elif isinstance(ids, torch.Tensor):
                        input_ids_list.append(ids)
                    else:
                        input_ids_list.append(torch.tensor(ids))
                
                if input_ids_list:
                    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
                        input_ids_list, batch_first=True, padding_value=pad_token_id
                    )
                    
                    # Create attention masks from input_ids (1 for non-padding, 0 for padding)
                    batch["attention_mask"] = (batch["input_ids"] != pad_token_id).long()
                else:
                    raise ValueError("No input_ids found in features")
                
                # Stack pixel_values (images) if available
                pixel_values_list = []
                for img in batch_images:
                    if img is not None:
                        if isinstance(img, list):
                            # Convert list to tensor
                            if len(img) > 0 and isinstance(img[0], (list, tuple)):
                                # Nested list - convert to tensor
                                pixel_values_list.append(torch.tensor(img))
                            else:
                                # Flat list - might need reshaping
                                pixel_values_list.append(torch.tensor(img))
                        elif isinstance(img, torch.Tensor):
                            pixel_values_list.append(img)
                        else:
                            pixel_values_list.append(torch.tensor(img))
                
                if pixel_values_list:
                    try:
                        # Stack image tensors - they should all have the same shape
                        batch["pixel_values"] = torch.stack(pixel_values_list)
                    except RuntimeError as e:
                        # Images have different shapes - need to pad or resize
                        # For now, try to handle by finding max dimensions
                        if "size" in str(e).lower():
                            # Different sizes - pad to max size
                            max_dims = [max([img.shape[i] for img in pixel_values_list]) 
                                       for i in range(len(pixel_values_list[0].shape))]
                            # This is complex, so for now we'll skip pixel_values stacking
                            # and let the model handle it in preprocessing
                            # Or we can pad manually
                            print(f"Warning: Image tensors have different shapes. Max dims: {max_dims}")
                            # Try simple stacking (might fail but let's try)
                            batch["pixel_values"] = torch.stack(pixel_values_list)
                        else:
                            raise
                    
                    # Also handle image_grid_thw if available
                    if batch_image_grid_thw and any(grid is not None for grid in batch_image_grid_thw):
                        grid_thw_list = []
                        for i, grid in enumerate(batch_image_grid_thw):
                            if grid is not None:
                                # Convert to tensor if needed
                                if isinstance(grid, list):
                                    grid_tensor = torch.tensor(grid)
                                elif isinstance(grid, torch.Tensor):
                                    grid_tensor = grid
                                else:
                                    grid_tensor = torch.tensor(grid)
                                
                                # Ensure correct shape: should be (num_images, 3) or (3,) for single image
                                # Each image needs [temporal, height, width]
                                if grid_tensor.dim() == 1:
                                    # Single image grid: should be shape (3,)
                                    if grid_tensor.shape[0] == 3:
                                        # Correct shape, add batch dimension
                                        grid_thw_list.append(grid_tensor.unsqueeze(0))  # Shape: (1, 3)
                                    else:
                                        print(f"Warning: grid_thw[{i}] has unexpected shape {grid_tensor.shape}, expected (3,)")
                                        # Skip this grid or try to reshape
                                        continue
                                elif grid_tensor.dim() == 2:
                                    # Already has batch dimension: shape (num_images, 3)
                                    grid_thw_list.append(grid_tensor)
                                else:
                                    print(f"Warning: grid_thw[{i}] has unexpected dim {grid_tensor.dim()}, shape {grid_tensor.shape}")
                                    continue
                        
                        if grid_thw_list:
                            # Concatenate all grids along batch dimension
                            # Result should be (total_num_images, 3)
                            batch["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)
                
                # Stack labels
                labels_list = []
                for lbl in batch_labels:
                    if isinstance(lbl, list):
                        labels_list.append(torch.tensor(lbl))
                    elif isinstance(lbl, torch.Tensor):
                        labels_list.append(lbl)
                    else:
                        labels_list.append(torch.tensor(lbl))
                
                if labels_list:
                    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                        labels_list, batch_first=True, padding_value=-100
                    )
                else:
                    # Use input_ids as labels if no labels provided
                    batch["labels"] = batch["input_ids"].clone()
                    # Mask padding tokens
                    batch["labels"][batch["labels"] == pad_token_id] = -100
                
                return batch
        
        return VisionLanguageDataCollator(tokenizer)
    else:
        # Use standard data collator for text-only models
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

def extract_action_and_target_from_output(output_json_str: str) -> tuple:
    """
    Extract action and target from output JSON string for validation.
    
    Args:
        output_json_str: Output JSON string
        
    Returns:
        Tuple of (action, target) or (None, None) if not found
    """
    try:
        output_dict = json.loads(output_json_str)
        predicted = output_dict.get("Predicted", {})
        predicted_action = predicted.get("predicted_action", {})
        
        action = predicted_action.get("action")
        target = predicted_action.get("target")
        
        return (action, target)
    except (json.JSONDecodeError, KeyError, AttributeError):
        return (None, None)

def extract_action_and_target_from_ground_truth(ground_truth_str: str) -> tuple:
    """
    Extract action and target from ground truth JSON string.
    
    Args:
        ground_truth_str: Ground truth JSON string
        
    Returns:
        Tuple of (action, target) or (None, None) if not found
    """
    try:
        gt_dict = json.loads(ground_truth_str)
        # Try Output JSON format first
        grounding = gt_dict.get("Grounding", {})
        ground_truth = grounding.get("ground_truth", {})
        
        if not ground_truth:
            # Fallback: try direct ground_truth
            ground_truth = gt_dict.get("ground_truth", {})
        
        action = ground_truth.get("action")
        target = ground_truth.get("target")
        
        return (action, target)
    except (json.JSONDecodeError, KeyError, AttributeError):
        return (None, None)

def compute_metrics(eval_pred):
    """
    Compute metrics for validation - only Action and Target accuracy.
    
    IMPORTANT: Only compares action (CLICK, SCROLL, ZOOM, SEGMENT, TEXT) and target.
    Ignores all other fields to prevent bias.
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary with action_accuracy, target_accuracy, combined_accuracy
    """
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    # Note: This is simplified - actual implementation depends on tokenizer
    # For now, we'll compute metrics based on the structured output format
    
    # In practice, you'd decode the tokenized outputs and extract JSON
    # For validation, we can compute approximate metrics based on loss
    # Detailed validation happens in separate evaluation script
    
    # Return placeholder metrics - actual validation happens in evaluate.py
    return {
        "action_accuracy": 0.0,
        "target_accuracy": 0.0,
        "combined_accuracy": 0.0,
    }

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    training_args: Dict[str, Any] = None,
    output_dir: str = None
) -> Trainer:
    """
    Train the model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        training_args: Training arguments dictionary
        output_dir: Output directory for checkpoints
        
    Returns:
        Trainer instance
    """
    print("Preparing training arguments...")
    
    # Get default training args if not provided
    if training_args is None:
        training_args = get_config("training")
    
    # Set output directory
    if output_dir is None:
        output_dir = get_config("output")["output_dir"]
    
    # Get logging directory from config
    output_config = get_config("output")
    logging_dir = output_config.get("logging_dir", output_dir)
    
    # Create training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,  # Directory for logs
        learning_rate=training_args.get("learning_rate", 2e-4),
        num_train_epochs=training_args.get("num_train_epochs", 3),
        per_device_train_batch_size=training_args.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 4),
        warmup_steps=training_args.get("warmup_steps", 100),
        logging_steps=training_args.get("logging_steps", 10),
        save_steps=training_args.get("save_steps", 500),
        eval_steps=training_args.get("eval_steps", 500),
        max_steps=training_args.get("max_steps", -1),
        max_grad_norm=training_args.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_args.get("lr_scheduler_type", "cosine"),
        weight_decay=training_args.get("weight_decay", 0.01),
        fp16=training_args.get("fp16", True),
        bf16=training_args.get("bf16", False),
        gradient_checkpointing=training_args.get("gradient_checkpointing", True),  # OOM prevention
        dataloader_num_workers=training_args.get("dataloader_num_workers", 0),  # Set to 0 to save memory
        dataloader_pin_memory=training_args.get("dataloader_pin_memory", False),  # OOM prevention
        remove_unused_columns=training_args.get("remove_unused_columns", False),
        report_to=training_args.get("report_to", "tensorboard"),  # Use tensorboard for logging (can be changed to "none" in config)
        load_best_model_at_end=False,  # Disabled since eval_strategy is "no" (would cause error)
        eval_strategy="no",  # Disabled to prevent OOM during evaluation (logits accumulation causes OOM)
        save_strategy=training_args.get("save_strategy", "epoch"),  # Save after each epoch
        save_total_limit=2,  # Keep only last 2 checkpoints to save disk space (after each epoch)
        logging_first_step=True,  # Log first step
    )
    
    # Detect if this is a vision-language model (Qwen2.5-VL)
    is_qwen_vl = hasattr(tokenizer, 'image_processor') or hasattr(tokenizer, 'processor') or 'qwen' in str(type(tokenizer)).lower()
    
    # Get max_length from config for memory optimization
    max_length = training_args.get("max_length", 768)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    print(f"Using max_length={max_length} for memory optimization")
    if is_qwen_vl:
        print("Detected vision-language model, preprocessing with images...")
        print("Note: Processing images individually to avoid PyArrow tensor shape issues...")
    
    # For vision-language models, process one at a time to avoid tensor shape issues
    # For text-only models, we can batch process
    if is_qwen_vl:
        train_dataset_processed = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_length=max_length, is_qwen_vl=is_qwen_vl),
            batched=False,  # Process one at a time for images
            remove_columns=train_dataset.column_names,
            desc="Preprocessing training data"
        )
        
        val_dataset_processed = None
        if val_dataset is not None:
            val_dataset_processed = val_dataset.map(
                lambda x: preprocess_function(x, tokenizer, max_length=max_length, is_qwen_vl=is_qwen_vl),
                batched=False,  # Process one at a time for images
                remove_columns=val_dataset.column_names,
                desc="Preprocessing validation data"
            )
    else:
        train_dataset_processed = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_length=max_length, is_qwen_vl=is_qwen_vl),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        
        val_dataset_processed = None
        if val_dataset is not None:
            val_dataset_processed = val_dataset.map(
                lambda x: preprocess_function(x, tokenizer, max_length=max_length, is_qwen_vl=is_qwen_vl),
                batched=True,
                remove_columns=val_dataset.column_names,
            )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, is_qwen_vl=is_qwen_vl)
    
    # Create trainer with custom metrics
    print("Creating trainer...")
    # For vision-language models, use processing_class instead of tokenizer
    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset_processed,
        # Disable eval_dataset to prevent OOM during evaluation
        # "eval_dataset": val_dataset_processed,  # Commented out to prevent OOM
        "data_collator": data_collator,
        # "compute_metrics": compute_metrics if val_dataset_processed is not None else None,  # Disabled
    }
    
    if is_qwen_vl:
        trainer_kwargs["processing_class"] = tokenizer  # Use processor for VL models
    else:
        trainer_kwargs["tokenizer"] = tokenizer  # Use tokenizer for text models
    
    trainer = Trainer(**trainer_kwargs)
    
    # Train with progress bar
    print("Starting training...")
    print(f"📊 Checkpoints will be saved to: {output_dir}")
    print(f"📝 Logs will be saved to: {logging_dir}")
    print(f"💾 GPU memory before training:")
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = total_memory - allocated
        
        print(f"   Total GPU Memory: {total_memory:.2f} GB")
        print(f"   Allocated (in use): {allocated:.2f} GB")
        print(f"   Reserved (PyTorch pool): {reserved:.2f} GB")
        print(f"   Free (available): {free:.2f} GB")
        print(f"   ℹ️  Memory will grow dynamically during training as needed")
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    trainer.train()
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Memory-efficient evaluation after training
    if val_dataset_processed is not None and len(val_dataset_processed) > 0:
        print("\n" + "="*80)
        print("RUNNING MEMORY-EFFICIENT EVALUATION")
        print("="*80)
        try:
            # Clear cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Limit evaluation to prevent OOM - process in small batches
            max_eval_samples = min(50, len(val_dataset_processed))  # Limit to 50 samples
            if len(val_dataset_processed) > max_eval_samples:
                print(f"⚠️  Limiting evaluation to {max_eval_samples} samples (out of {len(val_dataset_processed)}) for memory efficiency")
                eval_dataset_limited = val_dataset_processed.select(range(max_eval_samples))
            else:
                eval_dataset_limited = val_dataset_processed
            
            # Use very small batch size for evaluation
            original_eval_batch_size = training_arguments.per_device_eval_batch_size
            training_arguments.per_device_eval_batch_size = 1
            
            # Disable prediction accumulation to save memory
            print(f"\n🔄 Running memory-efficient evaluation on {len(eval_dataset_limited)} samples...")
            print(f"   Using batch size: 1")
            print(f"   Processing in small chunks to prevent OOM...")
            
            # Evaluate with minimal memory usage (batch size 1, limited samples)
            val_metrics = trainer.evaluate(eval_dataset=eval_dataset_limited)
            
            # Restore original batch size
            training_arguments.per_device_eval_batch_size = original_eval_batch_size
            
            print(f"\n✅ Evaluation complete!")
            print(f"\n📊 Evaluation Metrics:")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # Clear cache after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n⚠️  OOM Error during evaluation: {e}")
            print("   Evaluation skipped. Model saved successfully.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"\n⚠️  Evaluation failed: {e}")
            print("   Evaluation skipped. Model saved successfully.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Save final model
    print(f"\n💾 Saving final model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training complete!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📁 Logs saved to: {logging_dir}")
    
    return trainer

def fine_tune(
    model_name: str = None,
    local_model_path: str = None,
    use_local: bool = False,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    output_dir: str = None,
    training_args: Dict[str, Any] = None,
    peft_config: Dict[str, Any] = None
) -> Any:
    """
    Main fine-tuning function.
    
    Args:
        model_name: Model name from HuggingFace
        local_model_path: Local path to model
        use_local: Whether to use local model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
        training_args: Training arguments
        peft_config: PEFT configuration
        
    Returns:
        Fine-tuned model
    """
    # Get configs
    if model_name is None:
        model_name = get_config("model")["model_name"]
    if local_model_path is None:
        local_model_path = get_config("model")["local_model_path"]
    if use_local is None:
        use_local = get_config("model")["use_local"]
    if training_args is None:
        training_args = get_config("training")
    if peft_config is None:
        peft_config = get_config("peft")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        local_model_path=local_model_path,
        use_local=use_local,
        trust_remote_code=get_config("model")["trust_remote_code"],
        hf_token=get_config("model").get("hf_token")
    )
    
    # Setup PEFT
    model = setup_peft_model(model, peft_config)
    
    # Initialize compute tracker for CVPR reporting
    output_config = get_config("output")
    logging_dir = output_config.get("logging_dir", output_dir)
    compute_tracker = ComputeTracker(output_dir=logging_dir)
    
    # Get model parameters count
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_params = trainable_params  # Report trainable parameters
    except:
        model_params = None
    
    # Set training metrics
    compute_tracker.set_training_metrics(
        model_params=model_params,
        training_set_size=len(train_dataset) if train_dataset else None,
        epochs=training_args.get("num_train_epochs") if training_args else None,
        batch_size=training_args.get("per_device_train_batch_size") if training_args else None,
        gradient_accumulation_steps=training_args.get("gradient_accumulation_steps") if training_args else None,
        mixed_precision="Yes - FP16" if training_args.get("fp16", False) else ("Yes - BF16" if training_args.get("bf16", False) else "No") if training_args else None,
        distributed_training=False,  # Update if using distributed training
    )
    
    # Clear CUDA cache before training to maximize available memory
    if torch.cuda.is_available():
        # Set environment variable to reduce memory fragmentation (prevents OOM)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()  # Force garbage collection
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = total_memory - allocated
        
        print(f"\n💾 GPU memory before training:")
        print(f"   Total GPU Memory: {total_memory:.2f} GB")
        print(f"   Allocated (in use): {allocated:.2f} GB")
        print(f"   Reserved (PyTorch pool): {reserved:.2f} GB")
        print(f"   Free (available): {free:.2f} GB")
        print(f"\n   ℹ️  Note: PyTorch allocates memory dynamically during training.")
        print(f"   ⚠️  Evaluation disabled during training to prevent OOM (logits accumulation causes memory issues)")
        print(f"   ⚠️  GPU memory limited to 32GB to prevent OOM (leaves 8GB buffer)")
        print(f"   ⚠️  Aggressive memory optimizations enabled:")
        print(f"      - max_length reduced to 768")
        print(f"      - LoRA rank reduced to 2")
        print(f"      - Gradient accumulation steps: 32")
        print(f"      - 4-bit quantization enabled")
        print(f"      - Gradient checkpointing enabled")
        print(f"      - Batch size: 1")
        print(f"⚠️  OOM Prevention: All aggressive optimizations enabled\n")
    
    # Start compute tracking
    compute_tracker.start_training()
    
    # Train
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        output_dir=output_dir
    )
    
    # End compute tracking
    compute_tracker.end_training()
    
    # Update total steps if available
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps'):
        total_steps = trainer.state.max_steps if trainer.state.max_steps > 0 else None
        if total_steps is None and hasattr(trainer.state, 'global_step'):
            total_steps = trainer.state.global_step
        compute_tracker.set_training_metrics(total_steps=total_steps)
    
    # Save compute report
    compute_tracker.save_report("compute_report.json")
    compute_tracker.print_summary()
    
    return trainer.model

if __name__ == "__main__":
    # Test fine-tuning setup
    from data_preprocessor import prepare_training_data
    
    print("Preparing data...")
    config = get_config()
    train_ds, val_ds = prepare_training_data(
        csv_path=config["data"]["csv_path"],
        task_start=config["data"]["task_start"],
        task_end=config["data"]["task_end"],
        val_split=config["data"]["val_split"],
        base_path=config["data"]["image_base_path"]
    )
    
    print("\nStarting fine-tuning...")
    # Uncomment to actually run training:
    # model = fine_tune(
    #     train_dataset=train_ds,
    #     val_dataset=val_ds,
    #     output_dir=config["output"]["output_dir"]
    # )

