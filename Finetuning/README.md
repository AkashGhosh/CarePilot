# Finetuning

Supervised fine-tuning (SFT) code for training vision-language models on task results.

## Overview

The finetuning component provides:
- **Training**: Supervised fine-tuning using LoRA (Low-Rank Adaptation)
- **Evaluation**: Model evaluation on test datasets
- **Inference**: Sequential step prediction with memory propagation

## Structure

```
Finetuning/
├── training/           # Training scripts
│   ├── train.py       # Main training script
│   ├── finetuning.py # Fine-tuning implementation
│   ├── data_preprocessor.py # Data preprocessing
│   ├── config.py      # Training configuration
│   └── compute_tracker.py # Compute usage tracking
└── test/              # Evaluation and testing
    ├── evaluate.py    # Model evaluation
    ├── inference.py   # Inference with memory
    └── test_data_loader.py # Test data loading
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
```

3. Install PyTorch (based on your CUDA version):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU
pip install torch torchvision torchaudio
```

## Configuration

Edit `training/config.py` to configure:
- Model name and path
- Training hyperparameters
- PEFT (LoRA) configuration
- Data paths
- Output directories

## Usage

### Training

```bash
cd training
python train.py \
    --train_csv path/to/task_results.csv \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --output_dir outputs/model_name
```

### Training Arguments

- `--train_csv`: Path to training CSV (default: task_results.csv)
- `--task_start`: Starting task ID for training
- `--task_end`: Ending task ID for training
- `--val_split`: Validation split ratio (default: 0.1)
- `--model_path`: Local path to model
- `--model_name`: Model name from HuggingFace
- `--use_local`: Use local model instead of HuggingFace
- `--output_dir`: Output directory for checkpoints
- `--learning_rate`: Learning rate (default: 2e-4)
- `--num_epochs`: Number of training epochs (default: 2)
- `--batch_size`: Batch size (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 32)

### Evaluation

```bash
cd test
python evaluate.py \
    --model_path ../training/outputs/model_name \
    --test_csv path/to/test_results.csv \
    --output_dir results/
```

### Evaluation Arguments

- `--model_path`: Path to fine-tuned model
- `--test_csv`: Path to test CSV
- `--output_dir`: Output directory for results
- `--batch_size`: Batch size for evaluation
- `--max_tasks`: Maximum number of tasks to evaluate

### Inference

```python
from inference import load_model_for_inference

# Load model
predictor = load_model_for_inference(
    model_path="outputs/model_name",
    base_model_name="Qwen/Qwen3-VL-8B-Instruct"
)

# Predict next step
prediction = predictor.predict_step(
    image_path="path/to/image.png",
    user_goal="Load the MRI scan",
    short_term_memory={},
    long_term_memory={}
)
```

## Data Format

The training CSV should contain the following columns:
- `Task_id`: Task identifier
- `Image_id`: Image path or identifier
- `Instruction`: Task instruction
- `Output`: Ground truth output JSON
- `Grounding`: Grounding information JSON

## Model Support

Currently supports:
- Qwen3-VL-8B-Instruct
- Qwen2.5-VL models
- Llama-4-Maverick (with modifications)

## Memory Optimization

The training configuration is optimized for 40GB GPUs:
- 4-bit quantization enabled
- Gradient checkpointing
- Reduced batch size with gradient accumulation
- Optimized sequence length

## Output

Training outputs:
- Model checkpoints in `output_dir`
- Training logs in `logs/`
- Compute usage reports for CVPR reporting

Evaluation outputs:
- Predictions CSV
- Evaluation metrics
- Detailed results JSON

