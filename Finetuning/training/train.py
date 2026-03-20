#!/usr/bin/env python3
"""
Main training script for fine-tuning Llama-4-Maverick on task results CSV.
"""

import os
import argparse
import sys
from pathlib import Path

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from data_preprocessor import prepare_training_data
from finetuning import fine_tune

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-4-Maverick on task results CSV"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to training CSV (default: task_results/task_results.csv)"
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=None,
        help="Starting task ID for training (default: None = use all tasks)"
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=None,
        help="Ending task ID for training (default: None = use all tasks)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to Llama-4-Maverick model (default: load from HuggingFace)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name from HuggingFace (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct)"
    )
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local model instead of HuggingFace"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save fine-tuned model (default: finetuning/outputs)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per device batch size (default: 1)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps (default: 100)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log every N steps (default: 10)"
    )
    
    # Ablation study arguments
    parser.add_argument(
        "--no_grounding",
        action="store_true",
        help="Remove Grounding from training/inference (ablation study)"
    )
    parser.add_argument(
        "--no_short_term_memory",
        action="store_true",
        help="Remove Short Term Memory from training/inference (ablation study)"
    )
    parser.add_argument(
        "--no_long_term_memory",
        action="store_true",
        help="Remove Long Term Memory from training/inference (ablation study)"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Get default config
    config = get_config()
    
    # Resolve CSV path
    csv_path = args.train_csv
    if csv_path is None:
        csv_path = config["data"]["csv_path"]
    
    # Setup ablation config (need to do this early to modify output_dir)
    ablation_config = config["ablation"].copy()
    if args.no_grounding:
        ablation_config["use_grounding"] = False
    if args.no_short_term_memory:
        ablation_config["use_short_term_memory"] = False
    if args.no_long_term_memory:
        ablation_config["use_long_term_memory"] = False
    
    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config["output"]["output_dir"]
    
    # Append ablation suffix to output directory for tracking
    ablation_suffix_parts = []
    if not ablation_config["use_grounding"]:
        ablation_suffix_parts.append("no_grounding")
    if not ablation_config["use_short_term_memory"]:
        ablation_suffix_parts.append("no_stm")
    if not ablation_config["use_long_term_memory"]:
        ablation_suffix_parts.append("no_ltm")
    
    if ablation_suffix_parts:
        ablation_suffix = "_".join(ablation_suffix_parts)
        output_dir = f"{output_dir}_{ablation_suffix}"
    
    # Resolve model configuration
    model_name = args.model_name or config["model"]["model_name"]
    local_model_path = args.model_path or config["model"]["local_model_path"]
    use_local = args.use_local or config["model"]["use_local"]
    
    print("=" * 60)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 60)
    print(f"CSV Path: {csv_path}")
    if args.task_start is not None and args.task_end is not None:
        print(f"Task Range: {args.task_start} to {args.task_end}")
    else:
        print(f"Task Range: All tasks (no filtering)")
    print(f"Validation Split: {args.val_split}")
    print(f"Model: {model_name}")
    print(f"Use Local: {use_local}")
    if use_local:
        print(f"Local Model Path: {local_model_path}")
    print(f"Output Directory: {output_dir}")
    
    # Print ablation settings
    print("\nAblation Settings:")
    print(f"  Use Grounding: {ablation_config['use_grounding']}")
    print(f"  Use Short Term Memory: {ablation_config['use_short_term_memory']}")
    print(f"  Use Long Term Memory: {ablation_config['use_long_term_memory']}")
    print("=" * 60)
    
    # Prepare training data
    # Use config defaults if not specified via command line
    task_start = args.task_start if args.task_start is not None else config["data"]["task_start"]
    task_end = args.task_end if args.task_end is not None else config["data"]["task_end"]
    
    print("\nPreparing training data...")
    train_dataset, val_dataset = prepare_training_data(
        csv_path=csv_path,
        task_start=task_start,
        task_end=task_end,
        val_split=args.val_split,
        base_path=config["data"]["image_base_path"],
        ablation_config=ablation_config
    )
    
    print(f"\nTraining dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Prepare training arguments
    training_args = config["training"].copy()
    if args.learning_rate is not None:
        training_args["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        training_args["num_train_epochs"] = args.num_epochs
    if args.batch_size is not None:
        training_args["per_device_train_batch_size"] = args.batch_size
        training_args["per_device_eval_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        training_args["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.warmup_steps is not None:
        training_args["warmup_steps"] = args.warmup_steps
    if args.save_steps is not None:
        training_args["save_steps"] = args.save_steps
    if args.logging_steps is not None:
        training_args["logging_steps"] = args.logging_steps
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    model = fine_tune(
        model_name=model_name,
        local_model_path=local_model_path,
        use_local=use_local,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        training_args=training_args,
        peft_config=config["peft"]
    )
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print("\nTo evaluate the model, run:")
    print(f"  python finetuning/evaluate.py --model_path {output_dir}")

if __name__ == "__main__":
    main()

