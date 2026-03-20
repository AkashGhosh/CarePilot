#!/usr/bin/env python3
"""
Data preprocessing for fine-tuning Llama-4-Maverick on task results CSV.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config

def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load CSV file and return list of dictionaries.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of row dictionaries
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def filter_by_task_range(
    rows: List[Dict[str, Any]], 
    task_start: Optional[int] = None, 
    task_end: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter rows by Task_id range.
    
    Args:
        rows: List of row dictionaries
        task_start: Starting task ID (inclusive). If None, no lower bound.
        task_end: Ending task ID (inclusive). If None, no upper bound.
        
    Returns:
        Filtered rows. If both task_start and task_end are None, returns all rows.
    """
    if task_start is None and task_end is None:
        return rows
    
    filtered = []
    for row in rows:
        task_id = int(row['Task_id'])
        if task_start is not None and task_id < task_start:
            continue
        if task_end is not None and task_id > task_end:
            continue
        filtered.append(row)
    return filtered

def resolve_image_path(image_id: str, base_path: str = None, test_mode: bool = False) -> Optional[Path]:
    """
    Resolve image path from Image_id column.
    
    Args:
        image_id: Image path from CSV (can be relative or absolute)
        base_path: Base path for resolving relative paths
        test_mode: If True, try to resolve test images (task_results_test/images/)
        
    Returns:
        Path object or None if image not found
    """
    if not image_id or image_id.strip() == "":
        return None
    
    # Try as absolute path first
    img_path = Path(image_id)
    if img_path.is_absolute() and img_path.exists():
        return img_path
    
    # If test_mode and path references task_results/images/, try task_results_test/images/ instead
    if test_mode and "task_results/images/" in image_id:
        # Replace task_results/images/ with task_results_test/images/
        test_image_id = image_id.replace("task_results/images/", "task_results_test/images/")
        
        # Try with test base path
        if base_path:
            test_base = str(base_path).replace("task_results", "task_results_test")
            test_img_path = Path(test_base) / test_image_id
            if test_img_path.exists():
                return test_img_path
        
        # Try direct path replacement relative to project root
        from config import get_config
        project_root = Path(__file__).parent.parent
        test_img_path = project_root / test_image_id
        if test_img_path.exists():
            return test_img_path
        
        # Try extracting filename and looking in test_images directory
        filename = Path(image_id).name
        if base_path:
            test_base = Path(str(base_path).replace("task_results", "task_results_test"))
            test_img_path = test_base / "images" / filename
            if test_img_path.exists():
                return test_img_path
    
    # Try relative to base_path
    if base_path:
        # If image_id references task_results/images/, extract filename
        if "task_results/images/" in image_id or "task_results_test/images/" in image_id:
            filename = Path(image_id).name
            # Try in base_path directory (images are in finetuning/images/)
            img_path = Path(base_path) / filename
            if img_path.exists():
                return img_path
        
        # Try direct path
        img_path = Path(base_path) / image_id
        if img_path.exists():
            return img_path
        
        # Try with images subdirectory if it exists
        if (Path(base_path) / "images").exists():
            filename = Path(image_id).name
            img_path = Path(base_path) / "images" / filename
            if img_path.exists():
                return img_path
    
    # Try relative to CSV location
    img_path = Path(image_id)
    if img_path.exists():
        return img_path
    
    return None

def load_image(image_path: Path) -> Optional[Image.Image]:
    """
    Load image from path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None

def format_training_prompt(
    task_id: str,
    slide_number: str,
    grounding: str,
    short_term_memory: str,
    long_term_memory: str,
    image: Optional[Image.Image] = None,
    ablation_config: Optional[Dict[str, bool]] = None
) -> str:
    """
    Format training prompt from input components.
    
    Args:
        task_id: Task identifier
        slide_number: Step number
        grounding: Grounding JSON string
        short_term_memory: Short term memory JSON string
        long_term_memory: Long term memory JSON string
        image: PIL Image (will be handled separately)
        ablation_config: Ablation configuration dict with keys:
            - use_grounding: Include Grounding section (default: True)
            - use_short_term_memory: Include Short Term Memory section (default: True)
            - use_long_term_memory: Include Long Term Memory section (default: True)
        
    Returns:
        Formatted prompt string
    """
    # Get ablation config from config file if not provided
    if ablation_config is None:
        ablation_config = get_config("ablation")
    
    # Default to True if not specified
    use_grounding = ablation_config.get("use_grounding", True)
    use_short_term_memory = ablation_config.get("use_short_term_memory", True)
    use_long_term_memory = ablation_config.get("use_long_term_memory", True)
    
    prompt_parts = [
        "### Task Context",
        f"Task ID: {task_id}",
        f"Step Number: {slide_number}",
        "",
    ]
    
    # Add Grounding section if enabled
    if use_grounding:
        prompt_parts.extend([
            "### Grounding",
            grounding if grounding else "{}",
            "",
        ])
    
    # Add Short Term Memory section if enabled
    if use_short_term_memory:
        prompt_parts.extend([
            "### Short Term Memory",
            short_term_memory if short_term_memory else "{}",
            "",
        ])
    
    # Add Long Term Memory section if enabled
    if use_long_term_memory:
        prompt_parts.extend([
            "### Long Term Memory",
            long_term_memory if long_term_memory else "{}",
            "",
        ])
    
    prompt_parts.extend([
        "### Instruction",
        "Predict the complete next step output JSON based on the task context above.",
        "",
        "### Response",
    ])
    
    return "\n".join(prompt_parts)

def format_output(output: str) -> str:
    """
    Format output JSON string.
    
    Args:
        output: Output JSON string from CSV
        
    Returns:
        Formatted output string
    """
    # Try to parse and pretty-print JSON for consistency
    try:
        output_dict = json.loads(output)
        return json.dumps(output_dict, ensure_ascii=False)
    except:
        return output.strip()

def create_training_examples(
    rows: List[Dict[str, Any]],
    base_path: str = None,
    include_image: bool = True,
    ablation_config: Optional[Dict[str, bool]] = None
) -> List[Dict[str, Any]]:
    """
    Create training examples from CSV rows.
    
    Args:
        rows: List of CSV row dictionaries
        base_path: Base path for resolving image paths
        include_image: Whether to include images in examples
        
    Returns:
        List of training example dictionaries
    """
    examples = []
    
    for row in rows:
        # Extract fields
        task_id = row['Task_id']
        slide_number = row['Slide_number']
        grounding = row['Grounding'] or "{}"
        short_term_memory = row['Short_Term_Memory'] or "{}"
        long_term_memory = row['Long_term_Memory'] or "{}"
        image_id = row['Image_id']
        output = row['Output'] or "{}"
        
        # Format prompt with ablation config
        prompt = format_training_prompt(
            task_id=task_id,
            slide_number=slide_number,
            grounding=grounding,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            ablation_config=ablation_config
        )
        
        # Format output
        formatted_output = format_output(output)
        
        # Load image if needed
        image = None
        if include_image:
            image_path = resolve_image_path(image_id, base_path)
            if image_path:
                image = load_image(image_path)
        
        example = {
            "task_id": task_id,
            "slide_number": slide_number,
            "prompt": prompt,
            "output": formatted_output,
            "image": image,
            "image_path": str(image_path) if image_path else None,
        }
        
        examples.append(example)
    
    return examples

def create_hf_dataset(examples: List[Dict[str, Any]], split: str = None) -> Any:
    """
    Create HuggingFace Dataset from examples.
    
    Args:
        examples: List of training examples
        split: Optional split name ("train" or "val")
        
    Returns:
        HuggingFace Dataset
    """
    try:
        from datasets import Dataset
        
        # Prepare dataset dict
        dataset_dict = {
            "task_id": [ex["task_id"] for ex in examples],
            "slide_number": [ex["slide_number"] for ex in examples],
            "prompt": [ex["prompt"] for ex in examples],
            "output": [ex["output"] for ex in examples],
        }
        
        # Add images if available
        if examples[0].get("image") is not None:
            dataset_dict["image"] = [ex["image"] for ex in examples]
            dataset_dict["image_path"] = [ex.get("image_path") for ex in examples]
        
        dataset = Dataset.from_dict(dataset_dict)
        
        if split:
            # Add split info (not actual split, just metadata)
            dataset = dataset.add_column("split", [split] * len(examples))
        
        return dataset
    except ImportError:
        print("Warning: datasets library not available. Returning examples list.")
        return examples

def prepare_training_data(
    csv_path: str,
    task_start: Optional[int] = None,
    task_end: Optional[int] = None,
    val_split: float = 0.1,
    base_path: str = None,
    use_tqdm: bool = True,
    ablation_config: Optional[Dict[str, bool]] = None
) -> Tuple[Any, Any]:
    """
    Prepare training and validation datasets.
    
    IMPORTANT: For validation, memory columns are included for training purposes,
    but during actual evaluation, model should use memory from its own predictions.
    
    Args:
        csv_path: Path to CSV file
        task_start: Starting task ID (None = no lower bound)
        task_end: Ending task ID (None = no upper bound)
        val_split: Validation split ratio
        base_path: Base path for images
        use_tqdm: Show progress bar
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from tqdm import tqdm as tqdm_lib
    
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows")
    
    # Filter by task range
    if task_start is not None or task_end is not None:
        print(f"Filtering tasks {task_start or 'all'} to {task_end or 'all'}...")
        filtered_rows = filter_by_task_range(rows, task_start, task_end)
        print(f"Filtered to {len(filtered_rows)} rows")
    else:
        print("Using all tasks (no filtering)")
        filtered_rows = rows
    
    # Create training examples with progress bar
    print("Creating training examples...")
    if use_tqdm:
        examples = []
        for row in tqdm_lib(filtered_rows, desc="Creating examples", unit="row"):
            examples.extend(create_training_examples([row], base_path=base_path, ablation_config=ablation_config))
    else:
        examples = create_training_examples(filtered_rows, base_path=base_path, ablation_config=ablation_config)
    print(f"Created {len(examples)} examples")
    
    # Split into train/val
    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"Split: {len(train_examples)} train, {len(val_examples)} validation")
    print(f"⚠️  NOTE: Validation memory comes from CSV for training, but during evaluation")
    print(f"   model uses memory from its own predictions to prevent data leakage.")
    
    # Create HF datasets
    train_dataset = create_hf_dataset(train_examples, split="train")
    val_dataset = create_hf_dataset(val_examples, split="val")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Test data preprocessing
    config = get_config("data")
    train_ds, val_ds = prepare_training_data(
        csv_path=config["csv_path"],
        task_start=config["task_start"],
        task_end=config["task_end"],
        val_split=config["val_split"],
        base_path=config["image_base_path"]
    )
    
    print(f"\nTrain dataset: {train_ds}")
    print(f"Val dataset: {val_ds}")
    
    if hasattr(train_ds, "__len__"):
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")
        if len(train_ds) > 0:
            print(f"\nFirst example keys: {train_ds[0].keys()}")

