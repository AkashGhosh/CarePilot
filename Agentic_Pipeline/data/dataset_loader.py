#!/usr/bin/env python3
"""
Dataset loader module for handling HuggingFace datasets and image processing.
"""

import io
import os
from typing import Dict, List, Optional, Union, Any

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from PIL import Image

# Default HuggingFace dataset and token
HF_DATASET_NAME = "rishuKumar404/3dslicer-tabular-benchmark"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set HF_TOKEN environment variable

def load_hf_dataset(
    dataset_name: str = HF_DATASET_NAME, 
    token: str = HF_TOKEN, 
    split: Optional[str] = None,
    streaming: bool = False,
    max_tasks: Optional[int] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load the HuggingFace dataset with authentication.
    Uses streaming by default to avoid loading entire dataset into memory.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace authentication token
        split: Dataset split to load (e.g., 'train', 'test'). If None, loads all splits.
        streaming: If True, use streaming dataset (memory efficient). If False, load full dataset.
        max_tasks: If provided, only load this many examples (requires streaming=True)
    
    Returns:
        Dataset or DatasetDict or IterableDataset depending on streaming parameter
    """
    try:
        # Authenticate with HuggingFace
        login(token=token)
        
        # Load dataset (streaming disabled by default for simplicity)
        if streaming:
            # Only use streaming if explicitly requested
            try:
                if split:
                    dataset = load_dataset(dataset_name, split=split, token=token, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split='train', token=token, streaming=True)
                return dataset
            except TypeError:
                if split:
                    dataset = load_dataset(dataset_name, split=split, use_auth_token=token, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split='train', use_auth_token=token, streaming=True)
                return dataset
        else:
            # Non-streaming mode (loads everything into memory - use with caution!)
            try:
                if split:
                    dataset = load_dataset(dataset_name, split=split, token=token)
                else:
                    dataset = load_dataset(dataset_name, token=token)
                
                # Limit number of examples if specified
                if max_tasks:
                    if isinstance(dataset, DatasetDict):
                        if 'train' in dataset:
                            dataset = DatasetDict({'train': dataset['train'].select(range(min(len(dataset['train']), max_tasks)))})
                        else:
                            first_split = next(iter(dataset.keys()))
                            dataset = DatasetDict({first_split: dataset[first_split].select(range(min(len(dataset[first_split]), max_tasks)))})
                    else:
                        dataset = dataset.select(range(min(len(dataset), max_tasks)))
                
                # DO NOT process all images at once - process on-demand instead
                # processed_dataset = process_dataset_images(dataset)
                return dataset
                
            except TypeError:
                # Fallback to 'use_auth_token' parameter (older API)
                if split:
                    dataset = load_dataset(dataset_name, split=split, use_auth_token=token)
                else:
                    dataset = load_dataset(dataset_name, use_auth_token=token)
                
                if max_tasks:
                    if isinstance(dataset, DatasetDict):
                        if 'train' in dataset:
                            dataset = DatasetDict({'train': dataset['train'].select(range(min(len(dataset['train']), max_tasks)))})
                        else:
                            first_split = next(iter(dataset.keys()))
                            dataset = DatasetDict({first_split: dataset[first_split].select(range(min(len(dataset[first_split]), max_tasks)))})
                    else:
                        dataset = dataset.select(range(min(len(dataset), max_tasks)))
                
                return dataset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}") from e

def process_dataset_images(dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    """
    Process dataset to convert binary image data to PIL Images.
    
    WARNING: This processes ALL images in the dataset at once and can cause memory issues!
    For memory-efficient processing, use convert_example_images() on individual examples instead.
    
    Args:
        dataset: Input dataset that may contain binary image data
    
    Returns:
        Processed dataset with PIL Images
    """
    # This function is kept for backward compatibility but should be avoided for large datasets
    # Instead, use convert_example_images() on individual examples
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: split_dataset.map(convert_to_pil_images)
            for split, split_dataset in dataset.items()
        })
    else:
        return dataset.map(convert_to_pil_images)

def convert_to_pil_images(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert binary image data to PIL Images for a single example.
    
    Args:
        example: Dataset example with binary image data
    
    Returns:
        Processed example with PIL Images
    """
    # Make a copy of the example to avoid modifying the original
    processed_example = dict(example)
    
    # Check if 'images' field exists and is a list
    if 'images' in processed_example and isinstance(processed_example['images'], list):
        # Skip processing if the first item is already a dict or PIL Image
        if len(processed_example['images']) > 0 and (isinstance(processed_example['images'][0], dict) or 
                                                  hasattr(processed_example['images'][0], 'size')):
            return processed_example
            
        pil_images = []
        
        for img_binary in processed_example['images']:
            try:
                # Check if the binary data is already a dict (happens in some cases)
                if isinstance(img_binary, dict):
                    # If it's already processed somehow, keep it as is
                    pil_images.append(img_binary)
                else:
                    # Convert binary data to PIL Image
                    img = Image.open(io.BytesIO(img_binary))
                    pil_images.append(img)
            except Exception as e:
                print(f"Error converting image: {e}")
        
        # Replace binary data with PIL Images
        processed_example['images'] = pil_images
    
    return processed_example

def convert_example_images(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert binary image data to PIL Images for a single example (memory-efficient version).
    This processes images on-demand, one example at a time.
    
    Args:
        example: Dataset example with binary image data
    
    Returns:
        Processed example with PIL Images
    """
    # This is an alias for convert_to_pil_images but with clearer naming for on-demand processing
    return convert_to_pil_images(example)

def preprocess_task_context(task_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess task context to ensure it contains required fields.
    
    Args:
        task_context: Task context data
    
    Returns:
        Processed task context
    """
    if task_context is None:
        return {'instruction': 'Complete the task.', 'images': []}
    
    # Ensure 'images' field exists
    if 'images' not in task_context:
        task_context['images'] = []
        
        # Check if there's image data in another field
        if 'image_data' in task_context:
            # Process image data based on its type
            if isinstance(task_context['image_data'], list):
                for img_data in task_context['image_data']:
                    if isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data))
                        task_context['images'].append(img)
    
    return task_context

def get_dataset_example(dataset: Union[Dataset, DatasetDict], index: int = 0, split: str = 'train') -> Dict[str, Any]:
    """
    Get a specific example from the dataset.
    
    Args:
        dataset: Dataset to get example from
        index: Index of the example to get
        split: Split to get example from (if dataset is a DatasetDict)
    
    Returns:
        Example from the dataset
    """
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
        example = dataset[split][index]
    else:
        example = dataset[index]
    
    return example

def create_fixed_task_context(
    dataset: Union[Dataset, DatasetDict, Any], 
    index: int = 0, 
    split: str = 'train',
    example: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a fixed task context from a dataset example to ensure it has proper PIL images.
    Memory-efficient: processes images on-demand.
    
    Args:
        dataset: Dataset to get example from (optional if example is provided)
        index: Index of the example to get (if dataset is provided)
        split: Split to get example from (if dataset is a DatasetDict)
        example: Optional pre-fetched example (avoids dataset access if provided)
    
    Returns:
        Task context with proper PIL images
    """
    # Use provided example or fetch from dataset
    if example is None:
        # Check if dataset is a streaming iterable
        if hasattr(dataset, '__iter__') and not isinstance(dataset, (Dataset, DatasetDict)):
            # Streaming dataset - take the nth item
            import itertools
            example = next(itertools.islice(dataset, index, None))
        elif isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            example = dataset[split][index]
        else:
            example = dataset[index]
    
    # Create task context
    task_context = {
        'instruction': example.get('instruction', 'Complete the task.'),
        'images': []
    }
    
    # Convert images to PIL images
    if 'images' in example and example['images']:
        for i, img_data in enumerate(example['images']):
            try:
                if isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                    task_context['images'].append(img)
                elif hasattr(img_data, 'size'):  # Already a PIL image
                    task_context['images'].append(img_data)
                elif isinstance(img_data, dict) and 'path' in img_data:  # Image path in dict
                    img = Image.open(img_data['path'])
                    task_context['images'].append(img)
                else:
                    # Try to handle other formats
                    task_context['images'].append(img_data)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
    
    # Add other fields from the example
    if 'task_id' in example:
        task_context['task_id'] = example['task_id']
    if 'json_data' in example:
        task_context['json_data'] = example['json_data']
    if 'num_steps' in example:
        task_context['num_steps'] = example['num_steps']
    if 'num_images' in example:
        task_context['num_images'] = example['num_images']
    
    return task_context

# Test function to verify dataset loading and image processing
def test_dataset_loading(dataset_name: str = HF_DATASET_NAME, token: str = HF_TOKEN) -> None:
    """
    Test function to verify dataset loading and image processing.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace authentication token
    """
    try:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_hf_dataset(dataset_name, token)
        
        print("\nDataset loaded successfully.")
        if isinstance(dataset, DatasetDict):
            print(f"Dataset splits: {list(dataset.keys())}")
            for split, split_dataset in dataset.items():
                print(f"Split '{split}': {len(split_dataset)} examples")
                if len(split_dataset) > 0:
                    example = split_dataset[0]
                    if 'images' in example and example['images']:
                        print(f"First example has {len(example['images'])} images")
                        print(f"First image type: {type(example['images'][0])}")
                    else:
                        print("No images found in first example")
        else:
            print(f"Dataset has {len(dataset)} examples")
            if len(dataset) > 0:
                example = dataset[0]
                if 'images' in example and example['images']:
                    print(f"First example has {len(example['images'])} images")
                    print(f"First image type: {type(example['images'][0])}")
                    # Verify image content more deeply
                    first_img = example['images'][0]
                    if isinstance(first_img, dict):
                        print(f"Image dict keys: {first_img.keys() if hasattr(first_img, 'keys') else 'No keys'}")
                    elif hasattr(first_img, 'size'):
                        print(f"Image size: {first_img.size}")
                else:
                    print("No images found in first example")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error testing dataset loading: {e}")

if __name__ == "__main__":
    import sys
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else HF_DATASET_NAME
    token = sys.argv[2] if len(sys.argv) > 2 else HF_TOKEN
    
    test_dataset_loading(dataset_name, token)
