#!/usr/bin/env python3
"""
Main entry point for the Medical Visual Agent system.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional, Union

# Setup paths before any other imports
import _setup_paths  # noqa: E402

# Now we can import using absolute imports from project root
from config import get_config, update_config
from data.dataset_loader import load_hf_dataset, process_dataset_images, create_fixed_task_context
from utils.common import safe_json_loads, encode_image_to_data_uri
from controllers.task_controller import TaskController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("medical_visual_agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical Visual Agent System")
    
    parser.add_argument(
        "--mode",
        choices=["original", "enhanced", "revpt", "integration", "dataset"],
        default="dataset",
        help="Implementation mode to use"
    )
    
    parser.add_argument(
        "--goal",
        type=str,
        default="Load the MRI scan, create a segmentation of the tumor, and measure its volume.",
        help="User goal/task to execute"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock implementations for tools"
    )
    
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=2,
        help="Maximum number of tasks to process from dataset"
    )
    
    parser.add_argument(
        "--start_task",
        type=int,
        default=None,
        help="Starting task index (0-indexed). If specified, starts from this task number. Use with --max_tasks to specify a range."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name to use (overrides config)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (overrides config)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-tools",
        action="store_true",
        default=False,
        help="Disable tool calls (visual grounding, OCR, etc.) - for faster testing without tool overhead"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment based on command line arguments."""
    # Update configuration based on command line arguments
    if args.dataset:
        update_config("dataset", {"hf_dataset_name": args.dataset})
    
    if args.token:
        update_config("dataset", {"hf_token": args.token})
    
    if args.verbose:
        update_config("system", {"verbose": True})
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create cache directory if needed
    cache_dir = get_config("dataset")["dataset_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

def load_dataset_and_process(args):
    """Load and process the dataset."""
    logger.info("Loading dataset...")
    
    config = get_config("dataset")
    dataset_name = config["hf_dataset_name"]
    token = config["hf_token"]
    
    try:
        # Load dataset - limit is handled in process_tasks
        dataset = load_hf_dataset(dataset_name, token, split='train', streaming=False)
        logger.info(f"Dataset loaded successfully")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

def run_mode(args, dataset=None):
    """Run the system in the specified mode."""
    # Create task controller
    controller = TaskController(
        use_mock=args.mock,
        use_enhanced=(args.mode in ["enhanced", "integration"]),
        enable_tools=not args.no_tools  # Enable tools unless --no-tools is specified
    )
    
    if args.mode == "dataset" and dataset is not None:
        if args.start_task is not None:
            logger.info(f"Running in dataset mode from task {args.start_task} with {args.max_tasks} tasks...")
        else:
            logger.info(f"Running in dataset mode with {args.max_tasks} tasks...")
        
        # Process tasks from dataset
        results = controller.process_tasks(
            max_tasks=args.max_tasks,  # Limits to specified number of tasks
            dataset=dataset,
            start_task=args.start_task  # Starting task index (0-indexed)
        )
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETED {len(results)} TASK(S)")
        logger.info(f"{'='*60}")
        
        for i, result in enumerate(results):
            logger.info(f"\nTask {i+1}: {result.get('task_instruction', 'N/A')}")
            logger.info(f"  Task ID: {result.get('task_index', 'N/A')}")
            logger.info(f"  Finished: {result.get('finished', False)}")
            logger.info(f"  Steps taken: {result.get('steps_taken', 0)}")
            logger.info(f"  Total retries: {result.get('total_retries', 0)}")
            
            if result.get('finished'):
                logger.info("  Status: ✅ COMPLETED")
            else:
                logger.info("  Status: ❌ INCOMPLETE")
        
        return results
        
    elif args.mode == "original":
        logger.info("Running with original implementation...")
        # Use single goal mode
        result = controller.process_single_task(args.goal)
        logger.info(f"\nTask completed: {result.get('finished', False)}")
        return [result]
        
    elif args.mode == "enhanced":
        logger.info("Running with enhanced components...")
        result = controller.process_single_task(args.goal)
        logger.info(f"\nTask completed: {result.get('finished', False)}")
        return [result]
        
    elif args.mode == "revpt":
        logger.info("Running with REVPT medical tools...")
        result = controller.process_single_task(args.goal)
        logger.info(f"\nTask completed: {result.get('finished', False)}")
        return [result]
        
    else:  # integration
        logger.info("Running with integration module...")
        result = controller.process_single_task(args.goal)
        logger.info(f"\nTask completed: {result.get('finished', False)}")
        return [result]

def main():
    """Main entry point."""
    args = parse_arguments()
    setup_environment(args)
    
    logger.info("Starting Medical Visual Agent System...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Goal: {args.goal}")
    
    # Load dataset if needed
    dataset = None
    if args.mode == "dataset":
        dataset = load_dataset_and_process(args)
    
    # Run the system in the specified mode
    run_mode(args, dataset)
    
    logger.info("Execution completed.")

if __name__ == "__main__":
    main()
