#!/usr/bin/env python3
"""
Test dataset loader for fine-tuning evaluation.
Loads test dataset (tasks 1-47) for sequential evaluation.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from config import get_config
from data_preprocessor import load_csv, filter_by_task_range, resolve_image_path

class TestDataLoader:
    """
    Loader for test dataset.
    """
    
    def __init__(
        self,
        csv_path: str,
        task_start: Optional[int] = None,
        task_end: Optional[int] = None,
        image_base_path: Optional[str] = None,
        test_mode: bool = True
    ):
        """
        Initialize test data loader.
        
        Args:
            csv_path: Path to CSV file
            task_start: Starting task ID (inclusive). None = process all tasks
            task_end: Ending task ID (inclusive). None = process all tasks
            image_base_path: Base path for images
            test_mode: If True, use test image resolution
        """
        self.csv_path = csv_path
        self.task_start = task_start
        self.task_end = task_end
        self.image_base_path = image_base_path
        self.test_mode = test_mode
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and process CSV data."""
        print(f"Loading test data from {self.csv_path}...")
        rows = load_csv(self.csv_path)
        print(f"Loaded {len(rows)} rows")
        
        # Filter by task range
        if self.task_start is None and self.task_end is None:
            print("Processing all tasks (no task range specified)")
            filtered_rows = rows
        else:
            print(f"Filtering tasks {self.task_start or 'all'} to {self.task_end or 'all'}...")
            filtered_rows = filter_by_task_range(rows, self.task_start, self.task_end)
        print(f"Filtered to {len(filtered_rows)} rows")
        
        # Group by Task_id
        self.tasks = defaultdict(list)
        for row in filtered_rows:
            task_id = row['Task_id']
            self.tasks[task_id].append(row)
        
        # Sort steps within each task by Slide_number
        for task_id in self.tasks:
            self.tasks[task_id].sort(key=lambda x: int(x['Slide_number']))
        
        print(f"Grouped into {len(self.tasks)} tasks")
        print(f"Task IDs: {sorted([int(t) for t in self.tasks.keys()])}")
    
    def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get steps for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of step dictionaries
        """
        if task_id not in self.tasks:
            return []
        
        steps = []
        for row in self.tasks[task_id]:
            image_id = row['Image_id']
            
            # Resolve image path if needed
            image_path = None
            if image_id:
                image_path = resolve_image_path(
                    image_id,
                    base_path=self.image_base_path,
                    test_mode=self.test_mode
                )
                if image_path:
                    image_id = str(image_path)
            
            step = {
                "task_id": row['Task_id'],
                "slide_number": row['Slide_number'],
                "grounding": row['Grounding'] or "{}",
                "image_id": image_id,
                # Ground truth output for evaluation ONLY (never shown to model)
                "ground_truth_output": row.get('Output') or "",
                # Also include grounding for fallback extraction
                "grounding_json": row['Grounding'] or "{}",
                # NOTE: Short_Term_Memory and Long_term_Memory from CSV are IGNORED
                # Memory comes from model predictions, not from test CSV (prevents data leakage)
            }
            steps.append(step)
        
        return steps
    
    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all tasks.
        
        Returns:
            Dictionary mapping task_id to list of steps
        """
        all_tasks = {}
        for task_id in self.tasks:
            all_tasks[task_id] = self.get_task_steps(task_id)
        return all_tasks
    
    def get_task_ids(self) -> List[str]:
        """
        Get list of all task IDs.
        
        Returns:
            List of task IDs
        """
        return sorted(list(self.tasks.keys()), key=int)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_tasks": len(self.tasks),
            "total_steps": sum(len(steps) for steps in self.tasks.values()),
            "avg_steps_per_task": sum(len(steps) for steps in self.tasks.values()) / len(self.tasks) if self.tasks else 0,
            "min_steps": min(len(steps) for steps in self.tasks.values()) if self.tasks else 0,
            "max_steps": max(len(steps) for steps in self.tasks.values()) if self.tasks else 0,
        }
        return stats

def load_test_dataset(
    csv_path: str = None,
    task_start: Optional[int] = None,
    task_end: Optional[int] = None,
    image_base_path: str = None,
    test_mode: bool = True
) -> TestDataLoader:
    """
    Load test dataset.
    
    Args:
        csv_path: Path to CSV file (default: test_csv_path from config)
        task_start: Starting task ID (None = process all tasks)
        task_end: Ending task ID (None = process all tasks)
        image_base_path: Base path for images (default: test_image_base_path from config)
        test_mode: If True, use test image resolution
        
    Returns:
        TestDataLoader instance
    """
    config = get_config("data")
    
    if csv_path is None:
        csv_path = config.get("test_csv_path") or config["csv_path"]
    
    if image_base_path is None:
        image_base_path = config.get("test_image_base_path") or config["image_base_path"]
    
    return TestDataLoader(
        csv_path=csv_path,
        task_start=task_start,
        task_end=task_end,
        image_base_path=image_base_path,
        test_mode=test_mode
    )

if __name__ == "__main__":
    # Test data loader
    config = get_config("data")
    
    loader = load_test_dataset(
        csv_path=config.get("test_csv_path") or config["csv_path"],
        task_start=config["test_task_start"],
        task_end=config["test_task_end"],
        image_base_path=config.get("test_image_base_path"),
        test_mode=True
    )
    
    print("\nDataset statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample task (task_id=1):")
    steps = loader.get_task_steps("1")
    print(f"  Number of steps: {len(steps)}")
    if steps:
        print(f"  First step: slide_number={steps[0]['slide_number']}, has_image={bool(steps[0]['image_id'])}")

