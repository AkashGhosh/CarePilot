#!/usr/bin/env python3
"""
Task controller module for the Medical Visual Agent system.
This module orchestrates the task processing pipeline.
"""

import json
import logging
import re
import time
import os
import sys
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from tqdm import tqdm

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    # Fallback: add project root to path
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

# Use absolute imports from project root
from config import get_config
from agents.target_agent import TargetAgent, run_target_agent
from agents.feedback_agent import FeedbackAgent, run_critic_agent
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.memory_manager import MemoryManager
from tools.tool_executor import ToolExecutor
from tools.visual_tools import VisualTools
from tools.medical_tools import MedicalTools
from evaluation.action_verifier import ActionVerifier
from evaluation.reflection import ReflectionProcessor
from utils.common import safe_json_loads, encode_image_to_data_uri, convert_bbox_to_center_point
from utils.model_client import get_model_client
from data.dataset_loader import load_hf_dataset, create_fixed_task_context
from datasets import Dataset, DatasetDict
from utils.logging_utils import (
    setup_logging, format_separator, format_step_header, 
    format_task_header, format_progress, format_json_compact, 
    format_decision
)

# Set up colored logging
logger = setup_logging()

class TaskController:
    """
    Task controller that orchestrates the task processing pipeline.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        use_mock: bool = True,
        use_enhanced: bool = True,
        enable_tools: bool = True
    ):
        """
        Initialize the task controller.
        
        Args:
            config: Configuration dictionary
            use_mock: Whether to use mock implementations for tools
            use_enhanced: Whether to use enhanced components
            enable_tools: Whether to execute tool calls (visual grounding, OCR, etc.)
        """
        self.config = config or get_config()
        self.system_config = self.config.get("system", {})
        self.use_mock = use_mock
        self.use_enhanced = use_enhanced
        self.enable_tools = enable_tools  # Store the flag for tool execution control
        
        # Load thresholds
        thresholds = self.config.get("thresholds", {})
        self.action_accept_threshold = thresholds.get("action_accept_threshold", 0.80)
        self.action_fail_threshold = thresholds.get("action_fail_threshold", 0.40)
        self.semantic_min_threshold = thresholds.get("semantic_min", 0.60)
        self.terminate_confidence = thresholds.get("terminate_confidence", 0.90)
        
        # Maximum steps and retries
        self.max_total_steps = self.system_config.get("max_total_steps", 15)
        self.max_retries_per_step = self.system_config.get("max_retries_per_step", 3)
        
        # Initialize components
        self.tool_executor = ToolExecutor(use_mock=self.use_mock)
        self.visual_tools = VisualTools(self.tool_executor)
        self.medical_tools = MedicalTools(self.visual_tools)
        self.target_agent = TargetAgent()
        self.feedback_agent = FeedbackAgent()
        self.action_verifier = ActionVerifier()
        self.reflection_processor = ReflectionProcessor()
        
        # Enhanced components for tool selection (if available)
        self.tool_selector = None
        self.tool_processor = None
        if self.use_enhanced:
            self._init_enhanced_components()
        
        # Initialize API client for model inference
        try:
            self.api_client = get_model_client()
            logger.info("Initialized Deep Infra API client for model inference")
        except Exception as e:
            logger.warning(f"Failed to initialize API client: {e}. Will use mock responses.")
            self.api_client = None
    
    def _init_enhanced_components(self):
        """Initialize enhanced components for tool selection and processing."""
        try:
            # This would import from the appropriate modules in a real implementation
            # from ..tools.reinforced_tool_selection import ReinforcedToolSelector
            # from ..tools.confidence_weighted_results import ConfidenceWeightedToolResults
            
            # For now, we'll just log that we would initialize these components
            logger.info("Would initialize enhanced components for tool selection")
            
            # Mock implementation
            class MockToolSelector:
                def select_tools(self, query, available_tools, step_num):
                    return ["object_detection", "visual_grounding"]
                
                def update_reward(self, tool_name, reward):
                    pass
            
            class MockToolProcessor:
                def __init__(self, selector):
                    self.selector = selector
                
                def process_results(self, results, query):
                    return results
            
            self.tool_selector = MockToolSelector()
            self.tool_processor = MockToolProcessor(self.tool_selector)
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
    
    def process_tasks(
        self,
        user_goal: Optional[str] = None,
        max_tasks: Optional[int] = None,
        dataset = None,
        start_task: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple tasks from a dataset or a single user goal.
        
        Args:
            user_goal: Single task instruction/goal
            max_tasks: Maximum number of tasks to process
            dataset: HuggingFace dataset
            start_task: Starting task index (0-indexed). If None, starts from 0.
        
        Returns:
            List of task results
        """
        # Load tasks from dataset if provided
        logger.info(format_separator(" DATASET LOADING "))
        tasks = []
        
        if dataset is not None:
            # Get the right split
            if isinstance(dataset, DatasetDict):
                if 'train' in dataset:
                    dataset = dataset['train']
                    logger.info("Using 'train' split from dataset")
                else:
                    first_split = next(iter(dataset.keys()))
                    dataset = dataset[first_split]
                    logger.info(f"Using '{first_split}' split from dataset")
            
            original_size = len(dataset)
            
            # Determine the range of tasks to select
            start_idx = start_task if start_task is not None else 0
            start_idx = max(0, start_idx)  # Ensure non-negative
            
            if max_tasks:
                # Select tasks from start_idx to start_idx + max_tasks
                end_idx = min(original_size, start_idx + max_tasks)
                dataset = dataset.select(range(start_idx, end_idx))
                logger.info(f"PROGRESS Selecting tasks {start_idx} to {end_idx-1} (total: {len(dataset)} tasks from {original_size} total)")
            elif start_task is not None:
                # If only start_task is specified, select from start to end
                dataset = dataset.select(range(start_idx, original_size))
                logger.info(f"PROGRESS Selecting tasks {start_idx} to {original_size-1} (total: {len(dataset)} tasks from {original_size} total)")
            
            # Process each example
            logger.info("Processing dataset examples...")
            for i, example in enumerate(dataset):
                # Convert images on-demand
                from data.dataset_loader import convert_example_images
                processed_example = convert_example_images(example)
                
                # Calculate the actual index in the original dataset
                actual_index = start_idx + i
                
                # Handle different dataset formats
                # OpenHospital format: uses 'task' instead of 'instruction', 'task_number' instead of 'task_id'
                instruction = processed_example.get("instruction") or processed_example.get("task", "Complete the task")
                
                # Generate task_id from task_number if not present (OpenHospital format)
                task_id = processed_example.get("task_id")
                if not task_id and "task_number" in processed_example:
                    task_number = processed_example.get("task_number", actual_index)
                    # Format: openhospital_endtoend_XXX or openhospital_task_XXX
                    task_id = f"openhospital_endtoend_{task_number:03d}"
                elif not task_id:
                    task_id = f"task_{actual_index}"
                
                task = {
                    "instruction": instruction,
                    "task_id": task_id,
                    "images": processed_example.get("images", []),
                    "json_data": processed_example.get("json_data", "{}"),
                    "num_steps": processed_example.get("num_steps", 0),
                    "num_images": processed_example.get("num_images", 0)
                }
                tasks.append(task)
                
                # Show progress for large datasets
                if i > 0 and i % 5 == 0 and len(dataset) > 10:
                    logger.info(format_progress(i+1, len(dataset), "Loading examples"))
            
            logger.info(f"SUCCESS Loaded {len(tasks)} task(s) from dataset")
        
        # If no dataset but user_goal provided, create a single-task list
        if not tasks and user_goal:
            tasks = [{"instruction": user_goal}]
            logger.info(f"Using user-provided goal: {user_goal[:60]}{'...' if len(user_goal) > 60 else ''}")
        elif not tasks:
            # Default fallback
            user_goal = "Process medical images in 3D Slicer with segmentation and analysis."
            tasks = [{"instruction": user_goal}]
            logger.info(f"Using default goal: {user_goal}")
        
        # Process each task
        logger.info(format_separator(" TASK PROCESSING "))
        all_results = []
        
        for task_idx, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{task_idx}")
            
            # Extract software name and task number from task_id
            # Formats: "3dslicer_endtoend_048", "orthanc_endtoend_040", etc.
            software_name = "3DSlicer"  # Default
            task_number = None
            
            if task_id:
                # Extract software name (everything before "_endtoend_")
                if "_endtoend_" in task_id:
                    software_part = task_id.split("_endtoend_")[0]
                    # Capitalize first letter: "3dslicer" -> "3DSlicer", "orthanc" -> "Orthanc"
                    if software_part:
                        software_name = software_part.capitalize()
                        # Special cases
                        if software_part.lower() == "3dslicer":
                            software_name = "3DSlicer"
                        elif software_part.lower() == "openhospital":
                            software_name = "OpenHospital"
                
                # Extract task number (2-3 digits at the end)
                # Try 3 digits first, then 2 digits
                match = re.search(r'(\d{2,3})$', task_id)
                if match:
                    task_number = match.group(1).zfill(3)  # Zero-pad to 3 digits
                else:
                    # Fallback: use task_idx + 1 with zero padding
                    task_number = f"{task_idx + 1:03d}"
            else:
                task_number = f"{task_idx + 1:03d}"
            
            logger.info(format_separator(f" {task_id} "))
            logger.info(format_task_header(task_idx + 1, len(tasks), task_id))
            
            # Extract instruction/user_goal from task
            current_user_goal = task.get("instruction", user_goal or "Complete the task.")
            logger.info(f"Goal: {current_user_goal[:100]}{'...' if len(current_user_goal) > 100 else ''}")
            
            # Process the task
            start_time = time.time()
            task_result = self.process_single_task(current_user_goal, task)
            execution_time = time.time() - start_time
            
            # Add metadata
            task_result['task_index'] = task_idx + 1
            task_result['task_instruction'] = current_user_goal
            task_result['execution_time'] = execution_time
            
            # Show completion summary
            if task_result.get("finished", False):
                logger.info(f"SUCCESS Task {task_idx + 1}/{len(tasks)} completed in {task_result.get('steps_taken', 0)} steps ({execution_time:.1f}s)")
            else:
                logger.warning(f"WARNING Task {task_idx + 1}/{len(tasks)} incomplete after {task_result.get('steps_taken', 0)} steps ({execution_time:.1f}s)")
            all_results.append(task_result)
        
            # Save individual task result immediately for verification
            try:
                # Use absolute path based on project root
                try:
                    import _setup_paths
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                except ImportError:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                
                task_results_dir = os.path.join(project_root, "OOD")
                os.makedirs(task_results_dir, exist_ok=True)
                
                # Create directory for images
                images_dir = os.path.join(task_results_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                # Create a copy of task_result without base64 image data for smaller files
                task_result_save = json.loads(json.dumps(task_result, default=str))  # Deep copy
                
                # Restructure task result to new format
                # New format: metadata with user_goal, then steps array
                # Get num_steps from task context
                total_steps = task.get("num_steps", 0) if task else 0
                new_task_result = {
                    "metadata": {
                        "user_goal": task_result_save.get("user_goal", ""),
                        "finished": task_result_save.get("finished", False),
                        "total_steps": total_steps
                    },
                    "steps": []
                }
                
                # Process steps_history: save images to disk and store paths, restructure to new format
                if 'steps_history' in task_result_save:
                    for step_idx, step_dict in enumerate(task_result_save['steps_history']):
                        # Extract step key (e.g., "Step 1") and step data
                        step_key = None
                        step_data = None
                        if isinstance(step_dict, dict):
                            step_keys = [k for k in step_dict.keys() if k.startswith("Step ")]
                            if step_keys:
                                step_key = step_keys[0]
                                step_data = step_dict[step_key]
                            else:
                                # Old format - convert to new format
                                step_num_val = step_dict.get('step_num', step_idx + 1)
                                step_key = f"Step {step_num_val}"
                                step_data = step_dict
                        
                        if step_data:
                            # Process image_info if present
                            if 'image_info' in step_data:
                                image_info = step_data['image_info']
                                if 'image_data_uri' in image_info:
                                    uri = image_info['image_data_uri']
                                    if uri and uri.startswith('data:image'):
                                        try:
                                            # Extract image format and base64 data
                                            header, base64_data = uri.split(',', 1)
                                            format_part = header.split(';')[0]
                                            
                                            if '/' in format_part:
                                                format_str = format_part.split('/')[-1].lower()
                                                if format_str == 'jpeg':
                                                    format_str = 'jpg'
                                            else:
                                                format_str = 'png'
                                            
                                            import base64
                                            image_bytes = base64.b64decode(base64_data)
                                            
                                            step_num = step_data.get('image_info', {}).get('step_num', step_idx + 1)
                                            # Format: {SoftwareName}_endtoend{task_number}_step{step_num}.{format}
                                            # e.g., "3DSlicer_endtoend048_step1.png" or "Orthanc_endtoend040_step1.png"
                                            image_filename = f"{software_name}_endtoend{task_number}_step{step_num}.{format_str}"
                                            image_path = os.path.join(images_dir, image_filename)
                                            
                                            with open(image_path, 'wb') as img_file:
                                                img_file.write(image_bytes)
                                            
                                            relative_image_path = os.path.join("OOD", "images", image_filename)
                                            image_info['image_path'] = relative_image_path
                                            image_info['image_path_absolute'] = image_path
                                            image_info['image_format'] = format_str
                                            image_info['image_data_uri'] = f"[Saved to disk - {relative_image_path}]"
                                            
                                        except Exception as img_error:
                                            logger.warning(f"Failed to save image for step {step_idx + 1}: {img_error}")
                                            image_info['image_data_uri'] = f"[Image save failed: {str(img_error)}]"
                            
                            # Add step to new structure with Step N key
                            if step_key:
                                new_task_result["steps"].append({step_key: step_data})
                            else:
                                # Fallback: create step key from step_data
                                step_num_val = step_data.get('image_info', {}).get('step_num', step_idx + 1)
                                new_task_result["steps"].append({f"Step {step_num_val}": step_data})
                
                # Format: {SoftwareName}_endtoend_{task_number}.json
                # e.g., "3DSlicer_endtoend_048.json" or "Orthanc_endtoend_040.json"
                task_result_file = os.path.join(task_results_dir, f"{software_name}_endtoend_{task_number}.json")
                with open(task_result_file, 'w') as f:
                    json.dump(new_task_result, f, indent=2, default=str)
                
                logger.info(f"SAVED Task {task_idx + 1} result to: {task_result_file}")
                logger.info(f"  - Steps history length: {len(task_result.get('steps_history', []))}")
                logger.info(f"  - Finished: {task_result.get('finished', False)}")
                logger.info(f"  - Images saved to: {images_dir}")
                
            except Exception as e:
                logger.error(f"ERROR Failed to save task result: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Save synthetic dataset incrementally and after all tasks complete
        logger.info(format_separator(" SYNTHETIC DATASET CREATION "))
        
        try:
            from data.dataset_creator import DatasetCreator
            dataset_creator = DatasetCreator()
            
            # Collect all trajectories with steps_history (convert to new format)
            all_trajectories = []
            for result in all_results:
                # Use steps array if available (new format), otherwise convert from steps_history
                steps_data = result.get('steps', [])
                if not steps_data and 'steps_history' in result:
                    # Convert from dict format to array format
                    for step_dict in result['steps_history']:
                        if isinstance(step_dict, dict):
                            step_keys = [k for k in step_dict.keys() if k.startswith("Step ")]
                            if step_keys:
                                steps_data.append(step_dict[step_keys[0]])
                            else:
                                steps_data.append(step_dict)
                
                if steps_data:
                    all_trajectories.append({
                        'task_id': result.get('task_instruction', 'unknown')[:50],
                        'user_goal': result.get('user_goal'),
                        'steps': steps_data,
                        'finished': result.get('finished', False),
                        'steps_taken': result.get('steps_taken', 0),
                        'ground_truth_available': result.get('ground_truth_available', False),
                        'used_ground_truth_count': result.get('used_ground_truth_count', 0)
                    })
            
            if all_trajectories:
                # Save final dataset with all trajectories
                output_path = dataset_creator.create_dataset_from_trajectories(all_trajectories)
                logger.info(f"SUCCESS Synthetic dataset saved to: {output_path}")
                logger.info(f"STATS Total trajectories: {len(all_trajectories)}, Total steps: {sum(len(t['steps']) for t in all_trajectories)}")
                
                # Save combined dataset in new format to synthetic_dataset folder
                synthetic_dataset_path = os.path.join(dataset_creator.output_dir, "synthetic_dataset_latest.json")
                combined_dataset = {
                    "metadata": {
                        "created_at": datetime.datetime.now().isoformat(),
                        "total_tasks": len(all_trajectories),
                        "total_steps": sum(len(t['steps']) for t in all_trajectories)
                    },
                    "tasks": []
                }
                
                # Convert each trajectory to new format
                for traj in all_trajectories:
                    # Get total_steps from the steps array length or from original data
                    total_steps = len(traj.get('steps', []))
                    task_data = {
                        "metadata": {
                            "user_goal": traj.get('user_goal', ''),
                            "task_id": traj.get('task_id', ''),
                            "finished": traj.get('finished', False),
                            "total_steps": total_steps
                        },
                        "steps": traj.get('steps', [])
                    }
                    combined_dataset["tasks"].append(task_data)
                
                with open(synthetic_dataset_path, 'w') as f:
                    json.dump(combined_dataset, f, indent=2, default=str)
                
                logger.info(f"SUCCESS Combined synthetic dataset saved to: {synthetic_dataset_path}")
            else:
                logger.warning("No trajectories with steps_history found - synthetic dataset not created")
                
        except Exception as e:
            logger.error(f"ERROR Failed to create synthetic dataset: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return all_results
    
    def process_single_task(
        self,
        user_goal: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single task through the controller loop.
        
        Args:
            user_goal: The instruction/goal for this task
            task_context: Optional additional context from the dataset
        
        Returns:
            Dictionary with task results and trajectory
        """
        # Extract ground truth trajectory data
        trajectory_data = {}
        ground_truth_available = False
        
        if task_context:
            logger.info(f"Task Context Keys: {list(task_context.keys())}")
            
            # Check for trajectory data in json_data
            if 'json_data' in task_context and task_context['json_data']:
                try:
                    if isinstance(task_context['json_data'], str):
                        json_data = json.loads(task_context['json_data'])
                    else:
                        json_data = task_context['json_data']
                        
                    logger.info("JSON Data Structure:")
                    if isinstance(json_data, dict):
                        logger.info(f"  - Keys: {list(json_data.keys())}")
                        
                        # Extract trajectory info if available
                        if 'trajectory' in json_data:
                            trajectory_list = json_data['trajectory']
                            logger.info(f"  - Trajectory steps: {len(trajectory_list)}")
                            logger.info(f"  - Action types: {set(step.get('action', 'UNKNOWN') for step in trajectory_list)}")
                            
                            # Build ground truth dictionary indexed by step number
                            # Focus only on action and target - no coordinate data
                            for idx, step_data in enumerate(trajectory_list):
                                step_number = idx + 1  # Steps are 1-indexed
                                trajectory_data[step_number] = {
                                    'action': step_data.get('action'),
                                    'target': step_data.get('target'),
                                    'original_step_data': step_data  # Keep full original data for reference
                                }
                            ground_truth_available = True
                            logger.info(f"SUCCESS Extracted ground truth for {len(trajectory_data)} steps")
                except Exception as e:
                    logger.warning(f"Error parsing json_data: {e}")
        
        # Get available tools
        available_tools = self.medical_tools.get_available_tools()
        logger.info(f"Available actions: {list(available_tools.keys())}")
        
        # Extract images from task context
        task_images = task_context.get('images', []) if task_context else []
        
        step_num = 1
        memory_manager = MemoryManager()
        short_term_memory, long_term_memory = memory_manager.get_memories()
        full_trajectory = ""
        retries = 0
        finished = False
        steps_history = []  # Track all steps for synthetic dataset creation
        used_ground_truth_count = 0  # Track how many times we used ground truth fallback
        
        # Determine estimated steps from task context (actual number of steps needed)
        # Use max_total_steps only as fallback if num_steps is not provided
        estimated_steps = task_context.get("num_steps", self.max_total_steps) if task_context else self.max_total_steps
        
        # Helper function to generate reason for step
        def generate_reason(step_num, grounding, short_term_memory, long_term_memory, predicted_action, user_goal, used_ground_truth=False):
            """Generate reason explaining why this action was predicted."""
            action_type = predicted_action.get("intent") or predicted_action.get("tool_call", "")
            target = predicted_action.get("target_label") or predicted_action.get("target", "")
            
            reason_parts = []
            
            if used_ground_truth:
                reason_parts.append(f"Model predictions failed after multiple attempts. Using ground truth action {action_type} to continue task progress.")
            elif action_type == "COMPLETE":
                reason_parts.append("All required operations completed successfully. Task objectives have been fully achieved and task is now complete.")
            elif step_num == 1:
                reason_parts.append(f"Starting the task. Need to {action_type.lower()} on {target} to initiate the workflow and begin processing the task objectives.")
            elif short_term_memory and short_term_memory.get("last_action") and short_term_memory.get("last_action") != "NONE":
                last_action = short_term_memory.get("last_action", "")
                reason_parts.append(f"Previous step completed ({last_action}). Now proceeding with {action_type.lower()} on {target} to continue the task workflow.")
            else:
                reason_parts.append(f"Following the task progression. Need to {action_type.lower()} on {target} based on previous steps and current requirements.")
            
            # Combine and trim to 30-40 words
            reason = " ".join(reason_parts)
            words = reason.split()
            if len(words) > 40:
                reason = " ".join(words[:40])
            elif len(words) < 30 and action_type != "COMPLETE":
                reason += f" This action is necessary to progress towards completing the task objectives."
                words = reason.split()
                if len(words) > 40:
                    reason = " ".join(words[:40])
            
            return reason
        
        # Helper function to format step data in new structure
        def format_step_data(step_num, grounding, short_term_memory, long_term_memory, predicted_action, 
                           ground_truth, critic_obj, verifier, decision_branch, final_action_correct,
                           image_info, used_ground_truth=False):
            """Format step data according to new structure."""
            
            # Format ground_truth with original_step_data
            formatted_ground_truth = None
            if ground_truth:
                original_step_data = ground_truth.get("original_step_data", {})
                if not original_step_data and isinstance(ground_truth, dict):
                    # Extract from trajectory data if available
                    if 'json_data' in task_context and task_context['json_data']:
                        try:
                            json_data = task_context['json_data']
                            if isinstance(json_data, dict) and 'trajectory' in json_data:
                                trajectory = json_data['trajectory']
                                if step_num <= len(trajectory):
                                    original_step_data = trajectory[step_num - 1]
                        except Exception:
                            pass
                
                formatted_ground_truth = {
                    "action": ground_truth.get("action", ""),
                    "target": ground_truth.get("target", ""),
                    "original_step_data": original_step_data if original_step_data else {
                        "step": step_num,
                        "action": ground_truth.get("action", ""),
                        "target": ground_truth.get("target", ""),
                        "screenshot": "",
                        "note": f"Step {step_num}",
                        "bbox": []
                    }
                }
            
            # Format predicted action
            formatted_predicted = {
                "action": predicted_action.get("intent") or predicted_action.get("tool_call", ""),
                "target": predicted_action.get("target_label") or predicted_action.get("target", ""),
                "coords": predicted_action.get("coords", [])
            }
            
            # Format critic_result
            tool_evaluation = critic_obj.get("tool_evaluation", {})
            formatted_critic_result = {
                "action_correct": critic_obj.get("action_correct", False),
                "why_if_wrong": critic_obj.get("why_if_wrong", ""),
                "hint_if_wrong": critic_obj.get("hint_if_wrong", ""),
                "tool_evaluation": {
                    "tools_used": tool_evaluation.get("tools_used", []),
                    "tool_success": tool_evaluation.get("tool_success", True),
                    "tool_lessons": tool_evaluation.get("tool_lessons", "")
                },
                "decision_branch": decision_branch,
                "final_action_correct": final_action_correct
            }
            
            # Get grounding from step_obj if available, otherwise create from available data
            formatted_grounding = grounding if grounding else {
                "current_screen_state": "",
                "key_ui_elements": [],
                "relevant_affordances": []
            }
            
            # Generate reason
            reason = generate_reason(
                step_num=step_num,
                grounding=formatted_grounding,
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
                predicted_action=predicted_action,
                user_goal=user_goal,
                used_ground_truth=used_ground_truth
            )
            
            # Format Short Term and Long Term (empty for step 1)
            formatted_short_term = {} if step_num == 1 else (short_term_memory if short_term_memory else {})
            
            # Format Long Term memory, excluding debugging/analysis fields
            if step_num == 1:
                formatted_long_term = {}
            elif long_term_memory:
                # Create a copy and remove fields that are not needed in the final dataset
                formatted_long_term = long_term_memory.copy()
                # Remove debugging/analysis fields
                formatted_long_term.pop("decision_history", None)
                formatted_long_term.pop("backtrack_points", None)
                formatted_long_term.pop("failure_types", None)
                formatted_long_term.pop("tool_effectiveness", None)
            else:
                formatted_long_term = {}
            
            # Format image_info
            formatted_image_info = {
                "step_num": step_num,
                "has_image": image_info.get("has_image", False),
                "image_data_uri": image_info.get("image_data_uri", ""),
                "image_path": image_info.get("image_path", ""),
                "image_path_absolute": image_info.get("image_path_absolute", ""),
                "image_format": image_info.get("image_format", "")
            }
            
            return {
                "Grounding": {
                    "ground_truth": formatted_ground_truth
                },
                "Short Term": formatted_short_term,
                "Long Term": formatted_long_term,
                "Reason": reason,
                "Predicted": {
                    "predicted_action": formatted_predicted
                },
                "image_info": formatted_image_info
            }
        
        # Initialize progress bar with actual estimated steps
        pbar = tqdm(
            total=estimated_steps,
            desc="Processing task",
            unit="step",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Track task start time
        task_start_time = time.time()
        step_times = []  # Track time for each completed step
        
        def update_progress_bar():
            """Update progress bar after step completion"""
            step_elapsed = time.time() - step_start_time
            step_times.append(step_elapsed)
            
            # Calculate average time per step
            avg_step_time = sum(step_times) / len(step_times) if step_times else 0
            
            # Calculate remaining steps
            remaining_steps = max(0, estimated_steps - (step_num - 1))
            
            # Estimate remaining time
            estimated_remaining = avg_step_time * remaining_steps if avg_step_time > 0 else 0
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'step': f'{step_num - 1}/{estimated_steps}',
                'elapsed': f'{time.time() - task_start_time:.1f}s',
                'remaining': f'{estimated_remaining:.1f}s' if estimated_remaining > 0 else '?'
            })
        
        # Main control loop - continue until task complete or actual num_steps reached
        # Use estimated_steps (from task num_steps) not max_total_steps (safety limit only)
        while not finished and step_num <= estimated_steps:
            # Track step start time
            step_start_time = time.time()
            
            # Determine if this is the last step based on actual estimated_steps
            is_last_step = (step_num >= estimated_steps)
            
            # (A) Get current screenshot from task images or use placeholder
            if task_images and len(task_images) > 0:
                # Use the image corresponding to current step (or first image if step beyond available images)
                img_index = min(step_num - 1, len(task_images) - 1)
                current_image = task_images[img_index]
                screen_image_url = encode_image_to_data_uri(current_image)
            else:
                # Fallback: If no images available, raise an error since the API requires valid images
                raise ValueError(
                    f"No images available for task. "
                    f"The dataset should contain 'images' field with PIL Image objects. "
                    f"Task context: {task_context.keys() if task_context else 'None'}"
                )
            
            # (B) Optional pre-grounding: Check memory for tool hints
            pre_grounding_results = None
            if isinstance(long_term_memory, dict) and "tool_effectiveness" in long_term_memory:
                tool_eff = long_term_memory.get('tool_effectiveness', {})
                
                # If we have low success rate or lessons suggest using tools, do pre-grounding
                should_pre_ground = False
                if short_term_memory.get("last_lesson", "").lower() in ["need better coordinates", "wrong element", "not found"]:
                    should_pre_ground = True
                
                if should_pre_ground:
                    logger.info("Running pre-grounding tools based on memory hints...")
                    # This would call appropriate tools based on memory
                    # For now, we'll just run object detection as an example
                    pre_grounding_results = self.visual_tools.object_detection(
                        objects=["button", "menu", "icon"],
                        image=current_image
                    )
            
            # (C) Run Target agent to get action prediction
            logger.info(format_separator(f" STEP {step_num}/{estimated_steps} "))
            logger.info(format_step_header(step_num, estimated_steps))
            
            if is_last_step:
                logger.info("WARNING This is the FINAL STEP - agent must use COMPLETE action")
            
            target_answer_text = run_target_agent(
                step_num=step_num,
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
                screen_image_url=screen_image_url,
                available_tools=available_tools,
                user_goal=user_goal,
                pre_grounding_results=pre_grounding_results,
                api_client=self.api_client,
                max_steps=estimated_steps  # Pass max_steps so agent knows to use COMPLETE on last step
            )
            
            # (D) Parse Target agent's response (robust parsing from try.py)
            target_answer = None
            step_obj = None
            last_good_target_answer = None
            
            try:
                # Try to fix common JSON issues before parsing
                logger.info("Parsing TARGET agent JSON response...")
                fixed_text = target_answer_text
                
                # Replace placeholders like [x, y] with [0, 0]
                fixed_text = re.sub(r'\[\s*x\s*,\s*y\s*\](?:\s*//.*?$)?', '[0, 0]', fixed_text, flags=re.MULTILINE)
                
                # Remove comments
                fixed_text = re.sub(r'\s*//.*?$', '', fixed_text, flags=re.MULTILINE)
                
                # Handle more complex cases like "coords": [x, y]
                fixed_text = re.sub(r'"coords"\s*:\s*\[\s*[a-zA-Z]+\s*,\s*[a-zA-Z]+\s*\](?:\s*//.*?$)?', '"coords": [0, 0]', fixed_text, flags=re.MULTILINE)
                
                # Direct fix for the specific pattern: [ "Step 1": { ... which is invalid JSON
                if fixed_text.startswith("[") and re.search(r'\[\s*"?Step\s+\d+"?\s*:', fixed_text):
                    logger.info("Detected Step pattern in an array context - applying direct fix")
                    fixed_json = "{" + fixed_text[1:-1] + "}"
                    try:
                        target_answer = json.loads(fixed_json)
                        logger.info("Direct fix successful!")
                    except Exception as e:
                        logger.warning(f"Direct fix failed: {e}")
                        # Try one more approach - manually extract and rebuild the JSON
                        try:
                            logger.info("Trying manual extraction and rebuilding...")
                            # Extract key components
                            action_match = re.search(r'"predicted_next_action"\s*:\s*(\{[^\}]+\})', fixed_text, re.DOTALL)
                            
                            # Build a minimal valid JSON
                            minimal_json = {
                                f"Step {step_num}": {
                                    "predicted_next_action": {
                                        "tool_call": "CLICK",
                                        "target": "UI Element",
                                        "arguments": {
                                            "coords": [100, 100]
                                        }
                                    }
                                }
                            }
                            
                            # Try to parse action component
                            if action_match:
                                try:
                                    action_text = action_match.group(1)
                                    # Fix any remaining issues
                                    action_text = re.sub(r'\[\s*[a-zA-Z]+\s*,\s*[a-zA-Z]+\s*\]', '[0, 0]', action_text)
                                    action_text = re.sub(r'//.*?$', '', action_text, flags=re.MULTILINE)
                                    # Add missing quotes around keys if needed
                                    action_text = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)(\s*:)', r'\1"\2"\3', action_text)
                                    minimal_json[f"Step {step_num}"]["predicted_next_action"] = json.loads(action_text)
                                except Exception:
                                    pass
                            
                            target_answer = minimal_json
                            logger.info("Manual extraction successful!")
                        except Exception as e2:
                            logger.warning(f"Manual extraction failed: {e2}")
                            # Fall back to our comprehensive parser
                            target_answer = safe_json_loads(target_answer_text)
                else:
                    # Try parsing the fixed text
                    try:
                        target_answer = json.loads(fixed_text)
                        logger.info("Fixed text parsing successful!")
                    except Exception:
                        # Fall back to our comprehensive parser
                        target_answer = safe_json_loads(target_answer_text)
                
                logger.info("Successfully parsed TARGET agent response")
                
                # Last good parse for fallbacks
                last_good_target_answer = target_answer
                
            except Exception as e:
                logger.error(f"ERROR JSON parsing failed: {str(e)}")
                logger.debug(f"First 200 chars of response: {target_answer_text[:200]}")
                
                # Special handling for the specific case we're seeing
                if target_answer_text.startswith("[") and '"Step' in target_answer_text[:30]:
                    logger.info("Attempting emergency JSON structure extraction...")
                    try:
                        # Create a simpler representation by extracting just what we need
                        match = re.search(r'"Step\s+\d+"\s*:\s*(\{.*\})', target_answer_text, re.DOTALL)
                        if match:
                            step_content = match.group(1)
                            
                            # Handle nested JSON carefully
                            try:
                                content_obj = json.loads(step_content)
                                target_answer = {f"Step {step_num}": content_obj}
                                logger.info("Emergency extraction successful!")
                            except json.JSONDecodeError:
                                # Try one more level of bracket balancing
                                open_braces = step_content.count('{')
                                close_braces = step_content.count('}')
                                if open_braces > close_braces:
                                    # Add missing closing braces
                                    step_content += "}" * (open_braces - close_braces)
                                elif close_braces > open_braces:
                                    # Remove extra closing braces
                                    step_content = step_content[:step_content.rfind('}')] + '}'
                                
                                content_obj = json.loads(step_content)
                                target_answer = {f"Step {step_num}": content_obj}
                                logger.info("Emergency extraction with brace fixing successful!")
                        else:
                            # Try a more aggressive pattern
                            raw_match = re.search(r'Step\s+\d+.*?grounding.*?predicted_next_action', 
                                                target_answer_text, re.DOTALL)
                            if raw_match:
                                # Create a minimal valid object
                                target_answer = {
                                    f"Step {step_num}": {
                                        "grounding": {"current_screen_state": "Extracted partial content"},
                                        "predicted_next_action": {
                                            "tool_call": "CLICK",
                                            "target": "Extracted from raw text",
                                            "arguments": {"coords": [0, 0]}
                                        }
                                    }
                                }
                                logger.info("Created minimal valid object from raw text")
                            else:
                                raise ValueError("Could not find Step content")
                    except Exception as ex:
                        logger.warning(f"Emergency extraction failed: {ex}")
                        # For debugging, save the problematic JSON to a file
                        debug_file = f"debug_json_step_{step_num}.txt"
                        try:
                            with open(debug_file, "w") as f:
                                f.write(target_answer_text)
                            logger.info(f"Saved problematic JSON to {debug_file}")
                        except Exception:
                            pass
                        
                        # ULTIMATE FALLBACK: Create a valid object no matter what
                        logger.warning("ALL PARSING FAILED - Creating minimal valid object as ultimate fallback")
                        
                        # Extract as much information as possible from the raw text
                        tool_call = "CLICK"  # Default
                        target = "Main interface element"  # Default
                        coords = [100, 100]  # Default
                        
                        # Try to extract the tool_call
                        tool_match = re.search(r'"tool_call"\s*:\s*"([^"]+)"', target_answer_text)
                        if tool_match:
                            tool_call = tool_match.group(1)
                        
                        # Try to extract the target
                        target_match = re.search(r'"target"\s*:\s*"([^"]+)"', target_answer_text)
                        if target_match:
                            target = target_match.group(1)
                        
                        # Try to extract bbox from the task context (use trajectory data if available)
                        if task_context and 'json_data' in task_context and task_context['json_data']:
                            try:
                                json_data = task_context['json_data']
                                if isinstance(json_data, dict) and 'trajectory' in json_data:
                                    trajectory = json_data['trajectory']
                                    current_step_data = trajectory[min(step_num-1, len(trajectory)-1)] if trajectory else None
                                    if current_step_data and 'bbox' in current_step_data:
                                        bbox = current_step_data['bbox']
                                        coords = list(convert_bbox_to_center_point(bbox))
                            except Exception:
                                pass
                        
                        target_answer = {
                            f"Step {step_num}": {
                                "grounding": {
                                    "current_screen_state": "Parsing failed - using fallback",
                                },
                                "short_term_memory": {
                                    "last_action": "NONE",
                                    "last_observation": "NONE", 
                                    "last_lesson": "NONE"
                                },
                                "long_term_memory": {
                                    "overall_progress": "0%",
                                    "completed_subtasks": [],
                                    "remaining_subtasks": ["Complete task"],
                                    "known_pitfalls": []
                                },
                                "reasoning": {
                                    "why_next_action_is_correct_and_safe": "JSON parsing failed, using fallback",
                                    "why_it_aligns_with_user_goal": "Emergency fallback to keep process running",
                                    "why_alternatives_are_wrong_or_risky": "N/A"
                                },
                                "image_info": {
                                    "step_num": step_num,
                                    "has_image": bool(task_images and len(task_images) > 0),
                                    "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                                },
                                "predicted_next_action": {
                                    "tool_call": tool_call,
                                    "target": target, 
                                    "arguments": {
                                        "coords": coords if isinstance(coords, list) else [100, 100]
                                    }
                                }
                            }
                        }
                        
                        logger.info(f"Created ultimate fallback object with tool_call: {tool_call}")
                else:
                    # For debugging, save the problematic JSON to a file
                    debug_file = f"debug_json_step_{step_num}.txt"
                    try:
                        with open(debug_file, "w") as f:
                            f.write(target_answer_text)
                        logger.info(f"Saved problematic JSON to {debug_file}")
                    except Exception:
                        pass
                    
                    # Fallback: create minimal valid object
                    target_answer = {
                        f"Step {step_num}": {
                            "image_info": {
                                "step_num": step_num,
                                "has_image": bool(task_images and len(task_images) > 0),
                                "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                            },
                            "predicted_next_action": {
                                "tool_call": "CLICK",
                                "target": "Unknown",
                                "arguments": {"coords": [100, 100]}
                            }
                        }
                    }
            
            # Extract step object from target_answer
            if target_answer is None:
                # Ultimate fallback
                step_obj = {
                    "image_info": {
                        "step_num": step_num,
                        "has_image": bool(task_images and len(task_images) > 0),
                        "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                    },
                    "predicted_next_action": {"tool_call": "CLICK", "target": "Unknown", "arguments": {"coords": [100, 100]}}
                }
            else:
                # Extract step object
                if isinstance(target_answer, list) and len(target_answer) > 0:
                    # List format
                    first_item = target_answer[0]
                    if isinstance(first_item, dict):
                        step_key = list(first_item.keys())[0]
                        step_obj = first_item[step_key]
                    else:
                        step_obj = first_item
                elif isinstance(target_answer, dict):
                    # Dict format with step key
                    step_keys = [k for k in target_answer.keys() if k.startswith("Step ")]
                    if step_keys:
                        step_obj = target_answer[step_keys[0]]
                    else:
                        # Dict format without step key
                        step_obj = target_answer
                else:
                    step_obj = {
                        "image_info": {
                            "step_num": step_num,
                            "has_image": bool(task_images and len(task_images) > 0),
                            "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                        },
                        "predicted_next_action": {"tool_call": "CLICK", "target": "Unknown", "arguments": {"coords": [100, 100]}}
                    }
            
            # Add image_info to step_obj if not already present
            if "image_info" not in step_obj:
                step_obj["image_info"] = {
                    "step_num": step_num,
                    "has_image": bool(task_images and len(task_images) > 0),
                    "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                }
            
            # (E) Check for and execute tool calls from reasoning
            tool_calls = []
            tool_results_dict = {}
            tools_used = []
            
            # Only extract and execute tools if enabled
            if self.enable_tools:
                # Extract tool calls from reasoning
                if "reasoning" in step_obj and "tool_calls" in step_obj["reasoning"]:
                    tool_calls_data = step_obj["reasoning"]["tool_calls"]
                    
                    # Convert to standard format if needed
                    if isinstance(tool_calls_data, list):
                        tool_calls = tool_calls_data
                    elif isinstance(tool_calls_data, dict):
                        # Convert dict to list of tool calls
                        tool_calls = [{"tool": k, "args": v} for k, v in tool_calls_data.items()]
                    elif isinstance(tool_calls_data, str):
                        # Try to parse string as JSON
                        try:
                            parsed = json.loads(tool_calls_data)
                            if isinstance(parsed, list):
                                tool_calls = parsed
                            elif isinstance(parsed, dict):
                                tool_calls = [{"tool": k, "args": v} for k, v in parsed.items()]
                        except Exception:
                            tool_calls = []
                
                # Execute any tool calls
                if tool_calls:
                    logger.info(f"TOOL Executing {len(tool_calls)} tool call(s)...")
                    
                    # Use enhanced tool selection if available
                    if self.use_enhanced and self.tool_selector and self.tool_processor:
                        # Enhanced tool selection based on reinforcement learning
                        logger.info("Using enhanced tool selection...")
                        
                        # This would select the most effective tools based on learning
                        # For now, we just execute the requested tools
                        executed_results = self.tool_executor.execute_multiple_tools(
                            tool_calls, 
                            image=current_image
                        )
                        
                        # Process results with confidence weighting
                        executed_results = self.tool_processor.process_results(executed_results, user_goal)
                    else:
                        # Use standard tool execution
                        executed_results = self.tool_executor.execute_multiple_tools(
                            tool_calls, 
                            image=current_image
                        )
                    
                    tools_used = [tc.get("tool") for tc in tool_calls if tc.get("tool")]
                    tool_results_dict.update(executed_results)
                    
                    # Re-run Target agent with tool results
                    logger.info("Re-running TARGET agent with tool results...")
                    target_answer_text = run_target_agent(
                        step_num=step_num,
                        short_term_memory=short_term_memory,
                        long_term_memory=long_term_memory,
                        screen_image_url=screen_image_url,
                        available_tools=available_tools,
                        user_goal=user_goal,
                        pre_grounding_results=pre_grounding_results,
                        tool_results=tool_results_dict,
                        api_client=self.api_client,
                        max_steps=estimated_steps  # Pass max_steps so agent knows to use COMPLETE on last step
                    )
                    
                    # Re-parse after tool execution (use same robust parsing)
                    try:
                        # Use safe_json_loads which handles all the preprocessing
                        logger.info("Re-parsing target agent response after tool execution...")
                        target_answer = safe_json_loads(target_answer_text)
                        last_good_target_answer = target_answer
                        
                        # Re-extract step_obj
                        if isinstance(target_answer, list) and len(target_answer) > 0:
                            first_item = target_answer[0]
                            if isinstance(first_item, dict):
                                step_key = list(first_item.keys())[0]
                                step_obj = first_item[step_key]
                        elif isinstance(target_answer, dict):
                            step_keys = [k for k in target_answer.keys() if k.startswith("Step ")]
                            if step_keys:
                                step_obj = target_answer[step_keys[0]]
                            else:
                                step_obj = target_answer
                        
                        # Update tool_results_dict from new response
                        if isinstance(step_obj, dict):
                            new_tool_results = step_obj.get("tool_results", {})
                            tool_results_dict.update(new_tool_results)
                        
                        # Ensure image_info is present after re-parsing
                        if isinstance(step_obj, dict) and "image_info" not in step_obj:
                            step_obj["image_info"] = {
                                "step_num": step_num,
                                "has_image": bool(task_images and len(task_images) > 0),
                                "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                            }
                            
                        logger.info("Successfully re-parsed target agent response")
                        
                    except Exception as e:
                        logger.warning(f"Failed to re-parse target output after tool execution: {e}")
                        logger.info("Using previous successful parse - continuing with original step_obj")
                        # Keep using the original step_obj that was successfully parsed before tool execution
                        # This ensures we don't lose the main action prediction due to tool parsing issues
                else:
                    logger.info("No tool calls requested by agent for this step")
            else:
                logger.info("TOOLS DISABLED - Skipping tool execution (use --enable-tools or remove --no-tools to enable)")
                # Continue without tool results - agent will work based on image only
            
            # (F) Extract predicted action
            predicted_action = step_obj.get("predicted_next_action", {}) if step_obj else {}
            
            # Ensure predicted_action is a dict, not None
            if predicted_action is None:
                logger.warning("predicted_action is None, using default CLICK action")
                predicted_action = {
                    "tool_call": "CLICK",
                    "target": "Unknown",
                    "arguments": {"coords": [0, 0]}
                }
            
            # Enforce COMPLETE action on the last step
            if is_last_step:
                predicted_tool_call = predicted_action.get("tool_call", "")
                if predicted_tool_call != "COMPLETE":
                    logger.warning(f"WARNING Last step did not use COMPLETE action (got '{predicted_tool_call}') - enforcing COMPLETE")
                    predicted_action = {
                        "tool_call": "COMPLETE",
                        "target": predicted_action.get("target", "Task completion"),
                        "arguments": {}
                    }
                    # Update step_obj to reflect the enforced COMPLETE action
                    if step_obj:
                        step_obj["predicted_next_action"] = predicted_action
            
            # Canonicalize the prediction for the verifier
            # Ensure all values are strings/lists, not None
            predicted_canonical = {
                "intent": predicted_action.get("tool_call", "") if predicted_action.get("tool_call") else "",
                "target_label": predicted_action.get("target", "") if predicted_action.get("target") else "",
                "coords": predicted_action.get("arguments", {}).get("coords", [0, 0]) if predicted_action.get("arguments", {}).get("coords") else [0, 0]
            }
            
            # (G) Execute action in environment (or simulate)
            # In a real implementation, this would interact with a real environment
            # For this refactored example, we'll simulate the environment
            
            # Mock environment result
            env_canonical = {
                "changed_regions": [
                    {
                        "x1": predicted_canonical["coords"][0] - 20,
                        "y1": predicted_canonical["coords"][1] - 20,
                        "x2": predicted_canonical["coords"][0] + 20,
                        "y2": predicted_canonical["coords"][1] + 20
                    }
                ],
                "confirm_text": f"Executed {predicted_canonical['intent']} on {predicted_canonical['target_label']}"
            }
            
            # Record the full target output for trajectory
            if full_trajectory:
                full_trajectory += "\n\n"
            full_trajectory += f"Step {step_num}:\n{target_answer_text}\n\nEnvironment Result:\n{json.dumps(env_canonical, indent=2)}"
            
            # (H) Run Critic to assess the action
            logger.info("PROGRESS Running CRITIC agent to assess the action...")
            critic_text = run_critic_agent(
                step_num=step_num,
                target_output_json=json.dumps(step_obj, indent=2),
                ground_truth_after_action=json.dumps(env_canonical, indent=2),
                full_trajectory_so_far=full_trajectory,
                user_goal=user_goal,
                tool_results=tool_results_dict,
                tools_used=tools_used,
                short_term_memory=short_term_memory,
                api_client=self.api_client
            )
            
            # (I) Parse Critic's feedback
            # Check if critic_text is empty or None
            if not critic_text or not critic_text.strip():
                logger.error(f"Empty critic response received - using fallback structure")
                critic_obj = {
                    "action_correct": False,
                    "reflection.action": {
                        "last_action": predicted_canonical.get("intent") or predicted_canonical.get("tool_call") or "Action executed" if predicted_canonical else "Action executed",
                        "last_observation": "Empty response from critic model",
                        "last_lesson": "Critic model returned empty response"
                    },
                    "reflection.trajectory": {
                        "overall_progress": "",
                        "completed_subtasks": [],
                        "remaining_subtasks": [],
                        "known_pitfalls": []
                    },
                    "reflection.global": {
                        "status": "incomplete",
                        "missing_steps": [],
                        "next_instruction": "CONTINUE"
                    }
                }
            else:
                try:
                    critic_obj = safe_json_loads(critic_text)
                    # Log critic response structure for debugging
                    logger.debug(f"Critic response keys: {list(critic_obj.keys()) if isinstance(critic_obj, dict) else 'Not a dict'}")
                    if isinstance(critic_obj, dict) and "reflection.action" in critic_obj:
                        logger.debug(f"reflection.action type: {type(critic_obj.get('reflection.action'))}")
                    
                    # Validate that critic_obj has required structure
                    if not isinstance(critic_obj, dict):
                        logger.error(f"Critic response is not a dict: {type(critic_obj)}")
                        raise ValueError("Critic response is not a dictionary")
                        
                except Exception as e:
                    logger.error(f"Error parsing critic output: {e}")
                    logger.error(f"Critic response (first 500 chars): {critic_text[:500] if critic_text else 'EMPTY'}")
                critic_obj = {
                    "action_correct": False,
                    "reflection.action": {
                            "last_action": predicted_canonical.get("intent") or predicted_canonical.get("tool_call") or "Action executed" if predicted_canonical else "Action executed",
                            "last_observation": f"Error parsing critic output: {str(e)}",
                        "last_lesson": "Error in reflection system"
                    },
                    "reflection.trajectory": {
                        "overall_progress": "",
                        "completed_subtasks": [],
                        "remaining_subtasks": [],
                        "known_pitfalls": []
                    },
                    "reflection.global": {
                        "status": "incomplete",
                        "missing_steps": [],
                        "next_instruction": "CONTINUE"
                    }
                }
            
            # Extract the critic's judgment
            critic_action_ok = critic_obj.get("action_correct", False)
            
            # Extract reflection components - handle both flat and nested formats
            # Format 1: "reflection.action", "reflection.trajectory", "reflection.global" (flat keys)
            # Format 2: "reflection": { "action": {...}, "trajectory": {...}, "global": {...} } (nested)
            action_refl = {}
            traj_refl = {}
            glob_refl = {}
            
            # Helper function to safely extract and validate dict
            def safe_extract_dict(obj, key, default=None):
                """Extract dict from object, handling string conversions.
                Returns tuple: (dict_value, was_string) where was_string indicates if original was string.
                """
                value = obj.get(key, default)
                if value is None:
                    return (default or {}, False)
                if isinstance(value, dict):
                    return (value, False)
                if isinstance(value, str):
                    # Try to parse as JSON first
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return (parsed, False)  # Successfully parsed as dict
                        else:
                            logger.warning(f"{key} parsed to non-dict: {type(parsed)}, will use string conversion")
                            return ({}, True)  # Return empty dict but flag as string for later conversion
                    except:
                        # If not JSON, return empty dict but flag as string
                        logger.warning(f"{key} is a string (not JSON), will convert. Value: {value[:100]}")
                        return ({}, True)  # Return empty dict but flag as string for later conversion
                logger.warning(f"{key} is not a dict or string (got {type(value)}), using default")
                return (default or {}, False)
            
            # Track original string values for proper conversion
            action_refl_str = None
            traj_refl_str = None
            glob_refl_str = None
            
            # Try flat format first
            if "reflection.action" in critic_obj:
                action_refl, was_string = safe_extract_dict(critic_obj, "reflection.action", {})
                if was_string:
                    action_refl_str = critic_obj.get("reflection.action")
            elif "reflection" in critic_obj and isinstance(critic_obj["reflection"], dict):
                # Try nested format
                reflection = critic_obj.get("reflection", {})
                if "action" in reflection:
                    action_refl, was_string = safe_extract_dict(reflection, "action", {})
                    if was_string:
                        action_refl_str = reflection.get("action")
            
            if "reflection.trajectory" in critic_obj:
                traj_refl, was_string = safe_extract_dict(critic_obj, "reflection.trajectory", {})
                if was_string:
                    traj_refl_str = critic_obj.get("reflection.trajectory")
            elif "reflection" in critic_obj and isinstance(critic_obj["reflection"], dict):
                reflection = critic_obj.get("reflection", {})
                if "trajectory" in reflection:
                    traj_refl, was_string = safe_extract_dict(reflection, "trajectory", {})
                    if was_string:
                        traj_refl_str = reflection.get("trajectory")
            
            if "reflection.global" in critic_obj:
                glob_refl, was_string = safe_extract_dict(critic_obj, "reflection.global", {})
                if was_string:
                    glob_refl_str = critic_obj.get("reflection.global")
            elif "reflection" in critic_obj and isinstance(critic_obj["reflection"], dict):
                reflection = critic_obj.get("reflection", {})
                if "global" in reflection:
                    glob_refl, was_string = safe_extract_dict(reflection, "global", {})
                    if was_string:
                        glob_refl_str = reflection.get("global")
            
            # Handle string values - convert them to dicts with appropriate fields
            if action_refl_str and isinstance(action_refl_str, str):
                logger.warning(f"reflection.action is a string, converting to dict: {action_refl_str[:100]}...")
                # The string is likely the observation or lesson, use it as observation
                action_refl = {"last_action": "", "last_observation": action_refl_str, "last_lesson": ""}
            
            if traj_refl_str and isinstance(traj_refl_str, str):
                logger.warning(f"reflection.trajectory is a string, converting to dict: {traj_refl_str[:100]}...")
                # Use the string as overall_progress
                traj_refl = {"overall_progress": traj_refl_str, "completed_subtasks": [], "remaining_subtasks": [], "known_pitfalls": []}
            
            if glob_refl_str and isinstance(glob_refl_str, str):
                logger.warning(f"reflection.global is a string, converting to dict: {glob_refl_str[:100]}...")
                # Try to infer status from string content
                glob_str_lower = glob_refl_str.lower()
                if any(word in glob_str_lower for word in ["complete", "finished", "done"]):
                    glob_refl = {"status": "complete", "missing_steps": [], "next_instruction": "TERMINATE"}
                else:
                    glob_refl = {"status": "incomplete", "missing_steps": [glob_refl_str], "next_instruction": "CONTINUE"}
            
            # Final validation - ensure all are dicts
            if not isinstance(action_refl, dict):
                logger.warning(f"action_refl still not a dict (got {type(action_refl)}), setting to empty dict")
                action_refl = {}
            if not isinstance(traj_refl, dict):
                logger.warning(f"traj_refl still not a dict (got {type(traj_refl)}), setting to empty dict")
                traj_refl = {}
            if not isinstance(glob_refl, dict):
                logger.warning(f"glob_refl still not a dict (got {type(glob_refl)}), setting to empty dict")
                glob_refl = {}
            
            # Fallback: Populate last_action from predicted_action if critic didn't provide it
            if not action_refl.get("last_action") or action_refl.get("last_action") == "":
                if predicted_canonical:
                    action_type = predicted_canonical.get("intent") or predicted_canonical.get("tool_call") or ""
                    target = predicted_canonical.get("target_label") or predicted_canonical.get("target") or ""
                    if action_type:
                        action_refl["last_action"] = f"{action_type} {target}".strip() if target else action_type
                    elif target:
                        action_refl["last_action"] = f"CLICK {target}"
                    else:
                        action_refl["last_action"] = "Action executed"
                else:
                    action_refl["last_action"] = "Action executed"
                    
            if not isinstance(traj_refl, dict):
                logger.warning(f"traj_refl is not a dict (got {type(traj_refl)}), converting: {traj_refl}")
                if isinstance(traj_refl, str):
                    try:
                        traj_refl = json.loads(traj_refl)
                    except:
                        traj_refl = {"overall_progress": traj_refl, "completed_subtasks": [], "remaining_subtasks": [], "known_pitfalls": []}
                else:
                    traj_refl = {}
            
            # Fallback: Generate overall_progress if critic didn't provide it
            if not traj_refl.get("overall_progress") or traj_refl.get("overall_progress") == "":
                # Calculate progress percentage based on step number
                if estimated_steps > 0:
                    progress_pct = int((step_num / estimated_steps) * 100)
                    progress_pct = min(progress_pct, 100)  # Cap at 100%
                    
                    # Generate descriptive progress message
                    if progress_pct == 0:
                        traj_refl["overall_progress"] = "Just starting the task"
                    elif progress_pct < 25:
                        traj_refl["overall_progress"] = f"Early stage ({progress_pct}% complete). Working on initial setup and data loading."
                    elif progress_pct < 50:
                        traj_refl["overall_progress"] = f"Mid-stage ({progress_pct}% complete). Processing data and applying transformations."
                    elif progress_pct < 75:
                        traj_refl["overall_progress"] = f"Advanced stage ({progress_pct}% complete). Creating segmentations and analysis."
                    elif progress_pct < 100:
                        traj_refl["overall_progress"] = f"Near completion ({progress_pct}% complete). Finalizing results and exports."
                    else:
                        traj_refl["overall_progress"] = "Task completed"
                else:
                    traj_refl["overall_progress"] = f"Step {step_num} of task in progress"
                    
            if not isinstance(glob_refl, dict):
                logger.warning(f"glob_refl is not a dict (got {type(glob_refl)}), converting: {glob_refl}")
                if isinstance(glob_refl, str):
                    # Try to parse as JSON first
                    try:
                        glob_refl = json.loads(glob_refl)
                    except:
                        # If string, create a dict with status - be VERY conservative about "complete"
                        # Only mark as complete if explicitly stated, not just because "complete" appears in text
                        # Many strings like "completed steps" or "completing tasks" should NOT trigger completion
                        glob_str_lower = glob_refl.lower()
                        
                        # Check for explicit completion signals - be very strict
                        # Must explicitly say task is complete, not just mention the word "complete"
                        is_complete = False
                        
                        # Explicit completion patterns:
                        if any([
                            glob_str_lower.startswith("task is complete"),
                            glob_str_lower.startswith("status: complete"),
                            glob_str_lower.startswith("status is complete"),
                            '"status": "complete"' in glob_str_lower,
                            'status": "complete' in glob_str_lower,
                            ("all required steps" in glob_str_lower and "complete" in glob_str_lower and "missing" not in glob_str_lower and "incomplete" not in glob_str_lower),
                            ("all tasks" in glob_str_lower and "complete" in glob_str_lower and "missing" not in glob_str_lower and "remaining" not in glob_str_lower)
                        ]):
                            is_complete = True
                        
                        # If string mentions "missing", "remaining", "incomplete", or "but" - definitely not complete
                        if any(word in glob_str_lower for word in ["missing", "remaining", "incomplete", "but missed", "still need", "need to", "must do"]):
                            is_complete = False
                        
                        if is_complete:
                            glob_refl = {"status": "complete", "missing_steps": [], "next_instruction": "TERMINATE"}
                        else:
                            # Default to incomplete - even if string mentions "complete", it's likely partial progress
                            glob_refl = {"status": "incomplete", "missing_steps": [glob_refl] if glob_refl else [], "next_instruction": "CONTINUE"}
                else:
                    glob_refl = {}
            
            # Tool evaluation for learning
            tool_evaluation = critic_obj.get("tool_evaluation", {})
            tool_success = tool_evaluation.get("tool_success", True)
            tool_lesson = tool_evaluation.get("tool_lessons", "")
            
            # (J) Run verifier for objective checks
            logger.info("PROGRESS Running verifier for objective checks...")
            
            # Get ground truth for current step if available
            current_ground_truth = trajectory_data.get(step_num) if ground_truth_available else None
            if current_ground_truth:
                logger.info(f"Using ground truth for step {step_num}: action={current_ground_truth.get('action')}, target={current_ground_truth.get('target')}")
            
            verifier = self.action_verifier.verify_action(
                predicted_canonical=predicted_canonical,
                env_canonical=env_canonical,
                ui_tree=None,  # No UI tree in this example
                ground_truth=current_ground_truth,  # Pass actual ground truth from trajectory
                is_first_step=(step_num == 1)
            )
            
            # Extract verifier results
            match_score = verifier["match_score"]
            ui_changed = verifier["ui_changed"]
            semantic_match = verifier["semantic_match"]
            semantic_pass = verifier["semantic_pass"]
            failure_type = verifier["failure_type"]
            
            # (K) Combine critic verdict + verifier for decision
            checks = verifier.get('checks', {})
            match_score = verifier.get('match_score', 0)
            
            logger.info(f"STATS Verification: Match={match_score:.2f}, Critic={critic_action_ok}")
            
            final_action_correct, decision_branch = self.action_verifier.analyze_decision(
                verifier_result=verifier,
                critic_action_ok=critic_action_ok
            )
            
            # Format the decision summary for better readability
            logger.info(format_decision(decision_branch, match_score, checks))
            
            # Add more context if action is wrong
            if not final_action_correct:
                why_wrong = verifier.get('why_if_wrong', '')
                if why_wrong:
                    logger.warning(f"Reason wrong: {why_wrong}")
            logger.info(f"Hint if wrong: {verifier.get('hint_if_wrong', '')}")
            
            # Update rewards for tool selection if enhanced components are used
            if self.use_enhanced and self.tool_selector and tools_used:
                if final_action_correct:
                    # Positive reward for successful action
                    reward = 0.5 + 0.5 * verifier['match_score']  # 0.5 to 1.0 for successful actions
                else:
                    # Negative reward for failed action
                    reward = -0.1 - 0.2 * (1.0 - verifier['match_score'])  # -0.3 to -0.1 for failed actions
                
                # Update rewards for each tool used
                for tool in tools_used:
                    self.tool_selector.update_reward(tool, reward)
            
            # Determine tool lesson from evaluation
            tool_lesson_from_eval = ""
            if tools_used:
                if tool_lesson:
                    # Use critic's tool lesson
                    tool_lesson_from_eval = tool_lesson
                elif not tool_success:
                    # Generic lesson for unsuccessful tool
                    tool_names = ", ".join(tools_used)
                    tool_lesson_from_eval = f"{tool_names} did not provide useful results. Try different tools."
            
            # Determine wrong hint - prioritize ground truth-based hints, then critic, then verifier
            wrong_hint = None
            
            # Generate ground truth-based hint if available and action is wrong
            # ONLY focus on action type, ignore target descriptions
            if not final_action_correct and current_ground_truth:
                gt_hints = []
                
                # Check ONLY action type mismatch
                gt_action = current_ground_truth.get('action')
                pred_action = predicted_canonical.get('tool_call')
                if gt_action and pred_action and gt_action != pred_action:
                    gt_hints.append(f"Action mismatch: predicted '{pred_action}', should use '{gt_action}'")
                elif gt_action:
                    # Action matches or no prediction - just show expected action
                    gt_hints.append(f"Expected action: {gt_action}")
                
                if gt_hints:
                    wrong_hint = " | ".join(gt_hints)
                    logger.info(f"Ground truth hint: {wrong_hint}")
            
            # Fall back to critic hint if no ground truth hint
            if not wrong_hint and critic_obj.get("hint_if_wrong") and critic_obj.get("hint_if_wrong").strip():
                wrong_hint = critic_obj.get("hint_if_wrong")
            # Fall back to verifier hint if no critic hint
            elif not wrong_hint and verifier.get("hint_if_wrong") and verifier.get("hint_if_wrong").strip():
                wrong_hint = verifier.get("hint_if_wrong")
            
            # (L) Update memory based on action correctness
            if final_action_correct:
                # Action succeeded, advance to next step
                
                # Branch A: Both agree it's correct
                if decision_branch == "A_both_agree_correct":
                    logger.info("Branch A: Both agree correct - advancing step")
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                    
                    # Add additional insights from critic if available
                    if critic_obj.get("why_if_wrong") and not short_term_memory.get("last_lesson"):
                        short_term_memory["last_lesson"] = critic_obj.get("why_if_wrong")
                    
                    # Get grounding from step_obj
                    grounding = step_obj.get("grounding", {})
                    
                    # Track step history for synthetic dataset with new format
                    step_data = format_step_data(
                        step_num=step_num,
                        grounding=grounding,
                        short_term_memory=short_term_memory,
                        long_term_memory=long_term_memory,
                        predicted_action=predicted_canonical,
                        ground_truth=current_ground_truth,
                        critic_obj=critic_obj,
                        verifier=verifier,
                        decision_branch=decision_branch,
                        final_action_correct=final_action_correct,
                        image_info=step_obj.get("image_info", {
                            "step_num": step_num,
                            "has_image": bool(task_images and len(task_images) > 0),
                            "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                        }),
                        used_ground_truth=False
                    )
                    steps_history.append({
                        f"Step {step_num}": step_data
                    })
                    
                    retries = 0
                    update_progress_bar()
                    step_num += 1
                    
                # Branch E: Ambiguous verifier but accepting
                elif decision_branch == "E_ambiguous_accept":
                    logger.info("Branch E: Ambiguous but accepting - advancing with caution")
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint="Action accepted with caution - semantic intent matched but some uncertainty",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                    
                    retries = 0
                    update_progress_bar()
                    step_num += 1
                
                # Default fallback accept
                else:
                    logger.info("Default accept - advancing step")
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                    
                    retries = 0
                    update_progress_bar()
                    step_num += 1
                
            else:  # Action failed
                retries += 1  # Increment retry counter
                
                # Check if we should use ground truth fallback after 2 failures
                if retries >= 2 and ground_truth_available and current_ground_truth:
                    logger.warning(f"WARNING Step {step_num} failed {retries} times - using ground truth fallback")
                    logger.info(f"Ground truth action: {current_ground_truth.get('action')}")
                    
                    # Use ground truth action directly - only action type matters
                    gt_action = current_ground_truth.get('action')
                    gt_target = current_ground_truth.get('target', 'ground_truth_target')
                    
                    # Override predicted action with ground truth (target/coords not important)
                    predicted_canonical = {
                        'tool_call': gt_action,
                        'target': gt_target,  # Use actual ground truth target
                        'arguments': {'coords': [100, 100]}  # Default coords - not important for dataset
                    }
                    
                    # Track that we used ground truth
                    used_ground_truth_count += 1
                    
                    # Get grounding from step_obj
                    grounding = step_obj.get("grounding", {})
                    
                    # Track step history for synthetic dataset with new format
                    step_data = format_step_data(
                        step_num=step_num,
                        grounding=grounding,
                        short_term_memory=short_term_memory,
                        long_term_memory=long_term_memory,
                        predicted_action=predicted_canonical,
                        ground_truth=current_ground_truth,
                        critic_obj=critic_obj,
                        verifier=verifier,
                        decision_branch='GROUND_TRUTH_FALLBACK',
                        final_action_correct=True,  # We accept ground truth as correct
                        image_info=step_obj.get("image_info", {
                            "step_num": step_num,
                            "has_image": bool(task_images and len(task_images) > 0),
                            "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else ""
                        }),
                        used_ground_truth=True
                    )
                    steps_history.append({
                        f"Step {step_num}": step_data
                    })
                    
                    # Update memory with ground truth usage
                    memory_manager.update_from_reflection(
                        action_refl={"last_action": f"Used ground truth: {gt_action}", "last_observation": "Ground truth fallback after multiple failures", "last_lesson": "Model predictions failed, used ground truth"},
                        traj_refl="Fallback to ground truth to continue task progress",
                        global_refl="Multiple prediction failures - need to improve action accuracy",
                        wrong_hint=f"Failed {retries} times, used ground truth: {gt_action}",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch='GROUND_TRUTH_FALLBACK'
                    )
                    
                    # Get updated memories and advance to next step
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                    retries = 0
                    update_progress_bar()
                    step_num += 1
                    continue  # Skip normal failure handling
                
                # Branch B: Partial success (UI changed but semantic weak)
                if decision_branch == "B_partial_success":
                    logger.info("Branch B: Partial success - UI changed but semantic weak")
                    
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint=wrong_hint or "UI changed but semantic intent was weak",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch,
                        failure_type=failure_type
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                
                # Branch C: Semantic failure but UI change
                elif decision_branch == "C_semantic_failure_with_ui_change":
                    logger.info("Branch C: Semantic failure with UI change - wrong target")
                    
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint=wrong_hint or "Semantic mismatch - wrong target selected",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch,
                        failure_type=failure_type
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                
                # Branch D: Clear failure (low match score, critic disagrees)
                elif decision_branch == "D_clear_failure":
                    logger.info("Branch D: Clear failure - restarting step")
                    
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint=wrong_hint or "Action failed, try a different approach",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch,
                        failure_type=failure_type
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                
                # Branch E: Ambiguous verifier (intermediate score, critic disagrees)
                elif decision_branch == "E_ambiguous_reject":
                    logger.info("Branch E: Ambiguous verifier - critic disagrees")
                    
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint=wrong_hint or "Action partially succeeded but did not meet criteria",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch,
                        failure_type=failure_type
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                
                # Default rejection
                else:
                    logger.info("Default reject - restarting step")
                    
                    memory_manager.update_from_reflection(
                        action_refl=action_refl,
                        traj_refl=traj_refl,
                        global_refl=glob_refl,
                        wrong_hint=wrong_hint or "Action failed for unknown reason",
                        tools_used=tools_used,
                        tool_lesson=tool_lesson_from_eval,
                        decision_branch=decision_branch,
                        failure_type=failure_type
                    )
                    
                    # Get updated memories
                    short_term_memory, long_term_memory = memory_manager.get_memories()
                
                # If we've reached max retries for this step, add to known_pitfalls and move on
                if retries >= self.max_retries_per_step:
                    logger.warning(f"Max retries ({self.max_retries_per_step}) reached for step {step_num}. Moving on.")
                    
                    # Add to known pitfalls
                    if "known_pitfalls" in long_term_memory:
                        pitfall = f"Step {step_num} action failed after {retries} attempts: {short_term_memory.get('last_lesson', '')}"
                        if pitfall not in long_term_memory["known_pitfalls"]:
                            long_term_memory["known_pitfalls"].append(pitfall)
                    
                    # Reset retries and advance step
                    retries = 0
                    update_progress_bar()
                    step_num += 1
            
            # (M) Check for task completion - STRICT RULES:
            # 1. Global reflection explicitly says "complete" AND
            # 2. Verifier confirms with high confidence AND
            # 3. We've reached the ACTUAL last step (step_num >= estimated_steps) AND
            # 4. The predicted action is COMPLETE
            # NEVER terminate early - must complete all steps
            if glob_refl.get("status") == "complete" and glob_refl.get("next_instruction") == "TERMINATE":
                # High confidence score from verifier as confirmation
                global_conf = verifier['match_score']  # simple heuristic: use verifier to back global decision
                
                # STRICT: Only terminate if we're at the actual last step (not second-to-last)
                # And the predicted action must be COMPLETE
                is_at_last_step = step_num >= estimated_steps
                predicted_is_complete = predicted_canonical.get("intent", "").upper() == "COMPLETE"
                
                if global_conf >= self.terminate_confidence and is_at_last_step and predicted_is_complete:
                    logger.info(f"Task finished per global reflection + verifier (step {step_num}/{estimated_steps}). Stopping loop.")
                    finished = True
                else:
                    # If any condition is not met, continue - don't terminate early
                    if glob_refl.get("status") == "complete":
                        if not is_at_last_step:
                            logger.warning(f"Global reflection says complete at step {step_num}/{estimated_steps}, but not at last step. Continuing to step {estimated_steps}...")
                        elif not predicted_is_complete:
                            logger.warning(f"Global reflection says complete at step {step_num}/{estimated_steps}, but predicted action is not COMPLETE. Continuing...")
                        else:
                            logger.warning(f"Global reflection says complete but verifier confidence ({global_conf:.2f}) < threshold ({self.terminate_confidence}). Continuing...")
            
            # Wait a moment to avoid overwhelming the system
            time.sleep(0.1)
        
        # Close progress bar when task completes
        pbar.close()
        
        # If the last step wasn't saved (step_num was incremented but not saved), save it now
        # step_num is now 15 (after incrementing from 14), so we need to save step 14 if it exists
        if step_num > 1 and len(steps_history) < step_num - 1:
            # The last step wasn't saved, save it now with current memory state
            last_step_num = step_num - 1
            logger.info(f"Saving final step {last_step_num} that wasn't captured in loop")
            
            # Get last ground truth if available
            last_ground_truth = trajectory_data.get(last_step_num) if ground_truth_available else None
            
            # Get last image info
            last_image_info = {
                "step_num": last_step_num,
                "has_image": bool(task_images and len(task_images) > 0),
                "image_data_uri": screen_image_url if (task_images and len(task_images) > 0) else "",
                "image_path": "",
                "image_path_absolute": "",
                "image_format": ""
            }
            
            # Save the last step
            last_step_data = format_step_data(
                step_num=last_step_num,
                grounding={"ground_truth": last_ground_truth} if last_ground_truth else {},
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
                predicted_action=predicted_canonical or {},
                ground_truth=last_ground_truth,
                critic_obj={},
                verifier={},
                decision_branch="completed",
                final_action_correct=True,
                image_info=last_image_info,
                used_ground_truth=False
            )
            steps_history.append({
                f"Step {last_step_num}": last_step_data
            })
        
        # Return final result with new structure
        # Convert steps_history from dict format to array format
        steps_array = []
        for step_dict in steps_history:
            if isinstance(step_dict, dict):
                step_keys = [k for k in step_dict.keys() if k.startswith("Step ")]
                if step_keys:
                    steps_array.append(step_dict[step_keys[0]])
                else:
                    # If it's already in the new format, use it directly
                    steps_array.append(step_dict)
        
        result = {
            "user_goal": user_goal,
            "finished": finished,
            "steps_taken": step_num - 1,
            "total_retries": step_num - 1 - (0 if finished else 1),
            "final_short_term_memory": short_term_memory,
            "final_long_term_memory": long_term_memory,
            "trajectory": full_trajectory,
            "steps_history": steps_history,  # Keep dict format for processing
            "steps": steps_array,  # Array format for new structure
            "used_ground_truth_count": used_ground_truth_count,  # Track ground truth fallback usage
            "ground_truth_available": ground_truth_available  # Flag if ground truth was available
        }
        
        return result

def controller_loop_example(user_goal=None, max_tasks=None, dataset=None):
    """
    Run the controller loop with example tasks.
    
    Args:
        user_goal: Single task instruction/goal
        max_tasks: Maximum number of tasks to process
        dataset: HuggingFace dataset
    
    Returns:
        List of task results
    """
    controller = TaskController(use_mock=True, use_enhanced=True)
    return controller.process_tasks(user_goal, max_tasks, dataset)

def process_single_task(user_goal, task_context=None, use_enhanced=True):
    """
    Process a single task through the controller loop.
    
    Args:
        user_goal: The instruction/goal for this task
        task_context: Optional additional context from the dataset
        use_enhanced: Whether to use enhanced components
    
    Returns:
        Dictionary with task results and trajectory
    """
    controller = TaskController(use_mock=True, use_enhanced=use_enhanced)
    return controller.process_single_task(user_goal, task_context)
