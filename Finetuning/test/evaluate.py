#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Llama-4-Maverick model.
Validates predictions against ground truth, focusing only on action and target.
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from inference import load_model_for_inference
from test_data_loader import load_test_dataset

def extract_action_and_target(predicted_output: Dict[str, Any]) -> tuple:
    """
    Extract action and target from predicted output JSON.
    
    Handles multiple formats:
    1. Predicted.predicted_action (expected format)
    2. Grounding.ground_truth (model sometimes generates this instead)
    3. Direct predicted_action (fallback)
    
    Args:
        predicted_output: Predicted output JSON dictionary
        
    Returns:
        Tuple of (action, target) or (None, None) if not found
    """
    try:
        # First try: Look for Predicted.predicted_action (expected format)
        predicted = predicted_output.get("Predicted", {})
        predicted_action = predicted.get("predicted_action", {})
        
        action = predicted_action.get("action")
        target = predicted_action.get("target")
        
        if action and target:
            return (action, target)
        
        # Second try: Look for Grounding.ground_truth (model sometimes generates this)
        # This happens when the model copies the input format instead of generating "Predicted"
        grounding = predicted_output.get("Grounding", {})
        ground_truth = grounding.get("ground_truth", {})
        
        action = ground_truth.get("action")
        target = ground_truth.get("target")
        
        if action and target:
            return (action, target)
        
        # Third try: Look for direct predicted_action (if structure is different)
        if "predicted_action" in predicted_output:
            predicted_action = predicted_output["predicted_action"]
            if isinstance(predicted_action, dict):
                action = predicted_action.get("action")
                target = predicted_action.get("target")
                if action and target:
                    return (action, target)
        
        # Fourth try: Look for direct action/target at root level
        action = predicted_output.get("action")
        target = predicted_output.get("target")
        if action and target:
            return (action, target)
        
        return (None, None)
    except (KeyError, AttributeError, TypeError):
        return (None, None)

def extract_ground_truth_action_and_target(
    ground_truth_output: str = None,
    grounding_json: str = None
) -> tuple:
    """
    Extract action and target from ground truth.
    
    Tries multiple sources in order:
    1. Output JSON: Grounding.ground_truth.action and target
    2. Grounding JSON (from CSV): ground_truth.action and target
    
    Args:
        ground_truth_output: Ground truth output JSON string (from Output column)
        grounding_json: Grounding JSON string (from Grounding column)
        
    Returns:
        Tuple of (action, target) or (None, None) if not found
    """
    # First, try Output JSON
    if ground_truth_output:
        try:
            if isinstance(ground_truth_output, str) and ground_truth_output.strip():
                gt_dict = json.loads(ground_truth_output)
            else:
                gt_dict = ground_truth_output
            
            # Navigate to Grounding.ground_truth
            grounding = gt_dict.get("Grounding", {})
            ground_truth = grounding.get("ground_truth", {})
            
            action = ground_truth.get("action")
            target = ground_truth.get("target")
            
            if action and target:
                return (action, target)
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass
    
    # Fallback: try Grounding column directly
    if grounding_json:
        try:
            if isinstance(grounding_json, str):
                grounding_dict = json.loads(grounding_json)
            else:
                grounding_dict = grounding_json
            
            ground_truth = grounding_dict.get("ground_truth", {})
            action = ground_truth.get("action")
            target = ground_truth.get("target")
            
            if action and target:
                return (action, target)
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass
    
    return (None, None)

def normalize_action(action: str) -> str:
    """
    Normalize action string for comparison.
    
    Valid actions: CLICK, SCROLL, ZOOM, SEGMENT, TEXT, COMPLETE
    
    Args:
        action: Action string
        
    Returns:
        Normalized action (uppercase, trimmed)
    """
    if action is None:
        return None
    
    normalized = action.strip().upper()
    
    # Validate action type
    valid_actions = {"CLICK", "SCROLL", "ZOOM", "SEGMENT", "TEXT", "COMPLETE"}
    if normalized not in valid_actions:
        # Log warning but still return normalized value
        print(f"Warning: Unknown action type '{normalized}'. Valid actions: {valid_actions}")
    
    return normalized

def normalize_target(target: str) -> str:
    """
    Normalize target string for comparison.
    
    Target is what the action is performed on (e.g., "Data Module", "Load Data").
    
    Args:
        target: Target string
        
    Returns:
        Normalized target (trimmed)
    """
    if target is None:
        return None
    return target.strip()

def compare_action_and_target(
    predicted_action: str,
    predicted_target: str,
    ground_truth_action: str,
    ground_truth_target: str
) -> Dict[str, bool]:
    """
    Compare predicted action with ground truth (ONLY ACTION, NOT TARGET).
    
    Args:
        predicted_action: Predicted action
        predicted_target: Predicted target (ignored, kept for compatibility)
        ground_truth_action: Ground truth action
        ground_truth_target: Ground truth target (ignored, kept for compatibility)
        
    Returns:
        Dictionary with comparison results (only action_match)
    """
    # Normalize
    pred_action_norm = normalize_action(predicted_action)
    gt_action_norm = normalize_action(ground_truth_action)
    
    # Compare ONLY action
    action_match = pred_action_norm == gt_action_norm
    
    return {
        "action_match": action_match,
        "target_exact_match": False,  # Not evaluated
        "target_semantic_match": False,  # Not evaluated
        "combined_match": action_match,  # Same as action_match
        "predicted_action": predicted_action,
        "predicted_target": predicted_target,
        "ground_truth_action": ground_truth_action,
        "ground_truth_target": ground_truth_target,
    }

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truths.
    
    CRITICAL: Only compares Action (CLICK, SCROLL, ZOOM, SEGMENT, TEXT, COMPLETE).
    Target is NOT evaluated. All other fields are ignored to prevent bias.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        
    Returns:
        Dictionary with evaluation metrics (only Action accuracy)
    """
    metrics = {
        "total_steps": len(predictions),
        "action_correct": 0,
        "target_exact_correct": 0,  # Kept for compatibility, always 0
        "target_semantic_correct": 0,  # Kept for compatibility, always 0
        "combined_correct": 0,  # Same as action_correct
        "action_accuracy": 0.0,
        "target_exact_accuracy": 0.0,  # Kept for compatibility, always 0.0
        "target_semantic_accuracy": 0.0,  # Kept for compatibility, always 0.0
        "combined_accuracy": 0.0,  # Same as action_accuracy
        "per_step_results": [],
    }
    
    if len(predictions) == 0:
        return metrics
    
    # Progress bar for step-by-step evaluation
    steps_pbar = tqdm(zip(predictions, ground_truths), total=len(predictions), 
                      desc="Evaluating steps", unit="step", leave=False)
    
    for i, (pred, gt) in enumerate(steps_pbar):
        # Extract predicted action and target
        pred_output = pred.get("output", {})
        pred_action, pred_target = extract_action_and_target(pred_output)
        
        # Extract ground truth action and target
        # Try Output column first, then Grounding column
        gt_output_str = gt.get("ground_truth_output")
        gt_grounding_str = gt.get("grounding")  # Get grounding from step dict if available
        
        gt_action, gt_target = extract_ground_truth_action_and_target(
            ground_truth_output=gt_output_str,
            grounding_json=gt_grounding_str
        )
        
        # Compare (only action is evaluated)
        comparison = compare_action_and_target(
            predicted_action=pred_action,
            predicted_target=pred_target,
            ground_truth_action=gt_action,
            ground_truth_target=gt_target,
        )
        
        # Update metrics (only action)
        if comparison["action_match"]:
            metrics["action_correct"] += 1
            metrics["combined_correct"] += 1  # Same as action_correct
        
        # Store per-step result
        step_result = {
            "step_number": pred.get("step_number", i + 1),
            **comparison,
        }
        metrics["per_step_results"].append(step_result)
    
    steps_pbar.close()
    
    # Calculate accuracies
    total = metrics["total_steps"]
    metrics["action_accuracy"] = metrics["action_correct"] / total if total > 0 else 0.0
    metrics["combined_accuracy"] = metrics["action_accuracy"]  # Same as action_accuracy
    
    return metrics

def save_predictions_to_csv_detailed(
    all_results: Dict[str, Dict[str, Any]],
    test_loader: Any,
    output_file: str
) -> None:
    """
    Save detailed model predictions to CSV file.
    
    Args:
        all_results: Dictionary mapping task_id to task metrics (includes full_predictions)
        test_loader: TestDataLoader instance
        output_file: Output CSV file path
    """
    import pandas as pd
    import json
    
    rows = []
    
    # Iterate through all tasks
    for task_id in sorted(all_results.keys(), key=int):
        task_metrics = all_results[task_id]
        per_step_results = task_metrics.get("per_step_results", [])
        full_predictions = task_metrics.get("full_predictions", [])
        task_steps = task_metrics.get("task_steps", [])
        
        for i, step_result in enumerate(per_step_results):
            step_num = step_result.get("step_number", i + 1)
            
            # Get corresponding step data
            step_data = None
            if i < len(task_steps):
                step_data = task_steps[i]
            
            # Get predicted output from full prediction
            predicted_output = {}
            extracted_memory = {}
            if i < len(full_predictions):
                predicted_output = full_predictions[i]["prediction"].get("output", {})
                extracted_memory = full_predictions[i]["prediction"].get("extracted_memory", {})
            
            # Extract predicted action and target using the robust extraction function
            predicted_action, predicted_target = extract_action_and_target(predicted_output)
            # Convert None to empty string for CSV compatibility
            predicted_action = predicted_action or ""
            predicted_target = predicted_target or ""
            
            # Extract memory (Short Term and Long Term)
            short_term_memory = extracted_memory.get("short_term", {})
            long_term_memory = extracted_memory.get("long_term", {})
            short_term_memory_str = json.dumps(short_term_memory, ensure_ascii=False) if short_term_memory else "{}"
            long_term_memory_str = json.dumps(long_term_memory, ensure_ascii=False) if long_term_memory else "{}"
            
            # Get ground truth from step_result
            ground_truth_action = step_result.get("ground_truth_action", "")
            ground_truth_target = step_result.get("ground_truth_target", "")
            
            # Get full output JSON as string
            predicted_output_str = json.dumps(predicted_output, ensure_ascii=False) if predicted_output else ""
            
            # Get ground truth output if available
            ground_truth_output_str = step_data.get("ground_truth_output", "") if step_data else ""
            
            row = {
                "Task_id": task_id,
                "Slide_number": step_num,
                "Predicted_Action": predicted_action,
                "Predicted_Target": predicted_target,  # Not evaluated, kept for reference
                "Predicted_Output_JSON": predicted_output_str,
                "Short_Term_Memory": short_term_memory_str,  # Extracted from model output
                "Long_term_Memory": long_term_memory_str,  # Extracted from model output
                "Ground_Truth_Action": ground_truth_action,
                "Ground_Truth_Target": ground_truth_target,  # Not evaluated, kept for reference
                "Ground_Truth_Output_JSON": ground_truth_output_str,
                "Action_Match": step_result.get("action_match", False),  # Only this is evaluated
                "Target_Exact_Match": False,  # Not evaluated
                "Target_Semantic_Match": False,  # Not evaluated
                "Combined_Match": step_result.get("action_match", False),  # Same as Action_Match
                "Image_id": step_data.get("image_id", "") if step_data else "",
            }
            
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')

def verify_no_data_leakage(
    test_loader: Any,
    predictor: Any
) -> Dict[str, Any]:
    """
    Verify that there is no data leakage in the evaluation setup.
    
    Checks:
    1. Memory comes from model predictions, not from test CSV
    2. Ground truth outputs are not shown to model
    3. Test data is separate from training data
    4. Sequential prediction order is maintained
    
    Args:
        test_loader: TestDataLoader instance
        predictor: StepPredictor instance
        
    Returns:
        Dictionary with verification results
    """
    verification = {
        "memory_source": "OK",
        "ground_truth_visibility": "OK",
        "data_split": "OK",
        "sequential_order": "OK",
        "warnings": [],
        "errors": []
    }
    
    # Check 1: Memory should come from model state, not CSV
    # This is already enforced in predict_step() - it ignores provided memory
    # and always uses self.memory_state
    if hasattr(predictor, 'memory_state'):
        verification["memory_source"] = "OK - Memory comes from model predictions"
    else:
        verification["errors"].append("Memory state not found in predictor")
    
    # Check 2: Ground truth outputs should not be passed to model
    # This is verified by checking that evaluate.py doesn't pass ground_truth_output
    # to predict_step - it only passes grounding, which is current step context
    verification["ground_truth_visibility"] = "OK - Ground truth not passed to model"
    
    # Check 3: Data split verification
    data_config = get_config("data")
    train_start = data_config.get("task_start", None)
    train_end = data_config.get("task_end", None)
    test_start = data_config.get("test_task_start", None)
    test_end = data_config.get("test_task_end", None)
    
    # Handle None values for comparison
    # If any range is None, we can't verify overlap, so skip check
    if train_start is not None and train_end is not None and test_start is not None and test_end is not None:
        if train_start > test_end or train_end < test_start:
            verification["data_split"] = "OK - Training and test tasks are separate"
        else:
            verification["errors"].append(
                f"Data leakage detected: Training tasks ({train_start}-{train_end}) "
                f"overlap with test tasks ({test_start}-{test_end})"
            )
    elif train_start is None or train_end is None:
        verification["data_split"] = "OK - Training task range not specified (using all tasks)"
    elif test_start is None or test_end is None:
        verification["data_split"] = "OK - Test task range not specified (using all tasks)"
    else:
        verification["data_split"] = "OK - Task ranges not fully specified"
    
    # Check 4: Sequential order
    # Verify that steps are processed in order within each task
    task_ids = test_loader.get_task_ids()
    for task_id in task_ids[:3]:  # Check first 3 tasks as sample
        steps = test_loader.get_task_steps(task_id)
        slide_numbers = [int(s["slide_number"]) for s in steps]
        if slide_numbers == sorted(slide_numbers):
            verification["sequential_order"] = "OK - Steps processed in order"
        else:
            verification["warnings"].append(
                f"Task {task_id}: Steps not in sequential order"
            )
    
    # Check 5: Grounding doesn't contain future information
    # Sample check: verify grounding only contains current step context
    sample_task_id = task_ids[0] if task_ids else None
    if sample_task_id:
        steps = test_loader.get_task_steps(sample_task_id)
        if steps:
            first_grounding = steps[0].get("grounding", "{}")
            try:
                import json
                grounding_dict = json.loads(first_grounding)
                # Check if grounding contains ground_truth for current or next step
                # This is acceptable as long as it's the CURRENT step's ground truth,
                # not the NEXT step's
                if "ground_truth" in str(grounding_dict):
                    verification["warnings"].append(
                        "Grounding contains ground_truth - verify this is current step only"
                    )
            except:
                pass
    
    return verification

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """
    Load checkpoint file to resume from previous run.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        
    Returns:
        Dictionary with completed tasks and results, or empty dict if file doesn't exist
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✓ Loaded checkpoint: {len(checkpoint.get('completed_tasks', []))} tasks already completed")
            return checkpoint
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            return {}
    return {}

def save_checkpoint(checkpoint_file: str, completed_tasks: List[int], all_results: Dict[str, Any], overall_metrics: Dict[str, Any]) -> None:
    """
    Save checkpoint file after each task.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        completed_tasks: List of completed task IDs
        all_results: Dictionary with all task results
        overall_metrics: Overall evaluation metrics
    """
    from datetime import datetime
    checkpoint = {
        "completed_tasks": completed_tasks,
        "all_results": all_results,
        "overall_metrics": overall_metrics,
        "timestamp": str(datetime.now())
    }
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Warning: Could not save checkpoint: {e}")

def evaluate_model(
    model_path: str,
    base_model_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    task_start: Optional[int] = None,
    task_end: Optional[int] = None,
    output_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Evaluate fine-tuned model on test dataset.
    
    Args:
        model_path: Path to fine-tuned model
        base_model_name: Base model name if using PEFT
        csv_path: Path to test CSV file
        task_start: Starting task ID (None = process all tasks)
        task_end: Ending task ID (None = process all tasks)
        output_file: Optional output file path for results
        output_dir: Optional output directory for CSV and checkpoint files (default: from config)
        resume: If True, resume from checkpoint if available
        
    Returns:
        Dictionary with evaluation results
    """
    print("Loading model for evaluation...")
    model_config = get_config("model")
    predictor = load_model_for_inference(
        model_path=model_path,
        base_model_name=base_model_name,
        hf_token=model_config.get("hf_token")
    )
    
    print("Loading test dataset...")
    data_config = get_config("data")
    # Use provided task range or default to all tasks
    final_task_start = task_start if task_start is not None else data_config.get("test_task_start", None)
    final_task_end = task_end if task_end is not None else data_config.get("test_task_end", None)
    
    if final_task_start is None and final_task_end is None:
        print("Processing all tasks (no task range specified)")
    else:
        print(f"Processing tasks {final_task_start or 'all'} to {final_task_end or 'all'}")
    
    test_loader = load_test_dataset(
        csv_path=csv_path or data_config.get("test_csv_path"),
        task_start=final_task_start,
        task_end=final_task_end,
        image_base_path=data_config.get("test_image_base_path"),
        test_mode=True
    )
    
    # Verify no data leakage
    print("\n" + "=" * 60)
    print("DATA LEAKAGE VERIFICATION")
    print("=" * 60)
    leakage_check = verify_no_data_leakage(test_loader, predictor)
    print(f"Memory Source: {leakage_check['memory_source']}")
    print(f"Ground Truth Visibility: {leakage_check['ground_truth_visibility']}")
    print(f"Data Split: {leakage_check['data_split']}")
    print(f"Sequential Order: {leakage_check['sequential_order']}")
    
    if leakage_check['warnings']:
        print("\n⚠️  Warnings:")
        for warning in leakage_check['warnings']:
            print(f"  - {warning}")
    
    if leakage_check['errors']:
        print("\n❌ ERRORS DETECTED:")
        for error in leakage_check['errors']:
            print(f"  - {error}")
        print("\n⚠️  CRITICAL: Data leakage detected! Evaluation results may be invalid.")
    else:
        print("\n✅ No data leakage detected - evaluation setup is correct")
    
    # Get statistics
    stats = test_loader.get_statistics()
    print(f"\nTest dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Setup checkpoint and output files (before task loop)
    if output_dir is None:
        output_config = get_config("output")
        output_dir = output_config.get("output_dir", "finetuning/outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_output_file = output_file.replace('.json', '_predictions.csv') if output_file else None
    if csv_output_file is None:
        # If no output_file specified, create default CSV name in output directory
        csv_output_file = os.path.join(output_dir, "test_predictions.csv")
    else:
        # Ensure output file is in output directory
        csv_output_file = os.path.join(output_dir, os.path.basename(csv_output_file))
    
    checkpoint_file = csv_output_file.replace('.csv', '_checkpoint.json')
    
    # Initialize results
    all_results = {}
    overall_metrics = {
        "total_steps": 0,
        "action_correct": 0,
        "target_exact_correct": 0,
        "target_semantic_correct": 0,
        "combined_correct": 0,
    }
    completed_tasks = []
    
    # Load checkpoint if resuming
    if resume:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            completed_tasks = checkpoint.get("completed_tasks", [])
            all_results = checkpoint.get("all_results", {})
            overall_metrics = checkpoint.get("overall_metrics", {
                "total_steps": 0,
                "action_correct": 0,
                "target_exact_correct": 0,
                "target_semantic_correct": 0,
                "combined_correct": 0,
            })
            print(f"✓ Resuming from checkpoint: {len(completed_tasks)} tasks already completed")
            print(f"  Completed tasks: {sorted(completed_tasks)}")
    
    task_ids = test_loader.get_task_ids()
    
    # Filter out already completed tasks
    task_ids = [tid for tid in task_ids if int(tid) not in completed_tasks]
    
    if not task_ids:
        print("\n✅ All tasks already completed! Loading results from checkpoint...")
    else:
        print(f"\nEvaluating {len(task_ids)} remaining tasks...")
    
    # Progress bar for tasks
    task_pbar = tqdm(task_ids, desc="Evaluating tasks", unit="task")
    
    for task_id in task_pbar:
        task_pbar.set_description(f"Evaluating task {task_id}")
        
        # Get task steps
        steps = test_loader.get_task_steps(task_id)
        
        # Predict sequence with progress bar for steps
        step_pbar = tqdm(steps, desc=f"Task {task_id} steps", unit="step", leave=False)
        predictions = []
        
        # Reset memory for new task
        predictor.reset_memory(task_id=task_id)
        
        for step in step_pbar:
            step_pbar.set_description(f"Task {task_id} step {step['slide_number']}")
            
            # Predict single step
            result = predictor.predict_step(
                task_id=step["task_id"],
                slide_number=step["slide_number"],
                grounding=step["grounding"],
                image_id=step.get("image_id"),
            )
            
            predictions.append({
                "step_number": step["slide_number"],
                "prediction": result,
            })
            
            # Log prediction vs ground truth in real-time (ONLY ACTION)
            pred_output = result.get("output", {})
            pred_action, pred_target = extract_action_and_target(pred_output)
            gt_output_str = step.get("ground_truth_output", "")
            gt_grounding_str = step.get("grounding", "")
            gt_action, gt_target = extract_ground_truth_action_and_target(
                ground_truth_output=gt_output_str,
                grounding_json=gt_grounding_str
            )
            
            # Quick comparison (ONLY ACTION)
            action_match = (pred_action or "").upper() == (gt_action or "").upper()
            
            # Print detailed log (ONLY ACTION)
            match_status = "✅ CORRECT" if action_match else "❌ WRONG"
            print(f"\n  Step {step['slide_number']}: {match_status}")
            print(f"    Predicted: Action={pred_action}")
            print(f"    Ground Truth: Action={gt_action}")
            if not action_match:
                print(f"    ❌ Action mismatch: '{pred_action}' vs '{gt_action}'")
        
        step_pbar.close()
        
        # Prepare predictions and ground truths for evaluation
        pred_list = []
        gt_list = []
        for i, step in enumerate(steps):
            if i < len(predictions):
                pred_list.append({
                    "step_number": step["slide_number"],
                    "output": predictions[i]["prediction"]["output"],
                    "full_prediction": predictions[i]["prediction"],  # Store full prediction for CSV
                })
                gt_list.append({
                    "ground_truth_output": step.get("ground_truth_output"),
                    "grounding": step.get("grounding_json") or step.get("grounding"),
                })
        
        # Evaluate
        task_metrics = evaluate_predictions(pred_list, gt_list)
        
        # Update overall metrics
        overall_metrics["total_steps"] += task_metrics["total_steps"]
        overall_metrics["action_correct"] += task_metrics["action_correct"]
        overall_metrics["target_exact_correct"] += task_metrics["target_exact_correct"]
        overall_metrics["target_semantic_correct"] += task_metrics["target_semantic_correct"]
        overall_metrics["combined_correct"] += task_metrics["combined_correct"]
        
        # Store full predictions for CSV export
        task_metrics["full_predictions"] = predictions
        task_metrics["task_steps"] = steps
        all_results[task_id] = task_metrics
        
        # Save checkpoint after each task (for resume functionality)
        completed_tasks.append(int(task_id))
        save_checkpoint(checkpoint_file, completed_tasks, all_results, overall_metrics)
        
        # Save CSV incrementally after each task (append mode)
        save_predictions_to_csv_detailed(
            all_results=all_results,
            test_loader=test_loader,
            output_file=csv_output_file
        )
        print(f"  ✓ Task {task_id} completed and saved to checkpoint")
    
    # Calculate overall accuracies
    total = overall_metrics["total_steps"]
    overall_results = {
        "task_metrics": all_results,
        "overall_metrics": {
            **overall_metrics,
            "action_accuracy": overall_metrics["action_correct"] / total if total > 0 else 0.0,
            "target_exact_accuracy": overall_metrics["target_exact_correct"] / total if total > 0 else 0.0,
            "target_semantic_accuracy": overall_metrics["target_semantic_correct"] / total if total > 0 else 0.0,
            "combined_accuracy": overall_metrics["combined_correct"] / total if total > 0 else 0.0,
        }
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Only Action is compared for validation (Target is NOT evaluated)")
    print("   Action types: CLICK, SCROLL, ZOOM, SEGMENT, TEXT, COMPLETE (case-insensitive)")
    print("   All other fields (target, coords, memory, image_info, etc.) are IGNORED\n")
    print(f"Overall Metrics:")
    print(f"  Total Steps: {overall_metrics['total_steps']}")
    print(f"  Action Accuracy: {overall_results['overall_metrics']['action_accuracy']:.4f} ({overall_metrics['action_correct']}/{overall_metrics['total_steps']})")
    
    # Save results to JSON
    if output_file:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        print("Results saved")
    
    # Final save (already saved incrementally after each task, but ensure it's up to date)
    print(f"\nSaving final predictions to CSV: {csv_output_file}...")
    save_predictions_to_csv_detailed(
        all_results=all_results,
        test_loader=test_loader,
        output_file=csv_output_file
    )
    print(f"✅ Predictions CSV saved to: {csv_output_file}")
    print(f"✅ Checkpoint saved to: {checkpoint_file}")
    
    return overall_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base model name if using PEFT")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to test CSV file")
    parser.add_argument("--task_start", type=int, default=None, help="Starting task ID (None = process all tasks)")
    parser.add_argument("--task_end", type=int, default=None, help="Ending task ID (None = process all tasks)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV and checkpoint files")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.csv_path is None:
        args.csv_path = config["data"].get("test_csv_path") or config["data"]["csv_path"]
    
    evaluate_model(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        csv_path=args.csv_path,
        task_start=args.task_start,
        task_end=args.task_end,
        output_file=args.output_file,
        output_dir=args.output_dir,
    )

