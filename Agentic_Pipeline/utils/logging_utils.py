#!/usr/bin/env python3
"""
Logging utilities for the Medical Visual Agent system.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, Any, Optional, List, Union
import time
import re
from logging.handlers import RotatingFileHandler

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from config import get_config

# ANSI color codes for terminal output
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'BRIGHT_BLACK': '\033[90m',
    'BRIGHT_RED': '\033[91m',
    'BRIGHT_GREEN': '\033[92m',
    'BRIGHT_YELLOW': '\033[93m',
    'BRIGHT_BLUE': '\033[94m',
    'BRIGHT_MAGENTA': '\033[95m',
    'BRIGHT_CYAN': '\033[96m',
    'BRIGHT_WHITE': '\033[97m',
    'BG_BLACK': '\033[40m',
    'BG_RED': '\033[41m',
    'BG_GREEN': '\033[42m',
    'BG_YELLOW': '\033[43m',
    'BG_BLUE': '\033[44m',
    'BG_MAGENTA': '\033[45m',
    'BG_CYAN': '\033[46m',
    'BG_WHITE': '\033[47m',
}

# Status icons
ICONS = {
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'info': 'ℹ️',
    'progress': '🔄',
    'stats': '📊',
    'check': '✓',
    'cross': '✗',
    'question': '❓',
    'star': '⭐',
    'tool': '🔧',
    'step': '➤',
    'task': '📋',
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors and icons to log output."""
    
    level_colors = {
        logging.DEBUG: COLORS['BRIGHT_BLACK'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['BG_RED'] + COLORS['WHITE'] + COLORS['BOLD'],
    }
    
    level_icons = {
        logging.DEBUG: ICONS['info'],
        logging.INFO: ICONS['info'],
        logging.WARNING: ICONS['warning'],
        logging.ERROR: ICONS['error'],
        logging.CRITICAL: ICONS['error'],
    }

    def format(self, record):
        # Save original message
        original_msg = record.getMessage()
        
        # Check for special prefixes to override default styling
        icon = None
        color = None
        
        # Special handling for different message types
        if original_msg.startswith('STEP'):
            icon = ICONS['step']
            color = COLORS['BRIGHT_CYAN'] + COLORS['BOLD']
        elif original_msg.startswith('TASK'):
            icon = ICONS['task']
            color = COLORS['BRIGHT_BLUE'] + COLORS['BOLD']
        elif original_msg.startswith('PROGRESS'):
            icon = ICONS['progress']
            color = COLORS['BRIGHT_BLUE']
        elif original_msg.startswith('STATS'):
            icon = ICONS['stats']
            color = COLORS['CYAN']
        elif original_msg.startswith('TOOL'):
            icon = ICONS['tool']
            color = COLORS['MAGENTA']
        elif original_msg.startswith('SUCCESS'):
            icon = ICONS['success']
            color = COLORS['BRIGHT_GREEN']
        
        # Use default level-based styling if no special prefix
        if color is None:
            color = self.level_colors.get(record.levelno, COLORS['RESET'])
        if icon is None:
            icon = self.level_icons.get(record.levelno, ICONS['info'])
        
        # Format the message
        levelname = record.levelname.ljust(8)
        record.levelname = f"{color}{levelname}{COLORS['RESET']}"
        
        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)
        
        # Add icon and color to the message
        name_str = f"{record.name.split('.')[-1]}"
        record.msg = f"{icon} {color}{record.msg}{COLORS['RESET']}"
        
        # Format using the parent formatter
        formatted = super().format(record)
        
        # Reset record message to original
        record.msg = original_msg
        
        return formatted

# Configure root logger with colored output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the Medical Visual Agent system with colored output.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("medical_visual_agent")
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create colored formatter for console
    colored_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create plain formatter for file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"medical_visual_agent_{timestamp}.log")
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)  # Use plain formatter for file
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    return logger

class PerformanceTracker:
    """
    Track performance metrics for the Medical Visual Agent system.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics = {
            "task_success_rate": 0.0,
            "avg_steps_per_task": 0.0,
            "avg_retries_per_step": 0.0,
            "failure_types": {},
            "tool_effectiveness": {},
            "synthetic_dataset_quality": 0.0
        }
        
        self.task_results = []
        self.step_times = []
        self.start_time = time.time()
    
    def start_task(self) -> None:
        """Mark the start of a task."""
        self.start_time = time.time()
    
    def end_task(self, result: Dict[str, Any]) -> None:
        """
        Mark the end of a task and update metrics.
        
        Args:
            result: Task result dictionary
        """
        # Calculate task duration
        duration = time.time() - self.start_time
        
        # Add result to task_results
        result_with_duration = result.copy()
        result_with_duration["duration"] = duration
        self.task_results.append(result_with_duration)
        
        # Update metrics
        self._update_metrics()
    
    def start_step(self) -> None:
        """Mark the start of a step."""
        self.step_start_time = time.time()
    
    def end_step(
        self,
        success: bool,
        retries: int,
        failure_type: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        match_score: Optional[float] = None
    ) -> None:
        """
        Mark the end of a step and update metrics.
        
        Args:
            success: Whether the step was successful
            retries: Number of retries for this step
            failure_type: Type of failure if not successful
            tools_used: List of tools used in this step
            match_score: Match score from verifier
        """
        # Calculate step duration
        duration = time.time() - self.step_start_time
        
        # Add to step_times
        self.step_times.append({
            "duration": duration,
            "success": success,
            "retries": retries,
            "failure_type": failure_type,
            "tools_used": tools_used,
            "match_score": match_score
        })
        
        # Update metrics
        self._update_metrics()
    
    def update_tool_effectiveness(
        self,
        tool_name: str,
        success: bool,
        duration: float,
        match_score: Optional[float] = None
    ) -> None:
        """
        Update tool effectiveness metrics.
        
        Args:
            tool_name: Name of the tool
            success: Whether the tool usage was successful
            duration: Duration of tool usage
            match_score: Match score after tool usage
        """
        if tool_name not in self.metrics["tool_effectiveness"]:
            self.metrics["tool_effectiveness"][tool_name] = {
                "calls": 0,
                "successful_calls": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "success_rate": 0.0,
                "avg_match_score": 0.0,
                "total_match_score": 0.0
            }
        
        tool_stats = self.metrics["tool_effectiveness"][tool_name]
        tool_stats["calls"] += 1
        if success:
            tool_stats["successful_calls"] += 1
        
        tool_stats["total_duration"] += duration
        tool_stats["avg_duration"] = tool_stats["total_duration"] / tool_stats["calls"]
        tool_stats["success_rate"] = tool_stats["successful_calls"] / tool_stats["calls"]
        
        if match_score is not None:
            tool_stats["total_match_score"] += match_score
            tool_stats["avg_match_score"] = tool_stats["total_match_score"] / tool_stats["calls"]
    
    def _update_metrics(self) -> None:
        """Update aggregated metrics."""
        # Task success rate
        successful_tasks = sum(1 for result in self.task_results if result.get("finished", False))
        self.metrics["task_success_rate"] = successful_tasks / len(self.task_results) if self.task_results else 0.0
        
        # Average steps per task
        total_steps = sum(result.get("steps_taken", 0) for result in self.task_results)
        self.metrics["avg_steps_per_task"] = total_steps / len(self.task_results) if self.task_results else 0.0
        
        # Average retries per step
        total_retries = sum(step.get("retries", 0) for step in self.step_times)
        self.metrics["avg_retries_per_step"] = total_retries / len(self.step_times) if self.step_times else 0.0
        
        # Failure types
        self.metrics["failure_types"] = {}
        for step in self.step_times:
            failure_type = step.get("failure_type")
            if failure_type and not step.get("success", False):
                if failure_type not in self.metrics["failure_types"]:
                    self.metrics["failure_types"][failure_type] = 0
                self.metrics["failure_types"][failure_type] += 1
        
        # Synthetic dataset quality - percentage of steps with match_score >= 0.9
        high_quality_steps = sum(1 for step in self.step_times if step.get("match_score", 0.0) >= 0.9)
        self.metrics["synthetic_dataset_quality"] = high_quality_steps / len(self.step_times) if self.step_times else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def save_metrics(self, output_path: str) -> None:
        """
        Save performance metrics to file.
        
        Args:
            output_path: Path to save metrics
        """
        with open(output_path, 'w') as f:
            json.dump({
                "metrics": self.metrics,
                "task_results": self.task_results,
                "step_times": self.step_times,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)

def format_separator(title: str = None, width: int = 80, char: str = "=") -> str:
    """
    Create a separator line with an optional title.
    
    Args:
        title: Title to include in the separator
        width: Width of the separator line
        char: Character to use for the separator
        
    Returns:
        Formatted separator string
    """
    if title:
        # Calculate padding needed on each side
        title_with_space = f" {title} "
        padding = (width - len(title_with_space)) // 2
        return f"\n{char * padding}{title_with_space}{char * padding}"
    return f"\n{char * width}"

def format_step_header(step_num: int, total_steps: Optional[int] = None) -> str:
    """
    Format a step header with step number and optional total.
    
    Args:
        step_num: Current step number
        total_steps: Total number of steps (optional)
        
    Returns:
        Formatted step header
    """
    if total_steps:
        return f"STEP {step_num}/{total_steps}"
    return f"STEP {step_num}"

def format_task_header(task_num: int, total_tasks: Optional[int] = None, task_id: str = None) -> str:
    """
    Format a task header with task number and optional total.
    
    Args:
        task_num: Current task number
        total_tasks: Total number of tasks (optional)
        task_id: Task identifier (optional)
        
    Returns:
        Formatted task header
    """
    if task_id and total_tasks:
        return f"TASK {task_num}/{total_tasks} - {task_id}"
    elif task_id:
        return f"TASK {task_num} - {task_id}"
    elif total_tasks:
        return f"TASK {task_num}/{total_tasks}"
    return f"TASK {task_num}"

def format_progress(current: int, total: int, label: str = "Progress") -> str:
    """
    Format a progress message with percentage.
    
    Args:
        current: Current progress value
        total: Total progress value
        label: Label for the progress (optional)
        
    Returns:
        Formatted progress message
    """
    percentage = int(100 * current / total) if total > 0 else 0
    return f"PROGRESS {label}: {current}/{total} ({percentage}%)"

def format_json_compact(data: Union[Dict[str, Any], List[Any]]) -> str:
    """
    Format JSON data in a compact but readable way.
    
    Args:
        data: JSON data to format
        
    Returns:
        Formatted JSON string
    """
    if not data:
        return "{}"
        
    # For small objects, use a compact single line representation
    json_str = json.dumps(data, separators=(',', ':'))
    if len(json_str) < 80:
        return json_str
    
    # For larger objects, create a simplified summary
    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) <= 5:
            # Show up to 5 keys with compact values
            parts = []
            for key in keys:
                value = data[key]
                value_str = _compact_value(value)
                parts.append(f"{key}: {value_str}")
            return "{" + ", ".join(parts) + "}"
        else:
            # Just show keys for larger objects
            return "{" + ", ".join(keys[:5]) + f", ... ({len(keys)}) keys total"
    
    elif isinstance(data, list):
        if len(data) <= 5:
            # Show up to 5 items in compact form
            parts = [_compact_value(item) for item in data[:5]]
            return "[" + ", ".join(parts) + "]"
        else:
            # Show summary for longer lists
            return f"[... {len(data)} items]"
    
    return str(data)

def _compact_value(value: Any) -> str:
    """Helper to create compact string representation of values."""
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    elif isinstance(value, str):
        if len(value) > 30:
            return f'"{value[:27]}..."'
        return f'"{value}"'
    else:
        return str(value)

def format_decision(decision: str, match_score: float, checks: Dict[str, float]) -> str:
    """
    Format a decision summary with match score and checks.
    
    Args:
        decision: Decision string (e.g., "A_both_agree_correct")
        match_score: Match score (0.0-1.0)
        checks: Dictionary of check results
        
    Returns:
        Formatted decision message
    """
    # Convert decision to human-readable form
    decision_map = {
        "A_both_agree_correct": "Both agree correct",
        "B_critic_correct_verifier_unsure": "Critic correct, verifier unsure",
        "C_verifier_correct_critic_unsure": "Verifier correct, critic unsure",
        "D_both_unsure_but_match": "Both unsure but match",
        "E_neither_match_use_verifier": "Mismatch - using verifier",
        "F_neither_match_use_critic": "Mismatch - using critic",
        "X_hard_fail": "Hard failure",
    }
    
    human_decision = decision_map.get(decision, decision)
    
    # Create check indicators
    check_icons = []
    for check_name, check_value in checks.items():
        icon = ICONS['check'] if check_value >= 0.8 else (ICONS['question'] if check_value >= 0.5 else ICONS['cross'])
        check_icons.append(f"{check_name}: {icon}")
    
    # Format the final message
    if match_score >= 0.85:
        decision_prefix = "SUCCESS"
    elif match_score >= 0.6:
        decision_prefix = "PROGRESS"
    else:
        decision_prefix = "WARNING"
        
    return f"{decision_prefix} Decision: {human_decision} (Score: {match_score:.2f}) [{', '.join(check_icons)}]"

class AlertSystem:
    """
    Alert system for the Medical Visual Agent system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the alert system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.alerts = []
        self.logger = logging.getLogger("medical_visual_agent.alerts")
        
        # Alert thresholds
        self.thresholds = {
            "retry_rate": 0.2,                 # Alert if retry rate > 20%
            "parsing_failure_rate": 0.01,      # Alert if parsing failure rate > 1%
            "tool_confidence_variance": 0.2,   # Alert if tool confidence variance > 0.2
            "task_success_drop": 0.1,          # Alert if task success rate drops by 10%
            "step_loops": 3,                   # Alert if step loops > 3 times
        }
        
        # Override from config if provided
        if "alert_thresholds" in self.config:
            self.thresholds.update(self.config["alert_thresholds"])
    
    def check_retry_rate(self, retry_rate: float) -> bool:
        """
        Check if retry rate exceeds threshold.
        
        Args:
            retry_rate: Current retry rate
        
        Returns:
            True if alert triggered, False otherwise
        """
        if retry_rate > self.thresholds["retry_rate"]:
            alert = {
                "type": "retry_rate",
                "message": f"Retry rate of {retry_rate:.2f} exceeds threshold of {self.thresholds['retry_rate']:.2f}",
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": "warning"
            }
            self.alerts.append(alert)
            self.logger.warning(alert["message"])
            return True
        
        return False
    
    def check_parsing_failures(self, parsing_failure_rate: float) -> bool:
        """
        Check if parsing failure rate exceeds threshold.
        
        Args:
            parsing_failure_rate: Current parsing failure rate
        
        Returns:
            True if alert triggered, False otherwise
        """
        if parsing_failure_rate > self.thresholds["parsing_failure_rate"]:
            alert = {
                "type": "parsing_failure_rate",
                "message": f"Parsing failure rate of {parsing_failure_rate:.2f} exceeds threshold of {self.thresholds['parsing_failure_rate']:.2f}",
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": "warning"
            }
            self.alerts.append(alert)
            self.logger.warning(alert["message"])
            return True
        
        return False
    
    def check_tool_confidence(self, tool_name: str, confidence_variance: float) -> bool:
        """
        Check if tool confidence variance exceeds threshold.
        
        Args:
            tool_name: Name of the tool
            confidence_variance: Variance in tool confidence
        
        Returns:
            True if alert triggered, False otherwise
        """
        if confidence_variance > self.thresholds["tool_confidence_variance"]:
            alert = {
                "type": "tool_confidence_variance",
                "message": f"Tool {tool_name} has confidence variance of {confidence_variance:.2f}, exceeding threshold of {self.thresholds['tool_confidence_variance']:.2f}",
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": "warning"
            }
            self.alerts.append(alert)
            self.logger.warning(alert["message"])
            return True
        
        return False
    
    def check_task_success_drop(
        self,
        current_success_rate: float,
        previous_success_rate: float
    ) -> bool:
        """
        Check if task success rate has dropped significantly.
        
        Args:
            current_success_rate: Current task success rate
            previous_success_rate: Previous task success rate
        
        Returns:
            True if alert triggered, False otherwise
        """
        drop = previous_success_rate - current_success_rate
        if drop > self.thresholds["task_success_drop"]:
            alert = {
                "type": "task_success_drop",
                "message": f"Task success rate dropped by {drop:.2f}, from {previous_success_rate:.2f} to {current_success_rate:.2f}",
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": "error"
            }
            self.alerts.append(alert)
            self.logger.error(alert["message"])
            return True
        
        return False
    
    def check_step_loops(self, step_num: int, retries: int) -> bool:
        """
        Check if a step is looping too many times.
        
        Args:
            step_num: Current step number
            retries: Number of retries for this step
        
        Returns:
            True if alert triggered, False otherwise
        """
        if retries > self.thresholds["step_loops"]:
            alert = {
                "type": "step_loops",
                "message": f"Step {step_num} has looped {retries} times, exceeding threshold of {self.thresholds['step_loops']}",
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": "warning"
            }
            self.alerts.append(alert)
            self.logger.warning(alert["message"])
            return True
        
        return False
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all alerts.
        
        Returns:
            List of alerts
        """
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []
    
    def save_alerts(self, output_path: str) -> None:
        """
        Save alerts to file.
        
        Args:
            output_path: Path to save alerts
        """
        with open(output_path, 'w') as f:
            json.dump({
                "alerts": self.alerts,
                "thresholds": self.thresholds,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)
