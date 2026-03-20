#!/usr/bin/env python3
"""
Reflection module for the Medical Visual Agent system.
This module processes hierarchical reflections.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional, Union, Tuple

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from config import get_config

logger = logging.getLogger(__name__)

class ReflectionProcessor:
    """
    Reflection processor that handles hierarchical reflections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reflection processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config("agent")
    
    def process_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback from the feedback agent.
        
        Args:
            feedback: Raw feedback dictionary
        
        Returns:
            Processed feedback dictionary
        """
        # Ensure we have all required fields with defaults
        processed_feedback = {
            "reflection.action": self._get_action_reflection(feedback),
            "reflection.trajectory": self._get_trajectory_reflection(feedback),
            "reflection.global": self._get_global_reflection(feedback),
            "action_correct": feedback.get("action_correct", False),
            "why_if_wrong": feedback.get("why_if_wrong", ""),
            "hint_if_wrong": feedback.get("hint_if_wrong", ""),
            "tool_evaluation": self._get_tool_evaluation(feedback)
        }
        
        return processed_feedback
    
    def _get_action_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract action reflection from feedback.
        
        Args:
            feedback: Raw feedback dictionary
        
        Returns:
            Action reflection dictionary
        """
        action_refl = feedback.get("reflection.action", {})
        
        # Ensure all required fields are present
        default_action = {
            "last_action": "NONE",
            "last_observation": "NONE",
            "last_lesson": "NONE"
        }
        
        # If we didn't get a dictionary, try to find the fields directly in feedback
        if not isinstance(action_refl, dict):
            action_refl = {
                "last_action": feedback.get("last_action", "NONE"),
                "last_observation": feedback.get("last_observation", "NONE"),
                "last_lesson": feedback.get("last_lesson", "NONE")
            }
        
        # Apply defaults for missing fields
        for key, value in default_action.items():
            if key not in action_refl or not action_refl[key]:
                action_refl[key] = value
        
        return action_refl
    
    def _get_trajectory_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract trajectory reflection from feedback.
        
        Args:
            feedback: Raw feedback dictionary
        
        Returns:
            Trajectory reflection dictionary
        """
        traj_refl = feedback.get("reflection.trajectory", {})
        
        # Ensure all required fields are present
        default_traj = {
            "overall_progress": "",
            "completed_subtasks": [],
            "remaining_subtasks": [],
            "known_pitfalls": []
        }
        
        # If we didn't get a dictionary, create an empty one
        if not isinstance(traj_refl, dict):
            traj_refl = {}
        
        # Apply defaults for missing fields
        for key, value in default_traj.items():
            if key not in traj_refl or traj_refl[key] is None:
                traj_refl[key] = value
        
        # Ensure lists are actually lists
        for key in ["completed_subtasks", "remaining_subtasks", "known_pitfalls"]:
            if not isinstance(traj_refl.get(key), list):
                if traj_refl.get(key):
                    # Try to convert to list if it's a string
                    if isinstance(traj_refl.get(key), str):
                        traj_refl[key] = [traj_refl[key]]
                    else:
                        traj_refl[key] = []
                else:
                    traj_refl[key] = []
        
        return traj_refl
    
    def _get_global_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract global reflection from feedback.
        
        Args:
            feedback: Raw feedback dictionary
        
        Returns:
            Global reflection dictionary
        """
        global_refl = feedback.get("reflection.global", {})
        
        # Ensure all required fields are present
        default_global = {
            "status": "incomplete",
            "missing_steps": [],
            "next_instruction": "CONTINUE"
        }
        
        # If we didn't get a dictionary, create an empty one
        if not isinstance(global_refl, dict):
            global_refl = {}
        
        # Apply defaults for missing fields
        for key, value in default_global.items():
            if key not in global_refl or global_refl[key] is None:
                global_refl[key] = value
        
        # Ensure missing_steps is a list
        if not isinstance(global_refl.get("missing_steps"), list):
            if global_refl.get("missing_steps"):
                # Try to convert to list if it's a string
                if isinstance(global_refl.get("missing_steps"), str):
                    global_refl["missing_steps"] = [global_refl["missing_steps"]]
                else:
                    global_refl["missing_steps"] = []
            else:
                global_refl["missing_steps"] = []
        
        return global_refl
    
    def _get_tool_evaluation(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tool evaluation from feedback.
        
        Args:
            feedback: Raw feedback dictionary
        
        Returns:
            Tool evaluation dictionary
        """
        tool_eval = feedback.get("tool_evaluation", {})
        
        # Ensure all required fields are present
        default_tool_eval = {
            "tools_used": [],
            "tool_success": True,
            "tool_lessons": ""
        }
        
        # If we didn't get a dictionary, create an empty one
        if not isinstance(tool_eval, dict):
            tool_eval = {}
        
        # Apply defaults for missing fields
        for key, value in default_tool_eval.items():
            if key not in tool_eval or tool_eval[key] is None:
                tool_eval[key] = value
        
        # Ensure tools_used is a list
        if not isinstance(tool_eval.get("tools_used"), list):
            if tool_eval.get("tools_used"):
                # Try to convert to list if it's a string
                if isinstance(tool_eval.get("tools_used"), str):
                    # Split by comma if it's a comma-separated string
                    if "," in tool_eval["tools_used"]:
                        tool_eval["tools_used"] = [t.strip() for t in tool_eval["tools_used"].split(",")]
                    else:
                        tool_eval["tools_used"] = [tool_eval["tools_used"]]
                else:
                    tool_eval["tools_used"] = []
            else:
                tool_eval["tools_used"] = []
        
        return tool_eval
    
    def get_action_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action reflection from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            Action reflection dictionary
        """
        return feedback.get("reflection.action", {})
    
    def get_trajectory_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the trajectory reflection from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            Trajectory reflection dictionary
        """
        return feedback.get("reflection.trajectory", {})
    
    def get_global_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the global reflection from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            Global reflection dictionary
        """
        return feedback.get("reflection.global", {})
    
    def is_action_correct(self, feedback: Dict[str, Any]) -> bool:
        """
        Check if the action is correct according to feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            True if the action is correct, False otherwise
        """
        return feedback.get("action_correct", False)
    
    def is_task_complete(self, feedback: Dict[str, Any]) -> bool:
        """
        Check if the task is complete according to feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            True if the task is complete, False otherwise
        """
        global_refl = self.get_global_reflection(feedback)
        status = global_refl.get("status", "incomplete").lower()
        next_instr = global_refl.get("next_instruction", "CONTINUE").upper()
        
        return status == "complete" or next_instr == "TERMINATE"
    
    def get_wrong_hint(self, feedback: Dict[str, Any]) -> str:
        """
        Get the hint for wrong actions from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            Hint for wrong actions
        """
        return feedback.get("hint_if_wrong", "")
    
    def get_tool_evaluation(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the tool evaluation from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            Tool evaluation dictionary
        """
        return feedback.get("tool_evaluation", {})
    
    def get_tools_used(self, feedback: Dict[str, Any]) -> List[str]:
        """
        Get the tools used from feedback.
        
        Args:
            feedback: Processed feedback dictionary
        
        Returns:
            List of tools used
        """
        tool_eval = self.get_tool_evaluation(feedback)
        return tool_eval.get("tools_used", [])
