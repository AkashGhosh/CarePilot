#!/usr/bin/env python3
"""
Short-term memory module for the Medical Visual Agent system.
Manages the immediate action results and failures.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """
    Short-term memory class for managing immediate action results and failures.
    
    Short-term memory contains what happened in the most recent attempt (ONE step back):
    - last_action we tried
    - last_observation after that action
    - last_lesson = immediate takeaway / fix
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        Initialize the short-term memory.
        
        Args:
            initial_state: Initial state for the short-term memory
        """
        # Default short-term memory structure
        self._memory = {
            "last_action": "NONE",
            "last_observation": "NONE",
            "last_lesson": "NONE"
        }
        
        # Update with initial state if provided
        if initial_state:
            self._memory.update(initial_state)
    
    def get_memory(self) -> Dict[str, Any]:
        """
        Get the current short-term memory.
        
        Returns:
            Short-term memory dictionary
        """
        return self._memory.copy()
    
    def update_from_action_reflection(
        self, 
        action_refl: Dict[str, Any], 
        wrong_hint: Optional[str] = None, 
        tools_used: Optional[List[str]] = None, 
        tool_lesson: Optional[str] = None,
        decision_branch: Optional[str] = None, 
        failure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update short-term memory from action reflection.
        
        Args:
            action_refl: Action reflection dictionary with last_action, last_observation, last_lesson
            wrong_hint: Optional hint to override last_lesson if action was wrong
            tools_used: Optional list of tools used in the action
            tool_lesson: Optional lesson about tool usage
            decision_branch: Optional decision branch information
            failure_type: Optional failure type information
        
        Returns:
            Updated short-term memory
        """
        # Update basic fields with fallbacks
        # If action_refl provides last_action, use it; otherwise keep existing or use empty string
        new_last_action = action_refl.get("last_action", "")
        if new_last_action and new_last_action.strip():
            self._memory["last_action"] = new_last_action
        elif not self._memory.get("last_action") or self._memory.get("last_action") == "" or self._memory.get("last_action") == "NONE":
            # Keep existing if it's valid, otherwise leave empty
            if self._memory.get("last_action") == "NONE":
                self._memory["last_action"] = ""
        
        # Always update observation and lesson if provided
        self._memory["last_observation"] = action_refl.get("last_observation", self._memory.get("last_observation", ""))
        self._memory["last_lesson"] = action_refl.get("last_lesson", self._memory.get("last_lesson", ""))
        
        # Add decision branch info for better context
        if decision_branch:
            self._memory["decision_branch"] = decision_branch
        
        # Add failure type if available
        if failure_type and failure_type != "none":
            self._memory["failure_type"] = failure_type
        
        # Prioritize wrong_hint over existing last_lesson if provided
        if wrong_hint and wrong_hint.strip():
            if not self._memory["last_lesson"] or self._memory["last_lesson"].strip() == "":
                self._memory["last_lesson"] = wrong_hint
            else:
                # Combine lessons if they're different
                if wrong_hint not in self._memory["last_lesson"]:
                    self._memory["last_lesson"] = f"{self._memory['last_lesson']}; {wrong_hint}"
        
        # Add tools used information
        if tools_used:
            self._memory["last_tools_used"] = tools_used
        
        # Add tool lesson with priority (handle both string and list types)
        if tool_lesson:
            # Convert to string if it's a list
            if isinstance(tool_lesson, list):
                tool_lesson_str = "; ".join(str(lesson) for lesson in tool_lesson if lesson)
            else:
                tool_lesson_str = str(tool_lesson)
            
            if tool_lesson_str and tool_lesson_str.strip():
                self._memory["tool_lesson"] = tool_lesson_str
                # Add tool lesson to last_lesson if not already included
                if tool_lesson_str not in self._memory["last_lesson"]:
                    if self._memory["last_lesson"] and self._memory["last_lesson"].strip():
                        self._memory["last_lesson"] = f"{self._memory['last_lesson']}; {tool_lesson_str}"
                    else:
                        self._memory["last_lesson"] = tool_lesson_str
        
        # Add offset coordinates if provided
        if "offset_coords" in locals() and locals()["offset_coords"]:
            self._memory["offset_coords"] = locals()["offset_coords"]
        
        return self._memory
    
    def reset(self):
        """Reset the short-term memory to its initial state."""
        self._memory = {
            "last_action": "NONE",
            "last_observation": "NONE",
            "last_lesson": "NONE"
        }
    
    def add_lesson(self, lesson: str):
        """
        Add a lesson to the short-term memory.
        
        Args:
            lesson: Lesson to add
        """
        if not lesson or not lesson.strip():
            return
        
        if not self._memory["last_lesson"] or self._memory["last_lesson"] == "NONE":
            self._memory["last_lesson"] = lesson
        elif lesson not in self._memory["last_lesson"]:
            self._memory["last_lesson"] = f"{self._memory['last_lesson']}; {lesson}"
    
    def get_last_action(self) -> str:
        """Get the last action."""
        return self._memory.get("last_action", "NONE")
    
    def get_last_observation(self) -> str:
        """Get the last observation."""
        return self._memory.get("last_observation", "NONE")
    
    def get_last_lesson(self) -> str:
        """Get the last lesson."""
        return self._memory.get("last_lesson", "NONE")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to memory."""
        return self._memory.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like setting of memory."""
        self._memory[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator for memory."""
        return key in self._memory
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get method."""
        return self._memory.get(key, default)
    
    def update(self, other_dict: Dict[str, Any]):
        """Update memory with another dictionary."""
        if not other_dict:
            return
        self._memory.update(other_dict)

def build_short_term_memory_from_action_reflection(
    action_refl: Dict[str, Any], 
    wrong_hint: Optional[str] = None, 
    tools_used: Optional[List[str]] = None, 
    tool_lesson: Optional[str] = None,
    decision_branch: Optional[str] = None, 
    failure_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build short-term memory from action reflection.
    This is a helper function for backward compatibility.
    
    Args:
        action_refl: Action reflection dictionary
        wrong_hint: Optional hint to override last_lesson if action was wrong
        tools_used: Optional list of tools used in the action
        tool_lesson: Optional lesson about tool usage
        decision_branch: Optional decision branch information
        failure_type: Optional failure type information
    
    Returns:
        Short-term memory dictionary
    """
    stm = ShortTermMemory()
    return stm.update_from_action_reflection(
        action_refl, 
        wrong_hint, 
        tools_used, 
        tool_lesson, 
        decision_branch, 
        failure_type
    )
