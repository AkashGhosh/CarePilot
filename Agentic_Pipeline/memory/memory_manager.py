#!/usr/bin/env python3
"""
Memory manager module for the Medical Visual Agent system.
Coordinates short-term and long-term memory.
"""

import logging
import json
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

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import get_config

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory manager for coordinating short-term and long-term memory.
    """
    
    def __init__(
        self, 
        initial_short_term: Optional[Dict[str, Any]] = None,
        initial_long_term: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            initial_short_term: Initial state for short-term memory
            initial_long_term: Initial state for long-term memory
            config: Configuration dictionary
        """
        self.config = config or get_config("memory")
        self.short_term = ShortTermMemory(initial_short_term)
        self.long_term = LongTermMemory(initial_long_term)
        self.history = []  # History of memory states
    
    def update_from_reflection(
        self, 
        action_refl: Dict[str, Any], 
        traj_refl: Dict[str, Any], 
        global_refl: Dict[str, Any],
        wrong_hint: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        tool_lesson: Optional[str] = None,
        decision_branch: Optional[str] = None,
        failure_type: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Update both short-term and long-term memory from reflections.
        
        Args:
            action_refl: Action reflection dictionary
            traj_refl: Trajectory reflection dictionary
            global_refl: Global reflection dictionary
            wrong_hint: Optional hint to override last_lesson if action was wrong
            tools_used: Optional list of tools used in the action
            tool_lesson: Optional lesson about tool usage
            decision_branch: Optional decision branch information
            failure_type: Optional failure type information
        
        Returns:
            Tuple of (short-term memory, long-term memory)
        """
        # Save the current state to history before updating
        self._save_to_history()
        
        # Update short-term memory
        stm = self.short_term.update_from_action_reflection(
            action_refl, 
            wrong_hint, 
            tools_used, 
            tool_lesson, 
            decision_branch, 
            failure_type
        )
        
        # Update long-term memory
        ltm = self.long_term.update_from_reflection(
            traj_refl, 
            global_refl, 
            decision_branch, 
            failure_type
        )
        
        return stm, ltm
    
    def get_memories(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get both short-term and long-term memories.
        
        Returns:
            Tuple of (short-term memory, long-term memory)
        """
        return self.short_term.get_memory(), self.long_term.get_memory()
    
    def _save_to_history(self):
        """Save current memory state to history."""
        self.history.append({
            "short_term": self.short_term.get_memory(),
            "long_term": self.long_term.get_memory()
        })
        
        # Limit history size if needed
        max_history = self.config.get("memory_history_size", 10)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
    
    def add_backtrack_point(self, step_num: int):
        """
        Add a backtrack point with current memory state.
        
        Args:
            step_num: Current step number
        """
        self.long_term.add_backtrack_point(step_num, self.short_term.get_memory())
    
    def get_backtrack_point(self, step_num: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a backtrack point by step number.
        
        Args:
            step_num: Step number to retrieve, or None for the most recent
        
        Returns:
            Backtrack point or None if not found
        """
        backtrack_points = self.long_term.get("backtrack_points", [])
        
        if not backtrack_points:
            return None
        
        if step_num is None:
            # Return the most recent backtrack point
            return backtrack_points[-1]
        
        # Find backtrack point by step number
        for point in reversed(backtrack_points):
            if point.get("step_num") == step_num:
                return point
        
        return None
    
    def restore_from_backtrack(self, step_num: Optional[int] = None) -> bool:
        """
        Restore memory from a backtrack point.
        
        Args:
            step_num: Step number to restore from, or None for the most recent
        
        Returns:
            True if successful, False otherwise
        """
        backtrack_point = self.get_backtrack_point(step_num)
        
        if not backtrack_point:
            return False
        
        # Restore short-term memory
        stm = backtrack_point.get("short_term_memory", {})
        self.short_term = ShortTermMemory(stm)
        
        # Restore long-term memory
        ltm = backtrack_point.get("long_term_memory", {})
        self.long_term = LongTermMemory(ltm)
        
        # Add restoration note to history
        self._save_to_history()
        self.history[-1]["restored_from"] = backtrack_point.get("step_num")
        
        return True
    
    def reset(self):
        """Reset both short-term and long-term memory."""
        self.short_term.reset()
        self.long_term.reset()
        self.history = []
    
    def save_to_file(self, filepath: str):
        """
        Save memories to a JSON file.
        
        Args:
            filepath: Path to save the memories
        """
        memory_state = {
            "short_term": self.short_term.get_memory(),
            "long_term": self.long_term.get_memory(),
            "history": self.history
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(memory_state, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MemoryManager':
        """
        Load memories from a JSON file.
        
        Args:
            filepath: Path to load the memories from
        
        Returns:
            MemoryManager instance
        """
        with open(filepath, 'r') as f:
            memory_state = json.load(f)
        
        manager = cls(
            initial_short_term=memory_state.get("short_term"),
            initial_long_term=memory_state.get("long_term")
        )
        
        manager.history = memory_state.get("history", [])
        
        return manager
