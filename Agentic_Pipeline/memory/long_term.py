#!/usr/bin/env python3
"""
Long-term memory module for the Medical Visual Agent system.
Manages trajectory learning and global reflection.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    Long-term memory class for managing trajectory learning and global reflection.
    
    Long-term memory contains the run-level state:
    - overall_progress toward the user goal
    - completed_subtasks
    - remaining_subtasks
    - known_pitfalls / loops to avoid
    - tool effectiveness
    - recurring issues
    - decision history
    - backtrack points
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        Initialize the long-term memory.
        
        Args:
            initial_state: Initial state for the long-term memory
        """
        # Default long-term memory structure
        self._memory = {
            "overall_progress": "",
            "completed_subtasks": [],
            "remaining_subtasks": [],
            "known_pitfalls": [],
            "tool_effectiveness": {},
            "recurring_issues": [],
            "decision_history": [],
            "backtrack_points": [],
            "failure_types": {},
        }
        
        # Update with initial state if provided
        if initial_state:
            self.update(initial_state)
    
    def get_memory(self) -> Dict[str, Any]:
        """
        Get the current long-term memory.
        
        Returns:
            Long-term memory dictionary
        """
        return self._memory.copy()
    
    def update_from_reflection(
        self, 
        traj_refl: Dict[str, Any], 
        global_refl: Dict[str, Any],
        decision_branch: Optional[str] = None,
        failure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update long-term memory from trajectory and global reflection.
        
        Args:
            traj_refl: Trajectory reflection dictionary
            global_refl: Global reflection dictionary
            decision_branch: Optional decision branch information
            failure_type: Optional failure type information
        
        Returns:
            Updated long-term memory
        """
        # Ensure reflection dictionaries are valid
        traj_refl = traj_refl or {}
        global_refl = global_refl or {}
        
        # Update overall progress - always update if provided, even if empty (to clear old values)
        if "overall_progress" in traj_refl:
            new_progress = traj_refl["overall_progress"]
            # Only update if it's a non-empty string
            if new_progress and isinstance(new_progress, str) and new_progress.strip():
                self._memory["overall_progress"] = new_progress
            # If it's provided but empty, and we have existing progress, keep existing
            elif not self._memory.get("overall_progress"):
                # Only set to empty if we don't have existing progress
                self._memory["overall_progress"] = ""
        
        # Update completed subtasks
        if "completed_subtasks" in traj_refl:
            # Merge with existing completed subtasks and deduplicate
            new_completed = traj_refl["completed_subtasks"]
            
            # Ensure new_completed is a list - handle string case
            if isinstance(new_completed, str):
                logger.warning(f"completed_subtasks is a string, converting to list: {new_completed[:100]}...")
                # Save the original string for parsing
                desc = new_completed
                # Try to parse as JSON first
                try:
                    import json
                    parsed = json.loads(desc)
                    if isinstance(parsed, list):
                        new_completed = parsed
                    else:
                        # If parsed but not a list, try to extract
                        new_completed = []
                except:
                    # If not JSON, try to extract subtasks from the description
                    # Look for common patterns like "Load MRI", "Create segmentation", etc.
                    new_completed = []
                    # Try to extract action phrases (verb + noun patterns)
                    import re
                    # Common patterns: "Load X", "Create X", "Run X", "Export X", "Switch to X"
                    patterns = re.findall(r'\b(?:Load|Import|Open|Create|Run|Apply|Invert|Switch|Save|Export|Verify|Capture)\s+[^.,;]+', desc)
                    new_completed = [p.strip() for p in patterns if len(p.strip()) < 50]
                    # If no patterns found, try simple split
                    if not new_completed:
                        new_completed = [s.strip() for s in desc.replace('\n', ',').split(',') if 5 < len(s.strip()) < 50]
            
            if isinstance(new_completed, list):
                # Filter out any non-string items
                new_completed = [str(item).strip() for item in new_completed if item and isinstance(item, (str, int, float))]
                # Remove very long items (likely descriptions, not subtasks)
                # Subtasks should be 1-5 words (max ~50 chars), filter out long descriptions
                new_completed = [item for item in new_completed if len(item) < 50 and len(item.split()) <= 6]
                # Only merge if new_completed is not empty, otherwise keep existing
                if new_completed:
                    # Merge with existing and deduplicate
                    self._memory["completed_subtasks"] = list(dict.fromkeys(
                        self._memory["completed_subtasks"] + new_completed
                    ))
            else:
                logger.warning(f"completed_subtasks is not a list or string (got {type(new_completed)}), using empty list")
                if not self._memory["completed_subtasks"]:
                    self._memory["completed_subtasks"] = []
        
        # Update remaining subtasks
        if "remaining_subtasks" in traj_refl:
            new_remaining = traj_refl["remaining_subtasks"]
            
            # Ensure new_remaining is a list - handle string case
            if isinstance(new_remaining, str):
                logger.warning(f"remaining_subtasks is a string, converting to list: {new_remaining[:100]}...")
                # Save the original string for parsing
                desc = new_remaining
                # Try to parse as JSON first
                try:
                    import json
                    parsed = json.loads(desc)
                    if isinstance(parsed, list):
                        new_remaining = parsed
                    else:
                        # If parsed but not a list, try to extract
                        new_remaining = []
                except:
                    # If not JSON, try to extract subtasks from the description
                    # Look for common patterns like "Load MRI", "Create segmentation", etc.
                    new_remaining = []
                    # Try to extract action phrases (verb + noun patterns)
                    import re
                    # Common patterns: "Load X", "Create X", "Run X", "Export X", "Switch to X"
                    patterns = re.findall(r'\b(?:Load|Import|Open|Create|Run|Apply|Invert|Switch|Save|Export|Verify|Capture|Run bias|Invert grayscale|Create segmentation|Create ROI|Switch to|Save processed|Export intensity)\s+[^.,;]+', desc)
                    new_remaining = [p.strip() for p in patterns if len(p.strip()) < 50]
                    # If still empty, try splitting by commas/keywords
                    if not new_remaining:
                        # Split by common separators and keywords
                        parts = re.split(r'[,;]\s*|and\s+|including\s+', desc)
                        new_remaining = [p.strip() for p in parts if 5 < len(p.strip()) < 50]
            
            if isinstance(new_remaining, list):
                # Filter out any non-string items and ensure they're reasonable subtasks
                new_remaining = [str(item).strip() for item in new_remaining if item and isinstance(item, (str, int, float))]
                # Remove very long items (likely descriptions, not subtasks)
                # Subtasks should be 1-5 words (max ~50 chars), filter out long descriptions
                new_remaining = [item for item in new_remaining if len(item) < 50 and len(item.split()) <= 6]
                self._memory["remaining_subtasks"] = new_remaining.copy()
            else:
                logger.warning(f"remaining_subtasks is not a list or string (got {type(new_remaining)}), using empty list")
                self._memory["remaining_subtasks"] = []
            
            # Remove any subtasks that are now completed
            completed = self._memory.get("completed_subtasks", [])
            if completed:
                # Convert completed to strings for comparison
                completed_str = [str(c) for c in completed]
                self._memory["remaining_subtasks"] = [
                    subtask for subtask in self._memory["remaining_subtasks"]
                    if str(subtask) not in completed_str
                ]
            
            # Fold in global "missing_steps" if incomplete
            if global_refl.get("status") == "incomplete":
                missing = global_refl.get("missing_steps", [])
                if missing:
                    # Ensure missing is a list
                    if isinstance(missing, str):
                        missing = [s.strip() for s in missing.split(',') if s.strip()]
                    if isinstance(missing, list):
                        self._memory["remaining_subtasks"] = list(
                            dict.fromkeys(self._memory["remaining_subtasks"] + [str(m) for m in missing if m])
                        )
        
        # Update known pitfalls
        if "known_pitfalls" in traj_refl:
            # Merge with existing pitfalls and deduplicate
            self._memory["known_pitfalls"] = list(dict.fromkeys(
                self._memory["known_pitfalls"] + traj_refl["known_pitfalls"]
            ))
        
        # Track recurring issues - patterns in pitfalls that appear multiple times
        pitfall_count = {}
        for pitfall in self._memory["known_pitfalls"]:
            pitfall_count[pitfall] = pitfall_count.get(pitfall, 0) + 1
            
        # If a pitfall occurs more than once, add it to recurring issues
        recurring = [
            p for p, count in pitfall_count.items() 
            if count > 1 and p not in self._memory["recurring_issues"]
        ]
        
        if recurring:
            self._memory["recurring_issues"].extend(recurring)
            # Keep the list unique
            self._memory["recurring_issues"] = list(dict.fromkeys(self._memory["recurring_issues"]))
        
        # Track decision branches for debugging/analysis
        if decision_branch:
            self._memory["decision_history"].append(decision_branch)
            
            # Track multiple occurrences of the same failure type
            if failure_type and failure_type != "none":
                self._memory["failure_types"][failure_type] = self._memory["failure_types"].get(failure_type, 0) + 1
                
                # If a single failure type happens more than 2 times, add it to recurring issues
                if self._memory["failure_types"][failure_type] > 2:
                    issue = f"Recurring failure: {failure_type}"
                    if issue not in self._memory["recurring_issues"]:
                        self._memory["recurring_issues"].append(issue)
        
        return self._memory
    
    def add_backtrack_point(
        self, 
        step_num: int, 
        short_term_memory: Dict[str, Any]
    ):
        """
        Add a backtrack point to the long-term memory.
        
        Args:
            step_num: Current step number
            short_term_memory: Current short-term memory
        """
        self._memory["backtrack_points"].append({
            "step_num": step_num,
            "short_term_memory": short_term_memory.copy() if short_term_memory else {},
            "long_term_memory": self._memory.copy()
        })
    
    def update_tool_effectiveness(
        self, 
        tool_name: str, 
        success: bool,
        confidence: Optional[float] = None
    ):
        """
        Update tool effectiveness statistics.
        
        Args:
            tool_name: Name of the tool
            success: Whether the tool was successful
            confidence: Optional confidence of the tool result
        """
        if "tool_effectiveness" not in self._memory:
            self._memory["tool_effectiveness"] = {}
        
        if tool_name not in self._memory["tool_effectiveness"]:
            self._memory["tool_effectiveness"][tool_name] = {
                "used": 0,
                "successful": 0,
                "confidence_sum": 0,
                "avg_confidence": 0
            }
        
        tool_stats = self._memory["tool_effectiveness"][tool_name]
        tool_stats["used"] += 1
        
        if success:
            tool_stats["successful"] += 1
        
        if confidence is not None:
            tool_stats["confidence_sum"] += confidence
            tool_stats["avg_confidence"] = tool_stats["confidence_sum"] / tool_stats["used"]
        
        # Calculate success rate
        tool_stats["success_rate"] = tool_stats["successful"] / tool_stats["used"]
    
    def reset(self):
        """Reset the long-term memory to its initial state."""
        self._memory = {
            "overall_progress": "",
            "completed_subtasks": [],
            "remaining_subtasks": [],
            "known_pitfalls": [],
            "tool_effectiveness": {},
            "recurring_issues": [],
            "decision_history": [],
            "backtrack_points": [],
            "failure_types": {},
        }
    
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
        
        # Special handling for list fields to avoid duplicates
        list_fields = ["completed_subtasks", "remaining_subtasks", "known_pitfalls", 
                      "recurring_issues", "decision_history"]
        
        for key, value in other_dict.items():
            if key in list_fields and key in self._memory and isinstance(value, list):
                # Merge lists and remove duplicates
                self._memory[key] = list(dict.fromkeys(self._memory[key] + value))
            else:
                self._memory[key] = value

def merge_long_term_memory(
    prev_long_term_memory: Optional[Dict[str, Any]], 
    traj_refl: Dict[str, Any], 
    global_refl: Dict[str, Any],
    decision_branch: Optional[str] = None,
    failure_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge long-term memory with trajectory and global reflection.
    This is a helper function for backward compatibility.
    
    Args:
        prev_long_term_memory: Previous long-term memory
        traj_refl: Trajectory reflection dictionary
        global_refl: Global reflection dictionary
        decision_branch: Optional decision branch information
        failure_type: Optional failure type information
    
    Returns:
        Updated long-term memory dictionary
    """
    ltm = LongTermMemory(prev_long_term_memory)
    return ltm.update_from_reflection(traj_refl, global_refl, decision_branch, failure_type)
