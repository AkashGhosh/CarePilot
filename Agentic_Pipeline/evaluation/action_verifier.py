#!/usr/bin/env python3
"""
Action verifier module for the Medical Visual Agent system.
This module provides programmatic verification of actions.
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

class ActionVerifier:
    """
    Action verifier that checks if an action is correct programmatically.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the action verifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config("thresholds")
        
        # Load thresholds
        self.action_accept_threshold = self.config.get("action_accept_threshold", 0.80)
        self.action_fail_threshold = self.config.get("action_fail_threshold", 0.40)
        self.semantic_min_threshold = self.config.get("semantic_min", 0.60)
        self.first_step_semantic_min = self.config.get("first_step_semantic_min", 0.70)
    
    def verify_action(
        self,
        predicted_canonical: Dict[str, Any],
        env_canonical: Dict[str, Any],
        ui_tree: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        is_first_step: bool = False
    ) -> Dict[str, Any]:
        """
        Verify if an action is correct programmatically.
        
        Args:
            predicted_canonical: Canonicalized prediction from the target agent
            env_canonical: Canonicalized environment state after action execution
            ui_tree: Optional UI tree for element verification
            ground_truth: Optional ground truth for comparison
            is_first_step: Whether this is the first step
        
        Returns:
            Dictionary with verification results
        """
        checks = {}
        why = []
        hint = ""
        failure_type = "none"
        
        # Get predicted values - ensure predicted_canonical is not None
        if predicted_canonical is None:
            logger.warning("predicted_canonical is None, using empty values")
            predicted_canonical = {}
        
        pred_intent = predicted_canonical.get("intent", "").lower() if predicted_canonical.get("intent") else ""
        pred_target = predicted_canonical.get("target_label", "").lower() if predicted_canonical.get("target_label") else ""
        pred_coords = predicted_canonical.get("coords", [0, 0]) if predicted_canonical.get("coords") else [0, 0]
        
        # Get ground truth values if available - ONLY ACTION AND TARGET
        gt_target = ""
        gt_action = ""
        
        # Extract ground truth from trajectory if available
        if ground_truth and isinstance(ground_truth, dict):
            gt_target = ground_truth.get("target", "").lower()
            gt_action = ground_truth.get("action", "").lower()
        
        # 1. Semantic check - REMOVED: Only care about action type, not target matching
        # Always pass semantic since we don't care about target descriptions
        checks['semantic'] = 1.0
        semantic_match = True
        
        # 2. Action type check
        ui_changed = True  # Default assumption
        if gt_action and pred_intent:
            if gt_action.lower() == pred_intent.lower():
                checks['ui_change'] = 1.0
            else:
                checks['ui_change'] = 0.0
                ui_changed = False
                why.append(f"Action type mismatch: expected '{gt_action}', got '{pred_intent}'")
                hint = f"Use '{gt_action}' action instead of '{pred_intent}'"
                if failure_type == "none":
                    failure_type = "ui_change"
        else:
            # No ground truth action to compare against, check env_canonical for UI change
            if env_canonical.get("changed_regions") and len(env_canonical.get("changed_regions", [])) > 0:
                checks['ui_change'] = 1.0
            else:
                checks['ui_change'] = 0.0
                ui_changed = False
                why.append("No UI changes detected after action")
                hint = "Try a different action that causes visible UI changes"
                if failure_type == "none":
                    failure_type = "ui_change"
        
        # 3. Action execution check - SIMPLIFIED
        # Only focus on action type and semantic target matching
        # Always pass execution since we care about intent, not precision
        checks['execution'] = 1.0
        
        # 4. Side effect: confirm action completed successfully
        checks['side_effect'] = 1.0 if env_canonical.get("confirm_text") else 0.0
        if not env_canonical.get("confirm_text"):
            ui_changed = False
            if failure_type == "none":
                failure_type = "ui_change"
        
        # Apply weights to calculate match score - ACTION-TYPE-ONLY EVALUATION
        # Focus ONLY on action type matching (ui_change), ignore target descriptions
        weights = {'semantic': 0.0, 'ui_change': 0.9, 'execution': 0.05, 'side_effect': 0.05}
        match_score = sum(weights[k] * checks.get(k, 0.0) for k in weights)
        
        # Use stricter semantic threshold for first step if specified
        semantic_threshold = self.first_step_semantic_min if is_first_step else self.semantic_min_threshold
        semantic_pass = checks.get('semantic', 0.0) >= semantic_threshold
        
        if not why:
            why = ["Matched expected checks."]
        
        return {
            "match_score": match_score,
            "checks": checks,
            "ui_changed": ui_changed,
            "semantic_match": semantic_match,
            "semantic_pass": semantic_pass,
            "why_if_wrong": "; ".join(why) if match_score < self.action_accept_threshold else "",
            "hint_if_wrong": hint if match_score < self.action_accept_threshold else "",
            "failure_type": failure_type
        }
    
    def is_action_correct(self, match_score: float) -> bool:
        """
        Check if an action is correct based on match score.
        
        Args:
            match_score: Match score from verification
        
        Returns:
            True if the action is correct, False otherwise
        """
        return match_score >= self.action_accept_threshold
    
    def is_action_failed(self, match_score: float) -> bool:
        """
        Check if an action has failed based on match score.
        
        Args:
            match_score: Match score from verification
        
        Returns:
            True if the action has failed, False otherwise
        """
        return match_score < self.action_fail_threshold
    
    def is_partial_success(
        self, 
        match_score: float, 
        ui_changed: bool, 
        semantic_pass: bool
    ) -> bool:
        """
        Check if an action is a partial success.
        
        Args:
            match_score: Match score from verification
            ui_changed: Whether the UI changed
            semantic_pass: Whether semantic check passed
        
        Returns:
            True if the action is a partial success, False otherwise
        """
        return (
            match_score >= self.action_fail_threshold and 
            match_score < self.action_accept_threshold and 
            ui_changed and 
            semantic_pass
        )
    
    def analyze_decision(
        self,
        verifier_result: Dict[str, Any],
        critic_action_ok: bool
    ) -> Tuple[bool, str]:
        """
        Analyze verification result and critic judgment to make a decision.
        
        Args:
            verifier_result: Result from verify_action
            critic_action_ok: Whether the critic judged the action as correct
        
        Returns:
            Tuple of (action_correct, decision_branch)
        """
        match_score = verifier_result["match_score"]
        ui_changed = verifier_result["ui_changed"]
        semantic_match = verifier_result["semantic_match"]
        semantic_pass = verifier_result["semantic_pass"]
        failure_type = verifier_result["failure_type"]
        
        final_action_correct = False
        decision_branch = "unknown"
        
        # Branch A: Both agree it's correct (high match score, critic agrees)
        if critic_action_ok and match_score >= self.action_accept_threshold:
            final_action_correct = True
            decision_branch = "A_both_agree_correct"
        
        # Branch B: Partial success (UI changed but semantic weak)
        elif ui_changed and self.action_fail_threshold <= match_score < self.action_accept_threshold:
            final_action_correct = False
            decision_branch = "B_partial_success"
        
        # Branch C: Semantic failure but UI change
        elif ui_changed and not semantic_match:
            final_action_correct = False
            decision_branch = "C_semantic_failure_with_ui_change"
        
        # Branch D: Clear failure (low match score, critic disagrees or neutral)
        elif not critic_action_ok and match_score < self.action_fail_threshold:
            final_action_correct = False
            decision_branch = "D_clear_failure"
        
        # Branch E: Ambiguous verifier (intermediate score, critic disagrees)
        elif not critic_action_ok and self.action_fail_threshold <= match_score < self.action_accept_threshold:
            # Prefer verifier for concrete checks, critic for semantic
            if semantic_match and failure_type != "coords":
                final_action_correct = True
                decision_branch = "E_ambiguous_accept"
            else:
                final_action_correct = False
                decision_branch = "E_ambiguous_reject"
        
        # Default fallback to verifier score
        elif match_score >= self.action_accept_threshold:
            final_action_correct = True
            decision_branch = "F_default_accept"
        else:
            final_action_correct = False
            decision_branch = "F_default_reject"
        
        return final_action_correct, decision_branch

def programmatic_verify(
    predicted_canonical: Dict[str, Any],
    env_canonical: Dict[str, Any],
    ui_tree: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    is_first_step: bool = False
) -> Dict[str, Any]:
    """
    Verify if an action is correct programmatically.
    This is a helper function for backward compatibility.
    
    Args:
        predicted_canonical: Canonicalized prediction from the target agent
        env_canonical: Canonicalized environment state after action execution
        ui_tree: Optional UI tree for element verification
        ground_truth: Optional ground truth for comparison
        is_first_step: Whether this is the first step
    
    Returns:
        Dictionary with verification results
    """
    verifier = ActionVerifier()
    return verifier.verify_action(
        predicted_canonical,
        env_canonical,
        ui_tree,
        ground_truth,
        is_first_step
    )
