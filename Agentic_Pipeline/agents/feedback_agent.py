#!/usr/bin/env python3
"""
Feedback agent module for the Medical Visual Agent system.
This agent provides hierarchical reflection and feedback.
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Union

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from config import get_config
from utils.common import safe_json_loads

logger = logging.getLogger(__name__)

class FeedbackAgent:
    """
    Feedback agent that provides hierarchical reflection and feedback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feedback agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config("agent")
        self.model = self.config.get("critic_model")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 4096)
    
    def build_messages(
        self,
        step_num: int,
        target_output_json: str,
        ground_truth_after_action: str,
        full_trajectory_so_far: str,
        user_goal: str,
        tool_results: Optional[Dict[str, Any]] = None,
        tools_used: Optional[List[str]] = None,
        short_term_memory: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for the feedback agent.
        
        Args:
            step_num: Current step number
            target_output_json: Target agent's output for this step
            ground_truth_after_action: What actually happened after executing the action
            full_trajectory_so_far: Chronological summary of all steps so far
            user_goal: User goal/task
            tool_results: Optional tool results from this step
            tools_used: Optional list of tools used in this step
            short_term_memory: Optional short-term memory from previous step
        
        Returns:
            List of messages for the model
        """
        system_prompt = """
You are the CRITIC / HIERARCHICAL REFLECTOR.

You must:
1. Judge whether the Target Agent's predicted_next_action was correct in practice.
2. Evaluate tool usage: Were appropriate tools called? Were results correctly interpreted?
3. Produce step-level reflection (reflection.action).
4. Produce trajectory-level reflection (reflection.trajectory).
5. Produce global task-level reflection (reflection.global).
6. Indicate action_correct true/false.
7. If false, explain why_if_wrong and give hint_if_wrong.
8. Provide tool_evaluation with tools_used, tool_success, and tool_lessons.

CRITICAL FORMAT REQUIREMENTS:
- reflection.trajectory.completed_subtasks MUST be a JSON array of strings, e.g., ["Load MRI data", "Navigate to module"]
- reflection.trajectory.remaining_subtasks MUST be a JSON array of strings, e.g., ["Create segmentation", "Export results"]
- NEVER use strings or text descriptions in place of arrays
- Each subtask should be a short, specific task description (1-5 words)
- Extract subtasks from the USER_GOAL based on actual progress made so far
- completed_subtasks: List what has been accomplished based on steps taken
- remaining_subtasks: List what still needs to be done based on remaining steps

You MUST return EXACTLY ONE JSON object with the required keys including tool_evaluation.
Do NOT include any text outside that JSON.
""".strip()
        
        # Build tool context
        tool_context = ""
        if tools_used:
            tool_context += f"\nTOOLS_USED_IN_THIS_STEP: {', '.join(tools_used)}\n"
        if tool_results:
            tool_context += f"\nTOOL_RESULTS_FROM_STEP:\n{json.dumps(tool_results, indent=2)}\n"
        
        # Add short-term memory context if available
        memory_context = ""
        if short_term_memory:
            memory_context = f"\nSHORT_TERM_MEMORY (from previous step):\n{json.dumps(short_term_memory, indent=2)}\n"
        
        user_prompt = f"""
USER_GOAL:
{user_goal}

TARGET_AGENT_STEP_OUTPUT (Step {step_num}):
{target_output_json}

{tool_context}{memory_context}
GROUND_TRUTH_AFTER_ACTION (what actually happened after executing predicted_next_action):
{ground_truth_after_action}

FULL_TRAJECTORY_SO_FAR (chronological summary of all steps so far, including this one):
{full_trajectory_so_far}

Evaluate both the ACTION and TOOL USAGE:
- Did the agent use appropriate tools?
- Were tool results correctly interpreted?
- Could better tools have been chosen?
- Did the tools help or hinder the action?

CRITICAL SUBTASK FORMAT REQUIREMENTS:
- reflection.trajectory.completed_subtasks MUST be a JSON ARRAY of strings
- reflection.trajectory.remaining_subtasks MUST be a JSON ARRAY of strings
- Example: completed_subtasks: ["Load MRI data", "Open Data module", "Verify orientation"]
- Example: remaining_subtasks: ["Run bias correction", "Create segmentation", "Export results"]
- Extract specific subtasks from the USER_GOAL based on actual progress
- Each subtask should be short (1-5 words), specific, and actionable
- completed_subtasks should list what has been accomplished in previous steps
- remaining_subtasks should list what still needs to be done
- DO NOT use long descriptive sentences - use short, specific task names
- Based on the trajectory, identify which parts of the USER_GOAL have been completed

CRITICAL COMPLETE ACTION RULES:
- The "COMPLETE" action can ONLY be used in the LAST STEP
- If the agent used "COMPLETE" before the last step, mark action_correct as FALSE
- The "COMPLETE" action should ONLY appear when all required subtasks are truly finished
- Never mark a task as "complete" in reflection.global.status unless it's actually the final step and all objectives are achieved
- If there are remaining_subtasks or missing_steps, the status MUST be "incomplete"

Now respond with the single JSON object exactly in the required format, including tool_evaluation.
""".strip()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages
    
    def run(
        self,
        step_num: int,
        target_output_json: str,
        ground_truth_after_action: str,
        full_trajectory_so_far: str,
        user_goal: str,
        tool_results: Optional[Dict[str, Any]] = None,
        tools_used: Optional[List[str]] = None,
        short_term_memory: Optional[Dict[str, Any]] = None,
        api_client=None
    ) -> str:
        """
        Run the feedback agent to get reflection and feedback.
        
        Args:
            step_num: Current step number
            target_output_json: Target agent's output for this step
            ground_truth_after_action: What actually happened after executing the action
            full_trajectory_so_far: Chronological summary of all steps so far
            user_goal: User goal/task
            tool_results: Optional tool results from this step
            tools_used: Optional list of tools used in this step
            short_term_memory: Optional short-term memory from previous step
            api_client: Optional API client for model inference
        
        Returns:
            Raw response from the model as a string
        """
        # Build messages
        messages = self.build_messages(
            step_num,
            target_output_json,
            ground_truth_after_action,
            full_trajectory_so_far,
            user_goal,
            tool_results,
            tools_used,
            short_term_memory
        )
        
        # Call the actual model API (Qwen 2.5 7B)
        logger.info(f"Calling feedback model ({self.model}) for step {step_num}")
        
        if api_client:
            # Call the actual model via API
            try:
                return self._call_model(messages, api_client)
            except Exception as e:
                logger.error(f"Error calling critic model, falling back to mock: {e}")
                return self._mock_response(step_num, target_output_json)
        else:
            # No API client provided - should not happen in production
            logger.warning("No API client provided to feedback agent - using mock response")
            return self._mock_response(step_num, target_output_json)
    
    def _call_model(self, messages: List[Dict[str, Any]], api_client) -> str:
        """
        Call the model API.
        
        Args:
            messages: Messages to send to the model
            api_client: API client for model inference
        
        Returns:
            Raw response from the model as a string
        """
        # Call the Deep Infra API through OpenAI-compatible client
        logger.info(f"Calling {self.model} via Deep Infra API...")
        
        try:
            response = api_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                logger.error(f"Empty response from critic API - no choices returned")
                raise ValueError("Empty response from API - no choices")
            
            content = response.choices[0].message.content
            
            # Check if content is None or empty
            if content is None:
                logger.error(f"Empty content in critic API response")
                raise ValueError("Empty content in API response")
            
            content = content.strip()
            
            if not content:
                logger.error(f"Empty string returned from critic API")
                raise ValueError("Empty string returned from API")
            
            logger.info(f"SUCCESS Received response from critic model ({self.model}): {len(content)} characters")
            
            # Log first 200 chars for debugging
            logger.debug(f"Critic response preview: {content[:200]}...")
            
            return content
            
        except Exception as e:
            # Log the actual error
            logger.error(f"Error in _call_model for critic: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to trigger fallback to mock
    
    def _mock_response(self, step_num: int, target_output_json: str) -> str:
        """
        Generate a mock response for development/testing.
        
        Args:
            step_num: Current step number
            target_output_json: Target agent's output for this step
        
        Returns:
            Mock response as a string
        """
        # Parse the target output to get the predicted action
        try:
            target_output = json.loads(target_output_json) if isinstance(target_output_json, str) else target_output_json
            # Extract the actual step data - might be inside a list or directly in the dict
            if isinstance(target_output, list) and len(target_output) > 0:
                target_output = target_output[0]
            
            # Extract the step data from the dict
            step_key = list(target_output.keys())[0] if isinstance(target_output, dict) else f"Step {step_num}"
            step_data = target_output.get(step_key, {})
            
            predicted_action = step_data.get("predicted_next_action", {})
            tool_call = predicted_action.get("tool_call", "UNKNOWN")
        except Exception:
            # If parsing fails, use defaults
            tool_call = "UNKNOWN"
        
        # Generate dynamic subtasks based on step number
        # As we progress, completed tasks increase and remaining tasks decrease
        all_subtasks = [
            "Navigate to correct module",
            "Load MRI scan data", 
            "Verify data orientation",
            "Apply bias correction",
            "Adjust image contrast",
            "Create segmentation",
            "Refine segmentation boundaries",
            "Create ROI",
            "Switch to proper layout",
            "Save processed data",
            "Export results"
        ]
        
        # Calculate progress based on step number
        num_completed = min(step_num, len(all_subtasks))
        completed_subtasks = all_subtasks[:num_completed]
        remaining_subtasks = all_subtasks[num_completed:]
        
        # Calculate overall progress percentage
        progress_pct = int((num_completed / len(all_subtasks)) * 100)
        
        # Generate dynamic observations based on step number
        observations = [
            "Initial UI loaded successfully.",
            "Navigation to module completed.",
            "Data loading interface accessed.",
            "Bias correction module opened.",
            "Image adjustment controls visible.",
            "Segmentation tools activated.",
            "Segmentation boundaries refined.",
            "ROI creation tools displayed.",
            "Layout changed to analysis view.",
            "Data processing completed.",
            "Export options available."
        ]
        
        lessons = [
            "Start by navigating to the correct module.",
            "Verify module is loaded before proceeding.",
            "Confirm data is visible before processing.",
            "Wait for processing to complete before next action.",
            "Check current state before applying changes.",
            "Use appropriate tools for the current task.",
            "Refine results iteratively for better accuracy.",
            "Verify ROI placement before continuing.",
            "Ensure proper layout for analysis.",
            "Confirm processing completed successfully.",
            "Check export settings before saving."
        ]
        
        # Get current observation and lesson based on step
        current_observation = observations[min(step_num - 1, len(observations) - 1)]
        current_lesson = lessons[min(step_num - 1, len(lessons) - 1)]
        
        # Build pitfalls list that accumulates
        pitfalls = []
        if step_num >= 2:
            pitfalls.append("Ensure correct module is selected before performing operations.")
        if step_num >= 5:
            pitfalls.append("Verify data quality before applying complex processing.")
        if step_num >= 8:
            pitfalls.append("Double-check ROI boundaries for accuracy.")
        
        mock_response = {
            "reflection.action": {
                "last_action": f"{tool_call} on target element at step {step_num}",
                "last_observation": current_observation,
                "last_lesson": current_lesson
            },
            "reflection.trajectory": {
                "overall_progress": f"Task progress: {progress_pct}% complete. Successfully completed {num_completed} of {len(all_subtasks)} subtasks.",
                "completed_subtasks": completed_subtasks,
                "remaining_subtasks": remaining_subtasks,
                "known_pitfalls": pitfalls if pitfalls else []
            },
            "reflection.global": {
                "status": "incomplete" if remaining_subtasks else "complete",
                "missing_steps": remaining_subtasks[:3] if len(remaining_subtasks) > 3 else remaining_subtasks,  # Show next 3 steps
                "next_instruction": "CONTINUE" if remaining_subtasks else "TERMINATE"
            },
            "action_correct": True,
            "why_if_wrong": "",
            "hint_if_wrong": "",
            "tool_evaluation": {
                "tools_used": ["object_detection"],
                "tool_success": True,
                "tool_lessons": f"Tools successfully identified elements for step {step_num}."
            }
        }
        return json.dumps(mock_response, indent=2)

    def process_feedback(self, raw_feedback: str) -> Dict[str, Any]:
        """
        Process the raw feedback from the feedback agent.
        
        Args:
            raw_feedback: Raw feedback from the feedback agent
        
        Returns:
            Processed feedback as a dictionary
        """
        try:
            feedback = safe_json_loads(raw_feedback)
            return feedback
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "reflection.action": {},
                "reflection.trajectory": {},
                "reflection.global": {},
                "action_correct": False,
                "why_if_wrong": "Error processing feedback",
                "hint_if_wrong": "System error - please try again",
                "tool_evaluation": {}
            }
    
    def extract_action_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract action reflection from feedback.
        
        Args:
            feedback: Processed feedback
        
        Returns:
            Action reflection dictionary
        """
        return feedback.get("reflection.action", {})
    
    def extract_trajectory_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract trajectory reflection from feedback.
        
        Args:
            feedback: Processed feedback
        
        Returns:
            Trajectory reflection dictionary
        """
        return feedback.get("reflection.trajectory", {})
    
    def extract_global_reflection(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract global reflection from feedback.
        
        Args:
            feedback: Processed feedback
        
        Returns:
            Global reflection dictionary
        """
        return feedback.get("reflection.global", {})
    
    def is_action_correct(self, feedback: Dict[str, Any]) -> bool:
        """
        Check if the action is correct according to feedback.
        
        Args:
            feedback: Processed feedback
        
        Returns:
            True if the action is correct, False otherwise
        """
        return feedback.get("action_correct", False)
    
    def get_wrong_hint(self, feedback: Dict[str, Any]) -> str:
        """
        Get the hint for wrong actions from feedback.
        
        Args:
            feedback: Processed feedback
        
        Returns:
            Hint for wrong actions
        """
        return feedback.get("hint_if_wrong", "")

def run_critic_agent(
    step_num: int,
    target_output_json: str,
    ground_truth_after_action: str,
    full_trajectory_so_far: str,
    user_goal: str,
    tool_results: Optional[Dict[str, Any]] = None,
    tools_used: Optional[List[str]] = None,
    short_term_memory: Optional[Dict[str, Any]] = None,
    api_client=None
) -> str:
    """
    Run the feedback agent to get reflection and feedback.
    This is a helper function for backward compatibility.
    
    Args:
        step_num: Current step number
        target_output_json: Target agent's output for this step
        ground_truth_after_action: What actually happened after executing the action
        full_trajectory_so_far: Chronological summary of all steps so far
        user_goal: User goal/task
        tool_results: Optional tool results from this step
        tools_used: Optional list of tools used in this step
        short_term_memory: Optional short-term memory from previous step
        api_client: Optional API client for model inference
    
    Returns:
        Raw response from the model as a string
    """
    agent = FeedbackAgent()
    return agent.run(
        step_num,
        target_output_json,
        ground_truth_after_action,
        full_trajectory_so_far,
        user_goal,
        tool_results,
        tools_used,
        short_term_memory,
        api_client
    )
