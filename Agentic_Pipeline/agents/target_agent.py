#!/usr/bin/env python3
"""
Target agent module for the Medical Visual Agent system.
This agent proposes actions based on current state, memory, and tools.
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

class TargetAgent:
    """
    Target agent that proposes actions based on current state, memory, and tools.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the target agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config("agent")
        self.model = self.config.get("target_model")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 4096)
    
    def build_messages(
        self,
        step_num: int,
        short_term_memory: Dict[str, Any],
        long_term_memory: Dict[str, Any],
        screen_image_url: str,
        available_tools: Dict[str, Any],
        user_goal: str,
        pre_grounding_results: Optional[Dict[str, Any]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build messages for the target agent.
        
        Args:
            step_num: Current step number
            short_term_memory: Short-term memory dictionary
            long_term_memory: Long-term memory dictionary
            screen_image_url: URL of the current screen image
            available_tools: Dictionary of available tools
            user_goal: User goal/task
            pre_grounding_results: Optional pre-grounding tool results
            tool_results: Optional tool results from previous calls
        
        Returns:
            List of messages for the model
        """
        # Ensure we always pass something in memory fields for step 1
        stm = short_term_memory if short_term_memory else {
            "last_action": "NONE",
            "last_observation": "NONE",
            "last_lesson": "NONE",
        }
        ltm = long_term_memory if long_term_memory else {
            "overall_progress": "",
            "completed_subtasks": [],
            "remaining_subtasks": [],
            "known_pitfalls": [],
        }
        
        system_prompt = f"""
You are the TARGET EXECUTION AGENT controlling a mobile/GUI environment step by step.

You can use VISUAL GROUNDING TOOLS to better understand the screen:
- object_detection: Find UI elements like buttons, icons, text fields
- visual_grounding: Find specific elements by text query (e.g., "Load Data button")
- depth_estimation: Understand UI hierarchy and layering
- edge_detection: Identify UI boundaries
- zoom_tool: Zoom into specific regions for detailed inspection

If you're uncertain about UI element locations or need precise coordinates, you MUST request tools in "reasoning.tool_calls".
Tools will be executed and results provided back to you.

You MUST respond with EXACTLY ONE valid JSON list, no extra text, no explanations, no markdown fences.

Format:
[
  "Step {step_num}" : {{
    "grounding": {{
      "current_screen_state": "...",
      "key_ui_elements": ["...","..."],
      "relevant_affordances": ["..."]
    }},
    "short_term_memory": {{
      "last_action": "...",
      "last_observation": "...",
      "last_lesson": "..."
    }},
    "long_term_memory": {{
      "overall_progress": "...",
      "completed_subtasks": ["..."],
      "remaining_subtasks": ["..."],
      "known_pitfalls": ["..."]
    }},
    "reasoning": {{
      "tool_calls": [
        {{"tool": "visual_grounding", "args": {{"query": "Load Data button", "image_id": 0}}}},
        {{"tool": "object_detection", "args": {{"objects": ["button", "icon"]}}}}
      ],
      "why_next_action_is_correct_and_safe": "...",
      "why_it_aligns_with_user_goal": "...",
      "why_alternatives_are_wrong_or_risky": "..."
    }},
    "tool_results": {{
      "visual_grounding": {{...}},
      "object_detection": {{...}}
    }},
    "image_info": {{
      "step_num": {step_num},
      "has_image": true,
      "image_data_uri": "{{image_data_uri}}"
    }},
    "predicted_next_action": {{
      "tool_call": "ONE_OF_AVAILABLE_TOOLS",
      "target": "UI element / selector / coords to operate on",
      "target_id": "element_id_from_ui_tree",  # MUST use an ID from ui_tree if available
      "arguments": {{
        "text_to_type": "...",
        "coords": [x, y],
        "extra": "..."
      }}
    }}
  }}
]

Constraints:
1. "grounding": ONLY describe what is visible RIGHT NOW on THE CURRENT SCREEN. Do not invent elements.
2. "short_term_memory": ONLY summarize what happened in the immediately previous attempt (the last step),
   including last_action, what we observed, and the immediate lesson. If step {step_num} is the first step, use the given values (like "NONE").
3. "long_term_memory": Summarize cumulative progress in this task so far:
   - what subgoals are already done,
   - what remains,
   - known pitfalls (e.g. loops or dead ends we discovered),
   - overall_progress so far.
   If this is the first step, keep them minimal/empty.
4. "reasoning":
   - "tool_calls" (REQUIRED): You MUST call visual_grounding or object_detection if you need to interact with UI elements.
   - Explain why the chosen next action is safe, aligned with USER_GOAL, and better than other visible actions.
5. "tool_results" (OPTIONAL): Will be populated by system after tool execution.
6. "predicted_next_action":
   - tool_call MUST be one of AVAILABLE_TOOLS.
   - You MUST provide all arguments that tool needs (coords, text, selector, etc).
   - If ui_tree is available, you MUST use target_id from ui_tree instead of free-text target.
   - If ui_tree is not available, you MUST request visual_grounding in tool_calls.
7. DO NOT add any keys not listed.
8. DO NOT output anything except the JSON list described above.
9. NEVER guess coordinates - if you need coords, you MUST use visual_grounding tool first.
10. CRITICAL COMPLETE ACTION RULES:
   - The "COMPLETE" action can ONLY be used in the LAST STEP (step {max_steps if max_steps else 'unknown'})
   - If this is NOT the last step (step {step_num} < {max_steps if max_steps else 'unknown'}), you MUST NEVER use "COMPLETE"
   - If this IS the last step (step {step_num} of {max_steps if max_steps else 'unknown'}), you MUST use "tool_call": "COMPLETE"
   - NEVER use COMPLETE before completing all required actions - it should be the absolute final action
   - Using COMPLETE prematurely will cause the task to fail
""".strip()
        
        # Add explicit last step instruction if applicable
        is_last_step = max_steps is not None and step_num >= max_steps
        if is_last_step:
            system_prompt += f"\n\n⚠️ CRITICAL: This is STEP {step_num} of {max_steps} - the FINAL AND LAST STEP.\n"
            system_prompt += f"You MUST use \"tool_call\": \"COMPLETE\" to indicate task completion.\n"
            system_prompt += f"Do NOT use any other action type. This is your only chance to complete the task."
        else:
            # Add warning if NOT last step - remind not to use COMPLETE
            remaining_steps = max_steps - step_num if max_steps else 0
            system_prompt += f"\n\n⚠️ IMPORTANT: This is STEP {step_num} of {max_steps} - NOT the last step.\n"
            system_prompt += f"There are {remaining_steps} more steps remaining.\n"
            system_prompt += f"You MUST NOT use \"COMPLETE\" - use a regular action like CLICK, SEGMENT, etc.\n"
            system_prompt += f"COMPLETE can ONLY be used in the final step ({max_steps if max_steps else 'unknown'})."
        
        # Build tool context text
        tool_context = ""
        if pre_grounding_results:
            tool_context += f"\nPRE_GROUNDING_TOOL_RESULTS (tools executed proactively):\n{json.dumps(pre_grounding_results, indent=2)}\n"
        if tool_results:
            tool_context += f"\nTOOL_RESULTS (from previous tool calls):\n{json.dumps(tool_results, indent=2)}\n"
        
        # Add tool effectiveness hints from memory with more structured guidance
        tool_effectiveness_hints = ""
        if isinstance(ltm, dict) and "tool_effectiveness" in ltm:
            tool_eff = ltm.get('tool_effectiveness', {})
            
            # Create a more structured and helpful summary of tool effectiveness
            tool_guidance = {}
            for tool_name, stats in tool_eff.items():
                success_rate = stats.get("success_rate", 0) * 100
                recommended = stats.get("recommended_usage", "")
                recent_lessons = stats.get("lessons", [])[-2:] if stats.get("lessons") else []
                
                tool_guidance[tool_name] = {
                    "success_rate": f"{success_rate:.1f}%",
                    "recommended_usage": recommended,
                }
                
                if recent_lessons:
                    tool_guidance[tool_name]["recent_lessons"] = recent_lessons
            
            # Only add if we have meaningful data
            if tool_guidance:
                tool_effectiveness_hints = f"\nTOOL EFFECTIVENESS INSIGHTS:\n{json.dumps(tool_guidance, indent=2)}\n"

        # Convert the available_tools dict to a formatted string for the prompt
        formatted_tools = ""
        for tool_name, tool_info in available_tools.items():
            required_args = tool_info.get('required_args', [])
            description = tool_info.get('description', 'No description available')
            
            required_args_str = ", ".join(required_args) if required_args else "None"
            formatted_tools += f"\n- {tool_name}: {description}\n  Required arguments: {required_args_str}"
        
        # Build the user message with both text and image
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""USER_GOAL: {user_goal}

AVAILABLE_TOOLS: {formatted_tools}

SHORT_TERM_MEMORY (what happened in the previous step):
{json.dumps(stm, indent=2)}

LONG_TERM_MEMORY (cumulative knowledge and progress):
{json.dumps(ltm, indent=2)}
{tool_context}
{tool_effectiveness_hints}

Step {step_num}: Based on the USER_GOAL, AVAILABLE_TOOLS, SHORT_TERM_MEMORY, LONG_TERM_MEMORY, and the current screen, determine the next action."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": screen_image_url}
                }
            ]
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            user_message
        ]
        
        return messages
    
    def run(
        self,
        step_num: int,
        short_term_memory: Dict[str, Any],
        long_term_memory: Dict[str, Any],
        screen_image_url: str,
        available_tools: Dict[str, Any],
        user_goal: str,
        pre_grounding_results: Optional[Dict[str, Any]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        api_client=None,
        max_steps: Optional[int] = None
    ) -> str:
        """
        Run the target agent to get the next action.
        
        Args:
            step_num: Current step number
            short_term_memory: Short-term memory dictionary
            long_term_memory: Long-term memory dictionary
            screen_image_url: URL of the current screen image
            available_tools: Dictionary of available tools
            user_goal: User goal/task
            pre_grounding_results: Optional pre-grounding tool results
            tool_results: Optional tool results from previous calls
            api_client: Optional API client for model inference
            max_steps: Maximum number of steps (for determining last step)
        
        Returns:
            Raw response from the model as a string
        """
        # Build messages
        messages = self.build_messages(
            step_num,
            short_term_memory,
            long_term_memory,
            screen_image_url,
            available_tools,
            user_goal,
            pre_grounding_results,
            tool_results,
            max_steps
        )
        
        # Call the actual model API (Qwen 2.5 7B)
        logger.info(f"Calling target model ({self.model}) for step {step_num}")
        
        if api_client:
            # Call the actual model via API
            try:
                return self._call_model(messages, api_client)
            except Exception as e:
                logger.error(f"Error calling target model, falling back to mock: {e}")
                return self._mock_response(step_num, short_term_memory, long_term_memory, user_goal)
        else:
            # No API client provided - should not happen in production
            logger.warning("No API client provided to target agent - using mock response")
            return self._mock_response(step_num, short_term_memory, long_term_memory, user_goal)
    
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
        
        response = api_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        content = response.choices[0].message.content
        logger.info(f"SUCCESS Received response from target model ({self.model}): {len(content)} characters")
        
        # Log first 200 chars for debugging
        logger.debug(f"Target response preview: {content[:200]}...")
        
        return content
    
    def _mock_response(
        self, 
        step_num: int, 
        short_term_memory: Dict[str, Any], 
        long_term_memory: Dict[str, Any], 
        user_goal: str
    ) -> str:
        """
        Generate a mock response for development/testing.
        
        Args:
            step_num: Current step number
            short_term_memory: Short-term memory dictionary
            long_term_memory: Long-term memory dictionary
            user_goal: User goal/task
        
        Returns:
            Mock response as a string
        """
        mock_response = {
            f"Step {step_num}": {
                "grounding": {
                    "current_screen_state": "The screen shows a medical imaging application with a sidebar menu, main viewport, and tool controls.",
                    "key_ui_elements": ["Data module button", "Load Data button", "Viewport"],
                    "relevant_affordances": ["Click on Data module", "Load patient data", "Adjust view"]
                },
                "short_term_memory": short_term_memory or {
                    "last_action": "NONE",
                    "last_observation": "NONE",
                    "last_lesson": "NONE"
                },
                "long_term_memory": long_term_memory or {
                    "overall_progress": "Just starting the task",
                    "completed_subtasks": [],
                    "remaining_subtasks": ["Load MRI data", "Create segmentation", "Measure volume"],
                    "known_pitfalls": []
                },
                "reasoning": {
                    "tool_calls": [
                        {"tool": "object_detection", "args": {"objects": ["button", "menu"], "image_id": 0}}
                    ],
                    "why_next_action_is_correct_and_safe": "We need to first navigate to the Data module to load the MRI scan",
                    "why_it_aligns_with_user_goal": "Loading the MRI scan is the first step in the task",
                    "why_alternatives_are_wrong_or_risky": "Attempting segmentation before loading data would fail"
                },
                "predicted_next_action": {
                    "tool_call": "CLICK",
                    "target": "Data module button",
                    "arguments": {
                        "coords": [50, 100]
                    }
                }
            }
        }
        return json.dumps([mock_response], indent=2)

def run_target_agent(
    step_num: int,
    short_term_memory: Dict[str, Any],
    long_term_memory: Dict[str, Any],
    screen_image_url: str,
    available_tools: Dict[str, Any],
    user_goal: str,
    pre_grounding_results: Optional[Dict[str, Any]] = None,
    tool_results: Optional[Dict[str, Any]] = None,
    api_client=None,
    max_steps: Optional[int] = None
) -> str:
    """
    Run the target agent to get the next action.
    This is a helper function for backward compatibility.
    
    Args:
        step_num: Current step number
        short_term_memory: Short-term memory dictionary
        long_term_memory: Long-term memory dictionary
        screen_image_url: URL of the current screen image
        available_tools: Dictionary of available tools
        user_goal: User goal/task
        pre_grounding_results: Optional pre-grounding tool results
        tool_results: Optional tool results from previous calls
        api_client: Optional API client for model inference
    
    Returns:
        Raw response from the model as a string
    """
    agent = TargetAgent()
    return agent.run(
        step_num,
        short_term_memory,
        long_term_memory,
        screen_image_url,
        available_tools,
        user_goal,
        pre_grounding_results,
        tool_results,
        api_client,
        max_steps
    )

def process_target_output(raw_output: str) -> Dict[str, Any]:
    """
    Process the raw output from the target agent.
    
    Args:
        raw_output: Raw output from the target agent
    
    Returns:
        Processed output as a dictionary
    """
    try:
        output = safe_json_loads(raw_output)
        
        # If it's a list with a single item that is a dictionary
        if isinstance(output, list) and len(output) == 1 and isinstance(output[0], dict):
            # Get the first (and only) dictionary
            return output[0]
        
        # If it's already a dictionary
        elif isinstance(output, dict):
            return output
        
        # If it's a list with multiple items
        elif isinstance(output, list) and len(output) > 1:
            # Take the first item
            return output[0] if isinstance(output[0], dict) else {}
        
        else:
            logger.error(f"Unexpected output format: {output}")
            return {}
    
    except Exception as e:
        logger.error(f"Error processing target output: {e}")
        return {}
