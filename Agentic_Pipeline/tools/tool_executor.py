#!/usr/bin/env python3
"""
Tool executor module for the Medical Visual Agent system.
This module handles execution of visual tools.
"""

import json
import time
import logging
import requests
import os
import sys
from typing import Dict, Any, Optional, List
from PIL import Image
import base64
from io import BytesIO

# Setup paths for imports
try:
    import _setup_paths  # noqa: E402
except ImportError:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from config import get_config

logger = logging.getLogger(__name__)

# Default timeouts
TOOL_TIMEOUT = 5.0
TOOL_MAX_RETRIES = 2

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image to encode
    
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded

class ToolExecutor:
    """
    Executes visual grounding tools with fallback to mock implementations.
    """
    
    def __init__(self, use_mock: bool = False, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tool executor.
        
        Args:
            use_mock: If True, always use mock implementations (for testing)
            config: Configuration dictionary
        """
        self.config = config or get_config("tool")
        self.use_mock = use_mock
        self.timeout = self.config.get("timeout", TOOL_TIMEOUT)
        self.max_retries = self.config.get("max_retries", TOOL_MAX_RETRIES)
        self.session = requests.Session()
        
        # Tool endpoints from config
        self.tool_endpoints = self.config.get("tool_endpoints", {
            "object_detection": "http://localhost:8000/object_detection",
            "depth_estimation": "http://localhost:8003/depth_estimation", 
            "edge_detection": "http://localhost:8002/edge_detection",
            "zoom_tool": "http://localhost:8001/zoom",
            "visual_grounding": "http://localhost:8004/visual_grounding"
        })
    
    def _make_request(self, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to tool endpoint.
        
        Args:
            tool_name: Name of the tool
            data: Request payload
            
        Returns:
            Tool result or error dict
        """
        if tool_name not in self.tool_endpoints:
            logger.error(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
        
        endpoint = self.tool_endpoints[tool_name]
        logger.info(f"Making request to {endpoint}")
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    endpoint,
                    json=data,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    logger.info(f"Tool {tool_name} request successful")
                    return response.json()
                else:
                    logger.warning(f"Tool {tool_name} returned status {response.status_code}")
                    if attempt == self.max_retries - 1:
                        return {
                            "error": f"Tool {tool_name} returned status {response.status_code}",
                            "details": response.text[:200]
                        }
                    time.sleep(0.5)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout calling {tool_name}, attempt {attempt+1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Timeout calling {tool_name}"}
                time.sleep(0.5)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error calling {tool_name}, falling back to mock")
                # Endpoint not available, fall back to mock
                return self._mock_tool(tool_name, data)
            except Exception as e:
                logger.warning(f"Error calling {tool_name}: {str(e)}, attempt {attempt+1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Error calling {tool_name}: {str(e)}"}
                time.sleep(0.5)
        
        return {"error": f"Failed to call {tool_name} after {self.max_retries} attempts"}
    
    def _mock_tool(self, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock tool implementations for when endpoints are unavailable.
        
        Args:
            tool_name: Name of the tool
            data: Request payload
            
        Returns:
            Mock tool result
        """
        logger.info(f"Using mock implementation for {tool_name}")
        
        if tool_name == "object_detection":
            objects = data.get("objects", [])
            return {
                "detections": [
                    {
                        "label": obj,
                        "confidence": 0.75,
                        "bbox": [100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50]
                    }
                    for i, obj in enumerate(objects[:3])
                ]
            }
        
        elif tool_name == "depth_estimation":
            return {
                "depth_map": "mocked_depth_map",
                "message": "Depth estimation completed (mock)"
            }
        
        elif tool_name == "edge_detection":
            return {
                "edges": "mocked_edge_detection",
                "message": "Edge detection completed (mock)"
            }
        
        elif tool_name == "zoom_tool":
            image_id = data.get("image_id", 0)
            bbox = data.get("bbox", [100, 100, 200, 200])
            factor = data.get("factor", 1.5)
            return {
                "zoomed_image_id": image_id,
                "bbox": bbox,
                "factor": factor,
                "message": f"Zoomed image {image_id} on {bbox} with {factor}x magnification (mock)"
            }
        
        elif tool_name == "visual_grounding":
            query = data.get("query", "")
            image_id = data.get("image_id", 0)
            # Mock bbox based on query keywords
            if "button" in query.lower() or "click" in query.lower():
                bbox = [1054, 0, 1089, 35]  # Common button location
            elif "icon" in query.lower():
                bbox = [50, 50, 100, 100]
            else:
                bbox = [200, 200, 300, 300]  # Default location
            
            return {
                "image_id": image_id,
                "query": query,
                "bbox": bbox,
                "confidence": 0.80,
                "message": f"Found '{query}' at {bbox} (mock)"
            }
        
        return {"error": f"No mock implementation for {tool_name}"}
    
    def execute_tool(
        self, 
        tool_name: str, 
        image: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a visual tool.
        
        Args:
            tool_name: Name of the tool to execute
            image: PIL Image (if tool needs image input)
            **kwargs: Additional tool-specific arguments
            
        Returns:
            Tool result dict
        """
        if self.use_mock:
            return self._mock_tool(tool_name, kwargs)
        
        # Prepare request data
        data = kwargs.copy()
        
        # Add image if provided
        if image is not None:
            data["image"] = encode_image_to_base64(image)
            data["image_format"] = "base64_png"
        
        # Make request (with fallback to mock on connection error)
        result = self._make_request(tool_name, data)
        
        # If request failed, fall back to mock
        if "error" in result:
            logger.warning(f"Tool {tool_name} request failed, falling back to mock")
            return self._mock_tool(tool_name, data)
        
        return result
    
    def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool call from Target agent's reasoning.
        
        Args:
            tool_call: Dict with 'tool' and 'args' keys
            image: PIL Image for tools that need it
            
        Returns:
            Tool result dict
        """
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        if not tool_name:
            logger.error("Tool name missing in tool_call")
            return {"error": "Tool name missing in tool_call"}
        
        logger.info(f"Executing tool call: {tool_name}")
        return self.execute_tool(tool_name, image=image, **args)
    
    def execute_multiple_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        image: Optional[Image.Image] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple tool calls and return results keyed by tool name.
        
        Args:
            tool_calls: List of tool call dicts
            image: PIL Image for tools that need it
            
        Returns:
            Dict mapping tool names to their results
        """
        results = {}
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool", "unknown")
            result = self.execute_tool_call(tool_call, image=image)
            results[tool_name] = result
        
        return results
    
    def register_tool_endpoint(self, tool_name: str, endpoint: str):
        """
        Register or update a tool endpoint.
        
        Args:
            tool_name: Name of the tool
            endpoint: Tool endpoint URL
        """
        self.tool_endpoints[tool_name] = endpoint
        logger.info(f"Registered tool endpoint: {tool_name} -> {endpoint}")
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tool_endpoints.keys())
