#!/usr/bin/env python3
"""
Visual tools module for the Medical Visual Agent system.
This module provides visual tools such as object detection, depth estimation, etc.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image, ImageDraw

from .tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

class VisualTools:
    """
    Visual tools for medical image analysis.
    Provides interfaces for REVPT-style visual tools.
    """
    
    def __init__(self, tool_executor: Optional[ToolExecutor] = None, use_mock: bool = False):
        """
        Initialize visual tools.
        
        Args:
            tool_executor: Tool executor instance
            use_mock: Whether to use mock implementations
        """
        self.tool_executor = tool_executor or ToolExecutor(use_mock=use_mock)
        self.current_image = None
        self.zoomed_images = {}
    
    def set_current_image(self, image: Image.Image):
        """
        Set the current image for tool operations.
        
        Args:
            image: PIL Image to set as current
        """
        self.current_image = image
        # Reset zoomed images when a new base image is set
        self.zoomed_images = {}
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available visual tools with their descriptions.
        
        Returns:
            Dictionary of tool names to their descriptions
        """
        return {
            "object_detection": {
                "required_args": ["objects"],
                "description": "Detect objects of specified types in the image"
            },
            "depth_estimation": {
                "required_args": [],
                "description": "Generate a depth map of the image"
            },
            "edge_detection": {
                "required_args": [],
                "description": "Detect edges in the image"
            },
            "zoom_tool": {
                "required_args": ["bbox", "factor"],
                "description": "Zoom in on a specific region of the image"
            },
            "visual_grounding": {
                "required_args": ["query"],
                "description": "Find a specific element in the image based on text query"
            }
        }
    
    def object_detection(
        self, 
        objects: List[str], 
        image: Optional[Image.Image] = None,
        image_id: int = 0
    ) -> Dict[str, Any]:
        """
        Detect objects in an image.
        
        Args:
            objects: List of object types to detect
            image: Optional image (uses current image if not provided)
            image_id: Image identifier for multi-image scenarios
        
        Returns:
            Detection results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for object detection")
            return {"error": "No image provided"}
        
        # Execute tool
        result = self.tool_executor.execute_tool(
            "object_detection",
            image=image,
            objects=objects,
            image_id=image_id
        )
        
        # Format the response
        detections = result.get("detections", [])
        formatted_result = {
            "message": f"Detected {len(detections)} object(s) in image {image_id}",
            "detections": detections
        }
        
        return formatted_result
    
    def depth_estimation(
        self, 
        image: Optional[Image.Image] = None,
        image_id: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a depth map for an image.
        
        Args:
            image: Optional image (uses current image if not provided)
            image_id: Image identifier for multi-image scenarios
        
        Returns:
            Depth estimation results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for depth estimation")
            return {"error": "No image provided"}
        
        # Execute tool
        result = self.tool_executor.execute_tool(
            "depth_estimation",
            image=image,
            image_id=image_id
        )
        
        # Format the response
        return {
            "message": f"The colored depth map for image {image_id}",
            "depth_map": result.get("depth_map", "")
        }
    
    def edge_detection(
        self, 
        image: Optional[Image.Image] = None,
        image_id: int = 0
    ) -> Dict[str, Any]:
        """
        Detect edges in an image.
        
        Args:
            image: Optional image (uses current image if not provided)
            image_id: Image identifier for multi-image scenarios
        
        Returns:
            Edge detection results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for edge detection")
            return {"error": "No image provided"}
        
        # Execute tool
        result = self.tool_executor.execute_tool(
            "edge_detection",
            image=image,
            image_id=image_id
        )
        
        # Format the response
        return {
            "message": f"Detected edges in image {image_id}",
            "edges": result.get("edges", "")
        }
    
    def zoom_tool(
        self, 
        bbox: List[float], 
        factor: float = 1.5,
        image: Optional[Image.Image] = None,
        image_id: int = 0
    ) -> Dict[str, Any]:
        """
        Zoom in on a specific region of an image.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] to zoom in on
            factor: Zoom factor
            image: Optional image (uses current image if not provided)
            image_id: Image identifier for multi-image scenarios
        
        Returns:
            Zoomed image results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for zoom")
            return {"error": "No image provided"}
        
        # Execute tool
        result = self.tool_executor.execute_tool(
            "zoom_tool",
            image=image,
            bbox=bbox,
            factor=factor,
            image_id=image_id
        )
        
        # Store the zoomed image for later use
        zoomed_id = result.get("zoomed_image_id", image_id)
        self.zoomed_images[zoomed_id] = result.get("zoomed_image", None)
        
        # Format the response
        return {
            "message": f"Zoomed image {image_id} on {bbox} with {factor}x magnification",
            "zoomed_image_id": zoomed_id,
            "bbox": bbox,
            "factor": factor
        }
    
    def visual_grounding(
        self, 
        query: str, 
        image: Optional[Image.Image] = None,
        image_id: int = 0
    ) -> Dict[str, Any]:
        """
        Find a specific element in an image based on text query.
        
        Args:
            query: Text query describing what to find
            image: Optional image (uses current image if not provided)
            image_id: Image identifier for multi-image scenarios
        
        Returns:
            Visual grounding results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for visual grounding")
            return {"error": "No image provided"}
        
        # Execute tool
        result = self.tool_executor.execute_tool(
            "visual_grounding",
            image=image,
            query=query,
            image_id=image_id
        )
        
        # Format the response
        bbox = result.get("bbox", [])
        confidence = result.get("confidence", 0.0)
        
        if bbox and confidence > 0.5:
            message = f"Found '{query}' at {bbox} with {confidence:.2f} confidence"
            found = True
        else:
            message = f"No matching element found for '{query}'"
            found = False
        
        return {
            "message": message,
            "query": query,
            "bbox": bbox,
            "confidence": confidence,
            "found": found
        }
    
    def get_tool_result_visualization(
        self, 
        tool_name: str, 
        result: Dict[str, Any],
        image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Get a visualization of a tool result.
        
        Args:
            tool_name: Name of the tool
            result: Tool result
            image: Optional base image (uses current image if not provided)
        
        Returns:
            Visualization image
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for visualization")
            return Image.new("RGB", (400, 300), color="white")
        
        # Create a copy of the image for drawing
        visualization = image.copy()
        draw = ImageDraw.Draw(visualization)
        
        if tool_name == "object_detection":
            detections = result.get("detections", [])
            for detection in detections:
                bbox = detection.get("bbox", [])
                if len(bbox) == 4:
                    # Draw bounding box
                    draw.rectangle(bbox, outline="red", width=2)
                    # Draw label
                    label = detection.get("label", "")
                    confidence = detection.get("confidence", 0.0)
                    draw.text((bbox[0], bbox[1]-15), f"{label}: {confidence:.2f}", fill="red")
        
        elif tool_name == "visual_grounding":
            bbox = result.get("bbox", [])
            if len(bbox) == 4:
                # Draw bounding box
                draw.rectangle(bbox, outline="green", width=2)
                # Draw query
                query = result.get("query", "")
                draw.text((bbox[0], bbox[1]-15), query, fill="green")
        
        return visualization
