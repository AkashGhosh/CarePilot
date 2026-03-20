#!/usr/bin/env python3
"""
Medical tools module for the Medical Visual Agent system.
This module provides specialized tools for medical image analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image, ImageDraw

from .tool_executor import ToolExecutor
from .visual_tools import VisualTools

logger = logging.getLogger(__name__)

class MedicalTools:
    """
    Medical tools for 3D Slicer-style medical image analysis.
    Extends visual tools with specialized medical functionality.
    """
    
    def __init__(self, visual_tools: Optional[VisualTools] = None, use_mock: bool = False):
        """
        Initialize medical tools.
        
        Args:
            visual_tools: Visual tools instance
            use_mock: Whether to use mock implementations
        """
        self.visual_tools = visual_tools or VisualTools(use_mock=use_mock)
        self.current_image = None
        self.segmentations = {}
        self.measurements = {}
    
    def set_current_image(self, image: Image.Image):
        """
        Set the current image for tool operations.
        
        Args:
            image: PIL Image to set as current
        """
        self.current_image = image
        # Also set for visual tools
        self.visual_tools.set_current_image(image)
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available medical tools with their descriptions.
        
        Returns:
            Dictionary of tool names to their descriptions
        """
        # Start with visual tools
        tools = self.visual_tools.get_available_tools()
        
        # Add medical-specific tools
        tools.update({
            "CLICK": {
                "required_args": ["coords"],
                "description": "Click at specific coordinates on the medical image"
            },
            "SEGMENT": {
                "required_args": ["coords", "scale"],
                "description": "Perform segmentation on the medical image scan with measurements/fiducials/lines"
            },
            "ZOOM": {
                "required_args": ["scale", "coords"],
                "description": "Zoom in/out on medical image - magnification changes"
            },
            "TEXT": {
                "required_args": ["text", "field"],
                "description": "Type text into an input field"
            },
            "SCROLL": {
                "required_args": ["direction", "amount"],
                "description": "Scroll content (UP, DOWN, LEFT, RIGHT)"
            },
            "COMPLETE": {
                "required_args": [],
                "description": "Mark task as complete (only for final step)"
            }
        })
        
        return tools
    
    def click(self, coords: List[float], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Simulate a click at the specified coordinates.
        
        Args:
            coords: Coordinates [x, y] to click
            image: Optional image (uses current image if not provided)
        
        Returns:
            Click action results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for click")
            return {"error": "No image provided"}
        
        # Perform visual grounding at click location to identify what was clicked
        result = self.visual_tools.tool_executor.execute_tool(
            "visual_grounding",
            image=image,
            query=f"Element at {coords}",
            coords=coords
        )
        
        # Extract the element that was clicked
        element_name = result.get("element_name", "unknown element")
        
        return {
            "message": f"Clicked at {coords}",
            "coords": coords,
            "element": element_name,
            "changed_regions": [
                {
                    "x1": coords[0] - 20,
                    "y1": coords[1] - 20,
                    "x2": coords[0] + 20,
                    "y2": coords[1] + 20
                }
            ],
            "confirm_text": f"Clicked on {element_name}"
        }
    
    def segment(
        self, 
        coords: List[float], 
        scale: float = 1.0,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Perform segmentation at the specified coordinates.
        
        Args:
            coords: Coordinates [x, y] to start segmentation
            scale: Scale factor for segmentation size
            image: Optional image (uses current image if not provided)
        
        Returns:
            Segmentation results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for segmentation")
            return {"error": "No image provided"}
        
        # Generate a unique ID for this segmentation
        import uuid
        seg_id = str(uuid.uuid4())[:8]
        
        # Mock segmentation result
        segmentation = {
            "id": seg_id,
            "center": coords,
            "scale": scale,
            "volume": round(4/3 * 3.14159 * scale**3 * 10, 2),  # Mock volume calculation
            "dimensions": [scale * 10, scale * 10, scale * 10]  # Mock dimensions in mm
        }
        
        # Store the segmentation
        self.segmentations[seg_id] = segmentation
        
        return {
            "message": f"Created segmentation at {coords} with scale {scale}",
            "segmentation_id": seg_id,
            "volume": segmentation["volume"],
            "dimensions": segmentation["dimensions"],
            "changed_regions": [
                {
                    "x1": coords[0] - scale * 50,
                    "y1": coords[1] - scale * 50,
                    "x2": coords[0] + scale * 50,
                    "y2": coords[1] + scale * 50
                }
            ],
            "confirm_text": f"Segmentation created with volume {segmentation['volume']} mm³"
        }
    
    def zoom(
        self, 
        scale: float, 
        coords: List[float] = None,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Zoom in/out on the medical image.
        
        Args:
            scale: Scale factor (>1 for zoom in, <1 for zoom out)
            coords: Optional center coordinates for zoom
            image: Optional image (uses current image if not provided)
        
        Returns:
            Zoom results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for zoom")
            return {"error": "No image provided"}
        
        # If no coords provided, zoom on center of image
        if not coords:
            width, height = image.size
            coords = [width // 2, height // 2]
        
        # Calculate bounding box for zoom
        zoom_width = image.width // (scale if scale > 0 else 1)
        zoom_height = image.height // (scale if scale > 0 else 1)
        
        x1 = max(0, coords[0] - zoom_width // 2)
        y1 = max(0, coords[1] - zoom_height // 2)
        x2 = min(image.width, coords[0] + zoom_width // 2)
        y2 = min(image.height, coords[1] + zoom_height // 2)
        
        # Use visual_tools.zoom_tool which is just a wrapper for the actual tool
        result = self.visual_tools.zoom_tool(
            bbox=[x1, y1, x2, y2],
            factor=scale,
            image=image
        )
        
        result.update({
            "changed_regions": [
                {
                    "x1": 0,
                    "y1": 0,
                    "x2": image.width,
                    "y2": image.height
                }
            ],
            "confirm_text": f"Zoomed {'in' if scale > 1 else 'out'} by factor of {scale}"
        })
        
        return result
    
    def text(
        self, 
        text: str, 
        field: str,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Simulate typing text into a field.
        
        Args:
            text: Text to type
            field: Field to type into
            image: Optional image (uses current image if not provided)
        
        Returns:
            Text action results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for text input")
            return {"error": "No image provided"}
        
        # First locate the field using visual grounding
        field_result = self.visual_tools.visual_grounding(
            query=field,
            image=image
        )
        
        field_bbox = field_result.get("bbox", [])
        if not field_bbox:
            return {
                "message": f"Could not find field '{field}'",
                "success": False
            }
        
        return {
            "message": f"Typed '{text}' into field '{field}'",
            "field": field,
            "text": text,
            "changed_regions": [
                {
                    "x1": field_bbox[0],
                    "y1": field_bbox[1],
                    "x2": field_bbox[2],
                    "y2": field_bbox[3]
                }
            ],
            "confirm_text": f"Text entered in {field}"
        }
    
    def scroll(
        self, 
        direction: str, 
        amount: int = 1,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Simulate scrolling in a direction.
        
        Args:
            direction: Direction to scroll (UP, DOWN, LEFT, RIGHT)
            amount: Number of units to scroll
            image: Optional image (uses current image if not provided)
        
        Returns:
            Scroll action results
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for scroll")
            return {"error": "No image provided"}
        
        # Normalize direction
        direction = direction.upper()
        if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return {
                "message": f"Invalid scroll direction: {direction}",
                "success": False
            }
        
        # Calculate affected region based on direction
        width, height = image.size
        if direction in ["UP", "DOWN"]:
            changed_region = {
                "x1": 0,
                "y1": 0,
                "x2": width,
                "y2": height
            }
        else:  # LEFT, RIGHT
            changed_region = {
                "x1": 0,
                "y1": 0,
                "x2": width,
                "y2": height
            }
        
        return {
            "message": f"Scrolled {direction} by {amount} units",
            "direction": direction,
            "amount": amount,
            "changed_regions": [changed_region],
            "confirm_text": f"Scrolled {direction}"
        }
    
    def complete(self) -> Dict[str, Any]:
        """
        Mark the task as complete.
        
        Returns:
            Completion results
        """
        return {
            "message": "Task marked as complete",
            "success": True,
            "changed_regions": [],
            "confirm_text": "Task completed"
        }
    
    def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool call from Target agent's reasoning.
        
        Args:
            tool_call: Dict with 'tool_call' and 'arguments' keys
            image: Optional image (uses current image if not provided)
        
        Returns:
            Tool result dict
        """
        image = image or self.current_image
        if not image:
            logger.error("No image provided for tool execution")
            return {"error": "No image provided"}
        
        tool_name = tool_call.get("tool_call", "").upper()
        args = tool_call.get("arguments", {})
        
        # Medical-specific tools
        if tool_name == "CLICK":
            coords = args.get("coords", [0, 0])
            return self.click(coords, image=image)
        
        elif tool_name == "SEGMENT":
            coords = args.get("coords", [0, 0])
            scale = args.get("scale", 1.0)
            return self.segment(coords, scale, image=image)
        
        elif tool_name == "ZOOM":
            scale = args.get("scale", 1.5)
            coords = args.get("coords", None)
            return self.zoom(scale, coords, image=image)
        
        elif tool_name == "TEXT":
            text = args.get("text", "")
            field = args.get("field", "")
            return self.text(text, field, image=image)
        
        elif tool_name == "SCROLL":
            direction = args.get("direction", "DOWN")
            amount = args.get("amount", 1)
            return self.scroll(direction, amount, image=image)
        
        elif tool_name == "COMPLETE":
            return self.complete()
        
        # Fall back to visual tools for other tools
        return self.visual_tools.tool_executor.execute_tool_call(tool_call, image=image)
