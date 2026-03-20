#!/usr/bin/env python3
"""
Common utility functions used across the medical visual agent system.
"""

import io
import json
import base64
import logging
from typing import Any, Dict, List, Union, Optional, Tuple
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def safe_json_loads(json_str: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robust JSON loader for model outputs (from try.py).
    - strips triple backticks and optional language labels (```json)
    - attempts several heuristics (trailing commas, single->double quotes, unquoted keys)
    - extracts first JSON object/array if the full string can't parse
    - handles placeholder values like [x, y] and comments
    - fixes common LLM JSON generation issues
    """
    import re
    
    if not json_str:
        raise ValueError("Empty string provided to JSON parser")
    
    cleaned = json_str.strip()

    # Strip triple backticks and optional language label (e.g. ```json\n{...}\n```)
    if cleaned.startswith("```"):
        # Remove the outer fences and any language label on the first line
        parts = cleaned.split("```", 2)
        # parts = ['', 'json\n{...}', ''] or ['', '{...}', '']
        inner = parts[1] if len(parts) > 1 else ""
        # If inner begins with a language identifier (single word + newline), drop that first line
        if "\n" in inner:
            first_line, rest = inner.split("\n", 1)
            if re.fullmatch(r"[A-Za-z0-9_+\-]+", first_line.strip()):
                cleaned = rest.strip()
            else:
                cleaned = inner.strip()
        else:
            cleaned = inner.strip()
    
    # Also drop a leading "json\n" or "python\n" that may remain
    if re.match(r'^[A-Za-z0-9_+\-]+\s*\n', cleaned):
        cleaned = re.sub(r'^[A-Za-z0-9_+\-]+\s*\n', '', cleaned, count=1).strip()
    
    # Fix common issues before parsing
    
    # First, replace any [x, y] placeholders with [0, 0], including those with comments
    cleaned = re.sub(r'\[\s*x\s*,\s*y\s*\](?:\s*//.*?$)?', '[0, 0]', cleaned, flags=re.MULTILINE)
    
    # Remove any remaining comments (// style)
    cleaned = re.sub(r'\s*//.*?$', '', cleaned, flags=re.MULTILINE)
    
    # Handle more complex cases like "coords": [x, y] with or without comments
    cleaned = re.sub(r'"coords"\s*:\s*\[\s*[a-zA-Z]+\s*,\s*[a-zA-Z]+\s*\](?:\s*//.*?$)?', '"coords": [0, 0]', cleaned, flags=re.MULTILINE)
    
    # Fix Step-array pattern: [ "Step 1": {...} ] -> { "Step 1": {...} }
    try:
        if cleaned.startswith("[") and re.search(r'\[\s*"?Step\s+\d+"?\s*:', cleaned):
            if cleaned.endswith("]"):
                cleaned = "{" + cleaned[1:-1] + "}"
    except Exception:
        pass

    # Try straightforward parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # diagnostics
        logger.debug(f"Initial JSON parsing failed: {e}")
        logger.debug(f"Error location: line {e.lineno}, column {e.colno}, position {e.pos}")

    # Heuristic fixes
    fixed = cleaned

    # Remove obvious leading garbage before the first JSON bracket
    first_brace = re.search(r'([\{\[])', fixed)
    if first_brace and first_brace.start() > 0:
        fixed = fixed[first_brace.start():]

    # 1) remove trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

    # 2) attempt to quote unquoted keys like: { key: ... } -> { "key": ... }
    fixed = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)(\s*:)', r'\1"\2"\3', fixed)

    # 3) convert single quotes to double when it's likely JSON-like
    if "'" in fixed and '"' in fixed and fixed.count("'") < fixed.count('"') * 2:
        # conservative convert only simple cases
        fixed = fixed.replace("':", '":').replace("'", '"')
    elif "'" in fixed and '"' not in fixed:
        fixed = fixed.replace("'", '"')

    # Try again
    try:
        return json.loads(fixed)
    except Exception as e2:
        logger.debug(f"Fixed version parsing failed; trying substring extraction: {e2}")

    # Try to find the first JSON object/array substring and parse it
    for start_char in ['{', '[']:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        candidate = cleaned[start:]
        try:
            return json.loads(candidate)
        except Exception:
            # try removing trailing non-brace text
            # naive: find last closing brace and slice
            last_close = max(candidate.rfind('}'), candidate.rfind(']'))
            if last_close > 0:
                try:
                    return json.loads(candidate[:last_close+1])
                except Exception:
                    pass

    # If all attempts fail, raise clear error with snippet
    raise ValueError(f"Could not parse JSON response (first 200 chars): {cleaned[:200]}")

def encode_image_to_data_uri(image: Image.Image) -> str:
    """
    Encode a PIL image to a data URI for use in prompts.
    
    Args:
        image: PIL Image to encode
    
    Returns:
        Data URI string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def decode_data_uri_to_image(data_uri: str) -> Image.Image:
    """
    Decode a data URI to a PIL Image.
    
    Args:
        data_uri: Data URI to decode
    
    Returns:
        PIL Image
    """
    # Extract the base64 data
    if "base64," in data_uri:
        base64_data = data_uri.split("base64,")[1]
    else:
        base64_data = data_uri
    
    # Decode and convert to PIL Image
    img_data = base64.b64decode(base64_data)
    return Image.open(io.BytesIO(img_data))

def merge_dicts_recursive(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence for conflicting keys)
    
    Returns:
        Merged dictionary
    """
    merged = dict1.copy()
    
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts_recursive(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def extract_coordinates(coord_str: Union[str, List[float], Dict[str, Any]]) -> List[float]:
    """
    Extract coordinates from various formats.
    
    Args:
        coord_str: Coordinate string, list, or dictionary
    
    Returns:
        List of coordinates [x, y]
    """
    if isinstance(coord_str, list):
        if len(coord_str) >= 2:
            return [float(coord_str[0]), float(coord_str[1])]
    elif isinstance(coord_str, dict):
        if "x" in coord_str and "y" in coord_str:
            return [float(coord_str["x"]), float(coord_str["y"])]
    elif isinstance(coord_str, str):
        # Try to parse as JSON
        try:
            coords = json.loads(coord_str)
            if isinstance(coords, list) and len(coords) >= 2:
                return [float(coords[0]), float(coords[1])]
            elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                return [float(coords["x"]), float(coords["y"])]
        except json.JSONDecodeError:
            # Try to extract coordinates from string like "(123, 456)"
            import re
            match = re.search(r"\(?(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\)?", coord_str)
            if match:
                return [float(match.group(1)), float(match.group(2))]
    
    raise ValueError(f"Could not extract coordinates from: {coord_str}")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for the filesystem.
    
    Args:
        filename: Filename to sanitize
    
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Trim whitespace and ensure non-empty
    filename = filename.strip()
    if not filename:
        filename = "unnamed"
    
    return filename

def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the overlap between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU (Intersection over Union) value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    return intersection_area / union_area if union_area > 0 else 0.0

def convert_bbox_to_center_point(bbox: Union[List[float], Tuple[float, ...]]) -> Tuple[float, float]:
    """
    Convert a bounding box [x1, y1, x2, y2] to a center point [center_x, center_y].
    
    Args:
        bbox: List or tuple of 4 values [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    if not bbox or len(bbox) != 4:
        # Default fallback if bbox is invalid
        return (100, 100)
    
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    return (center_x, center_y)

def is_point_in_bbox(point: List[float], bbox: List[float]) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: Point coordinates [x, y]
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        True if the point is inside the bounding box, False otherwise
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    
    return x1 <= x <= x2 and y1 <= y <= y2

def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Normalize a bounding box to be within image dimensions.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Normalized bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x1, x2 = x2, x1 + 1
    if y2 <= y1:
        y1, y2 = y2, y1 + 1
    
    return [x1, y1, x2, y2]
