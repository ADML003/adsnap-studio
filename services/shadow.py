# Import necessary modules for type hints, HTTP requests, and base64 encoding
from typing import Dict, Any, List, Optional
import requests
import base64
import logging
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


def _is_valid_hex_color(value: str) -> bool:
    return bool(HEX_COLOR_RE.match(value))


def add_shadow(
    api_key: str,
    image_data: Optional[bytes] = None,
    image_url: Optional[str] = None,
    shadow_type: str = "regular",
    background_color: Optional[str] = None,
    shadow_color: str = "#000000",
    shadow_offset: Optional[List[int]] = None,
    shadow_intensity: int = 60,
    shadow_blur: Optional[int] = None,
    shadow_width: Optional[int] = None,
    shadow_height: Optional[int] = 70,
    sku: Optional[str] = None,
    force_rmbg: bool = False,
    content_moderation: bool = False,
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Add shadow to an image using the Bria AI API.
    
    This function calls the Bria AI shadow API to add realistic shadows to product images.
    It supports both regular and floating shadows with customizable properties.
    
    Args:
        api_key: Bria AI API key for authentication
        image_data: Image data in bytes (optional if image_url provided)
        image_url: URL of the image (optional if image_data provided)
        shadow_type: Type of shadow ("regular" for ground shadow or "float" for floating shadow)
        background_color: Optional background color in hex format (e.g., "#FFFFFF")
        shadow_color: Shadow color in hex format (default: black "#000000")
        shadow_offset: [x, y] offset for shadow positioning (default: [0, 15])
        shadow_intensity: Shadow opacity/intensity from 0-100 (default: 60)
        shadow_blur: Shadow blur amount for softness
        shadow_width: Optional shadow width for float shadows
        shadow_height: Optional shadow height for float shadows (default: 70)
        sku: Optional SKU identifier for tracking
        force_rmbg: Whether to force background removal before adding shadow
        content_moderation: Whether to enable content moderation checks
    
    Returns:
        Dict containing the API response with the processed image
    """
    # Bria AI shadow API endpoint
    url = "https://engine.prod.bria-api.com/v1/product/shadow"
    
    # Set up authentication and content-type headers
    headers = {
        'api_token': api_key,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    

    # Input validation
    allowed_shadow_types = {"regular", "float"}
    if shadow_type not in allowed_shadow_types:
        raise ValueError(f"shadow_type must be one of {allowed_shadow_types}")

    if shadow_offset is None:
        shadow_offset = [0, 15]
    if (
        not isinstance(shadow_offset, list)
        or len(shadow_offset) != 2
        or not all(isinstance(v, int) for v in shadow_offset)
    ):
        raise ValueError("shadow_offset must be a list of two integers [x, y]")

    if not (0 <= shadow_intensity <= 100):
        raise ValueError("shadow_intensity must be between 0 and 100")

    if shadow_color and not _is_valid_hex_color(shadow_color):
        raise ValueError("shadow_color must be a valid hex color like #000000 or #000")
    if background_color and not _is_valid_hex_color(background_color):
        raise ValueError("background_color must be a valid hex color like #FFFFFF")

    if shadow_blur is not None and shadow_blur < 0:
        raise ValueError("shadow_blur must be >= 0")
    if shadow_width is not None and shadow_width <= 0:
        raise ValueError("shadow_width must be > 0")
    if shadow_height is not None and shadow_height <= 0:
        raise ValueError("shadow_height must be > 0")

    data = {
        'shadow_type': shadow_type,
        'shadow_color': shadow_color,
        'shadow_intensity': shadow_intensity,
        'force_rmbg': force_rmbg,
        'content_moderation': content_moderation,
        'shadow_offset': shadow_offset,
    }
    
   
    if image_url:
        data['image_url'] = image_url
    elif image_data:
        data['file'] = base64.b64encode(image_data).decode('utf-8')
    else:
        raise ValueError("Either image_data or image_url must be provided")
    
    # Add optional parameters only if they are specified
    if background_color:
        data['background_color'] = background_color
    if shadow_blur is not None:
        data['shadow_blur'] = shadow_blur
    if shadow_width is not None:
        data['shadow_width'] = shadow_width
    if shadow_height is not None:
        data['shadow_height'] = shadow_height
    if sku:
        data['sku'] = sku
    
    try:
        logger.debug("Making request to %s with data: %s", url, data)

        # Prepare session with retry logic for transient errors if not provided
        sess = session or requests.Session()
        if session is None:
            retries = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("POST",),
            )
            adapter = HTTPAdapter(max_retries=retries)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)

        # Make POST request to the Bria AI shadow API
        response = sess.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()

        logger.debug("Response status: %s", response.status_code)
        # Parse and return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Bubble up a clearer error with context
        raise RuntimeError(f"Shadow addition request failed: {e}")