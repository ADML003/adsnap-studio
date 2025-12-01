# Import necessary modules for type hints, HTTP requests, and base64 encoding
from typing import Dict, Any, List, Optional
import requests
import base64

def add_shadow(
    api_key: str,
    image_data: bytes = None,
    image_url: str = None,
    shadow_type: str = "regular",
    background_color: Optional[str] = None,
    shadow_color: str = "#000000",
    shadow_offset: List[int] = [0, 15],
    shadow_intensity: int = 60,
    shadow_blur: Optional[int] = None,
    shadow_width: Optional[int] = None,
    shadow_height: Optional[int] = 70,
    sku: Optional[str] = None,
    force_rmbg: bool = False,
    content_moderation: bool = False
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
    
    # Prepare the base request payload with required shadow parameters
    data = {
        'shadow_type': shadow_type,
        'shadow_color': shadow_color,
        'shadow_intensity': shadow_intensity,
        'force_rmbg': force_rmbg,
        'content_moderation': content_moderation,
        'shadow_offset': shadow_offset
    }
    
    # Add image source - either URL or base64-encoded file data
    if image_url:
        # Use image URL if provided
        data['image_url'] = image_url
    elif image_data:
        
        data['file'] = base64.b64encode(image_data).decode('utf-8')
    else:
        # At least one image source must be provided
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
        # Debug logging: print request details
        print(f"Making request to: {url}")
        print(f"Headers: {headers}")
        print(f"Data: {data}")
        
        # Make POST request to the Bria AI shadow API
        response = requests.post(url, headers=headers, json=data)
        # Raise an exception if the request failed (4xx or 5xx status codes)
        response.raise_for_status()
        
        # Debug logging: print response details
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        # Parse and return the JSON response
        return response.json()
    except Exception as e:
        # Handle any errors that occur during the API request
        raise Exception(f"Shadow addition failed: {str(e)}") 