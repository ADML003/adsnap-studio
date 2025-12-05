# Import necessary modules for type hints, HTTP requests, and base64 encoding
from typing import Dict, Any, Optional
import requests
import base64

def erase_foreground(
    api_key: str,
    image_data: bytes = None,
    image_url: str = None,
    content_moderation: bool = False
) -> Dict[str, Any]:
    """
    Erase the foreground from an image and intelligently generate the area behind it using Bria AI.
    
    This function removes the main subject/foreground from an image and uses AI to fill in
    the background area that was previously occluded. Useful for creating clean backgrounds
    or removing unwanted subjects from images.
    
    Args:
        api_key: Bria AI API key for authentication
        image_data: Image data in bytes (optional if image_url provided)
        image_url: URL of the image (optional if image_data provided)
        content_moderation: Whether to enable content moderation checks (default: False)
    
    Returns:
        Dict containing the API response with the processed image (foreground erased)
    
    Raises:
        ValueError: If neither image_data nor image_url is provided
        Exception: If the API request fails
    """
    # Bria AI erase foreground API endpoint
    url = "https://engine.prod.bria-api.com/v1/erase_foreground"
    
    # Set up authentication and content-type headers for the API request
    headers = {
        'api_token': api_key,  # API key for authentication
        'Accept': 'application/json',  # Expect JSON response
        'Content-Type': 'application/json'  # Sending JSON data
    }
    
    # Prepare the base request payload with content moderation setting
    data = {
        'content_moderation': content_moderation  # Enable/disable content safety checks
    }
    
    # Add image source - either URL or base64-encoded file data
    if image_url:
        # Use image URL if provided (image hosted online)
        data['image_url'] = image_url
    elif image_data:
        # Convert image bytes to base64 string for API transmission
        data['file'] = base64.b64encode(image_data).decode('utf-8')
    else:
        # At least one image source must be provided
        raise ValueError("Either image_data or image_url must be provided")
    
    try:
        # Debug logging: print request details for troubleshooting
        print(f"Making request to: {url}")
        print(f"Headers: {headers}")
        print(f"Data: {data}")
        
        # Make POST request to the Bria AI erase foreground API
        response = requests.post(url, headers=headers, json=data)
        # Raise an exception if the request failed (4xx or 5xx status codes)
        response.raise_for_status()
        
        # Debug logging: print response details
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        # Parse and return the JSON response containing the processed image
        return response.json()
    except Exception as e:
        # Handle any errors that occur during the API request
        raise Exception(f"Erase foreground failed: {str(e)}")


# Export the erase_foreground function for use in other modules
__all__ = ['erase_foreground'] 
