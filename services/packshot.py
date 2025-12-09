# Import type hints, HTTP client, and base64 for encoding image bytes
from typing import Dict, Any
import requests
import base64

def create_packshot(
    api_key: str,
    image_data: bytes,
    background_color: str = "#FFFFFF",
    sku: str = None,
    force_rmbg: bool = False,
    content_moderation: bool = False
) -> Dict[str, Any]:
    """
    Create a studio-style packshot via the Bria AI API.

    This calls Bria's packshot endpoint to remove the background (if needed),
    apply a clean background color or transparency, and return a production-ready
    product image. Useful for e-commerce listings and catalogs.

    Args:
        api_key: Bria AI API key for authentication
        image_data: Raw image bytes to process
        background_color: Background color hex (e.g., "#FFFFFF") or "transparent"
        sku: Optional SKU identifier to track the product
        force_rmbg: Force background removal even if the image already has alpha
        content_moderation: Enable content safety checks

    Returns:
        Dict containing the API response payload with the processed packshot
    """
    url = "https://engine.prod.bria-api.com/v1/product/packshot"
    
    # API authentication and JSON content headers
    headers = {
        'api_token': api_key,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
  
    # Convert image bytes to base64 so it can be sent in JSON
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
   
 
    data = {
        'file': image_base64,
        'background_color': background_color,
        'force_rmbg': force_rmbg,
        'content_moderation': content_moderation
    }
    
   
    if sku:
        data['sku'] = sku
    
    try:
        print(f"Making request to: {url}")
        print(f"Headers: {headers}")
        print(f"Data keys: {list(data.keys())}")
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        return response.json()
    except Exception as e:
        raise Exception(f"Packshot creation failed: {str(e)}") 