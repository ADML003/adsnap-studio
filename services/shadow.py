# Import necessary modules for type hints, HTTP requests, and base64 encoding
from typing import Dict, Any, List, Optional, Tuple
import requests
import base64
import logging
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO

try:
    from PIL import Image, ImageFilter, ImageOps, ImageColor
except Exception:
    Image = None  # Pillow optional for local fallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


def _is_valid_hex_color(value: str) -> bool:
    return bool(HEX_COLOR_RE.match(value))


def add_shadow(
    api_key: str,
    image_data: Optional[bytes] = None,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    shadow_type: str = "regular",
    background_color: Optional[str] = None,
    background_transparent: Optional[bool] = None,
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
    use_local_fallback: bool = False,
    return_format: str = "json",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add shadow to an image using the Bria AI API.
    
    This function calls the Bria AI shadow API to add realistic shadows to product images.
    It supports both regular and floating shadows with customizable properties.
    
    Args:
        api_key: Bria AI API key for authentication
        image_data: Image data in bytes (optional if image_url provided)
        image_url: URL of the image (optional if image_data provided)
        shadow_type: Type of shadow. Accepts aliases: "regular"|"float"|"natural"|"drop".
        background_color: Optional background color in hex format (e.g., "#FFFFFF")
        background_transparent: If True, output background will be transparent (overrides background_color)
        shadow_color: Shadow color in hex format (default: black "#000000")
        shadow_offset: [x, y] offset for shadow positioning (default: [0, 15])
        shadow_intensity: Shadow opacity/intensity from 0-100 (default: 60)
        shadow_blur: Shadow blur amount for softness
        shadow_width: Optional shadow width for float shadows
        shadow_height: Optional shadow height for float shadows (default: 70)
        sku: Optional SKU identifier for tracking
        force_rmbg: Whether to force background removal before adding shadow
        content_moderation: Whether to enable content moderation checks
        image_path: Optional local image path to load if bytes/url not provided
        timeout: Request timeout for API calls
        session: Optional requests.Session to reuse
        use_local_fallback: If True or on API error, generate a basic local shadow with Pillow
        return_format: Output format for local fallback: "json" (default). Currently API returns JSON only.
        output_path: Optional path to save the resulting image (local fallback only)
    
    Returns:
        Dict containing the API response with the processed image
    """
    # Normalize shadow type aliases early
    original_shadow_type = shadow_type
    shadow_type = shadow_type.lower()
    alias_map = {"natural": "regular", "drop": "regular"}
    shadow_type = alias_map.get(shadow_type, shadow_type)

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
        raise ValueError(f"shadow_type must be one of {allowed_shadow_types} (received: {original_shadow_type})")

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
    # background_transparent overrides background_color when True
    if background_transparent is True:
        background_color = None

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
    
   
    # If image_data is a string and looks like a URL, treat as image_url
    if isinstance(image_data, str) and image_data.startswith("http"):
        image_url = image_data
        image_data = None

    # Allow loading from local path when provided
    if image_path and not image_data and not image_url:
        with open(image_path, "rb") as f:
            image_data = f.read()

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
    
    def _hex_to_rgba(hex_color: str, opacity_pct: int) -> Tuple[int, int, int, int]:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join(c*2 for c in hex_color)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(255 * (max(0, min(100, opacity_pct)) / 100.0))
        return r, g, b, a

    def _local_float_shadow(img: Any) -> Any:
        w, h = img.size
        bg = Image.new('RGBA', (w, h), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))

        # Create an ellipse for float shadow
        sw = shadow_width if shadow_width is not None else int(w * 0.6)
        sh = shadow_height if shadow_height is not None else int(h * 0.1)
        sw = max(1, abs(sw))
        sh = max(1, abs(sh))

        ellipse = Image.new('RGBA', (sw, sh), (0, 0, 0, 0))
        mask = Image.new('L', (sw, sh), 0)
        mask_draw = Image.new('L', (sw, sh), 0)
        # Simple oval mask
        mask = ImageOps.expand(mask, 0)
        m = Image.new('L', (sw, sh), 0)
        m_draw = Image.new('L', (sw, sh), 0)
        # Use a filled ellipse via ImageDraw (lazy import)
        from PIL import ImageDraw
        d = ImageDraw.Draw(mask)
        d.ellipse([(0, 0), (sw - 1, sh - 1)], fill=255)
        rgba = _hex_to_rgba(shadow_color, shadow_intensity)
        shadow_img = Image.new('RGBA', (sw, sh), rgba)
        shadow_img.putalpha(mask)
        blur_amt = shadow_blur if shadow_blur is not None else 20
        if blur_amt > 0:
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_amt))
        # Position near bottom center + offset
        x = (w - sw) // 2 + (shadow_offset[0] if shadow_offset else 0)
        y = h - sh - 10 + (shadow_offset[1] if shadow_offset else 15)
        bg.alpha_composite(shadow_img, (x, y))
        # Composite product image on top
        bg.alpha_composite(img)
        return bg

    def _local_drop_shadow(img: Any) -> Any:
        # Use alpha channel if present to create a drop shadow
        w, h = img.size
        bg = Image.new('RGBA', (w, h), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        alpha = img.split()[-1]
        # Create shadow from alpha
        rgba = _hex_to_rgba(shadow_color, shadow_intensity)
        shadow = Image.new('RGBA', (w, h), rgba)
        shadow.putalpha(alpha)
        blur_amt = shadow_blur if shadow_blur is not None else 15
        if blur_amt > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_amt))
        off_x = shadow_offset[0] if shadow_offset else 0
        off_y = shadow_offset[1] if shadow_offset else 15
        # First composite shadow then the image
        bg.alpha_composite(shadow, (off_x, off_y))
        bg.alpha_composite(img)
        return bg

    def _apply_local(image_bytes: bytes) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Local fallback requires Pillow; please install Pillow.")
        img = Image.open(BytesIO(image_bytes)).convert('RGBA')
        out = _local_float_shadow(img) if shadow_type == 'float' else _local_drop_shadow(img)
        buf = BytesIO()
        out.save(buf, format='PNG')
        result_bytes = buf.getvalue()
        b64 = base64.b64encode(result_bytes).decode('utf-8')
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(result_bytes)
        return {
            'source': 'local',
            'image_base64': b64,
            'result_url': None,
            'shadow_type': shadow_type,
            'background_color': background_color,
        }

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
        # Use local fallback if requested explicitly
        if use_local_fallback:
            # Need image bytes for local processing; fetch when only URL provided
            if not image_data and image_url:
                resp = sess.get(image_url, timeout=timeout)
                resp.raise_for_status()
                image_bytes = resp.content
            else:
                image_bytes = image_data  # type: ignore
            return _apply_local(image_bytes)

        response = sess.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()

        logger.debug("Response status: %s", response.status_code)
        # Parse and return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # If API fails and local fallback is allowed, try local generation
        if (image_data or image_url) and not use_local_fallback and Image is not None:
            try:
                logger.warning("API failed (%s). Falling back to local shadow.", e)
                # Fetch bytes if only URL provided
                sess = session or requests.Session()
                if not image_data and image_url:
                    resp = sess.get(image_url, timeout=timeout)
                    resp.raise_for_status()
                    image_bytes = resp.content
                else:
                    image_bytes = image_data  # type: ignore
                return _apply_local(image_bytes)
            except Exception as le:
                raise RuntimeError(f"Shadow addition request failed and local fallback also failed: {le}")
        # Bubble up a clearer error with context
        raise RuntimeError(f"Shadow addition request failed: {e}")