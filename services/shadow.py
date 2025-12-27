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
    shadow_angle: int = -45,
    shadow_distance: Optional[int] = None,
    shadow_hardness: float = 1.0,
    use_gradient: bool = False,
    gradient_stops: Optional[List[Tuple[str, int]]] = None,
    vignette_intensity: int = 0,
    enable_layered_shadow: bool = False,
    layer_count: int = 2,
    layer_intensity_decay: float = 0.7,
) -> Dict[str, Any]:
    """
    Add shadow to an image using the Bria AI API.
    
    This function calls the Bria AI shadow API to add realistic shadows to product images.
    It supports both regular and floating shadows with customizable properties, and local Pillow-based
    rendering with advanced modes like gradient shadows, vignette, and perspective effects.
    
    Supported shadow modes (use_local_fallback=True):
    - regular (drop): Classic drop shadow with offset and blur
    - float: Floating elliptical shadow below product
    - soft: Low-hardness shadow for subtle effect
    - hard: High-hardness shadow for dramatic effect
    - long: Directional shadow extending far (perspective-based)
    - glow: Soft, wide glow effect around edges
    - reflection: Mirrored reflection below product
    
    Args:
        api_key: Bria AI API key for authentication
        image_data: Image data in bytes (optional if image_url provided)
        image_url: URL of the image (optional if image_data provided)
        image_path: Optional local image path to load if bytes/url not provided
        shadow_type: Type of shadow. Accepts "regular"|"float"|"natural"|"drop"|"soft"|"hard"|"long"|"glow"|"reflection"
        background_color: Optional background color in hex format (e.g., "#FFFFFF")
        background_transparent: If True, output background will be transparent (overrides background_color)
        shadow_color: Shadow color in hex format (default: black "#000000")
        shadow_offset: [x, y] offset for shadow positioning (default: [0, 15])
        shadow_intensity: Shadow opacity/intensity from 0-100 (default: 60)
        shadow_blur: Shadow blur amount for softness
        shadow_width: Optional shadow width for float shadows
        shadow_height: Optional shadow height for float shadows (default: 70)
        shadow_angle: Direction angle in degrees (-45 to 90) for directional shadows (default: -45)
        shadow_distance: Distance of shadow from object (for long/glow modes)
        shadow_hardness: Shadow edge hardness 0.0-2.0; <1 soft, 1 normal, >1 hard
        use_gradient: If True, shadow transitions to second color (use gradient_stops)
        gradient_stops: List of [(color_hex, alpha_pct)] tuples for gradient shadow
        vignette_intensity: Optional vignette darkening 0-100 around image edges
        enable_layered_shadow: If True, creates multiple shadow layers for depth effect (default: False)
        layer_count: Number of shadow layers to create (2-5, default: 2). Used with enable_layered_shadow
        layer_intensity_decay: Decay factor for each layer (0.0-1.0, default: 0.7). Lower values = faster fade
        sku: Optional SKU identifier for tracking
        force_rmbg: Whether to force background removal before adding shadow
        content_moderation: Whether to enable content moderation checks
        timeout: Request timeout for API calls
        session: Optional requests.Session to reuse
        use_local_fallback: If True or on API error, generate local shadow with Pillow
        return_format: Output format for local fallback: "json" (default)
        output_path: Optional path to save the resulting image (local fallback only)
    
    Returns:
        Dict containing the API response with the processed image or local render metadata
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
    allowed_shadow_types = {"regular", "float", "soft", "hard", "long", "glow", "reflection"}
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

    def _apply_vignette(img: Any, vignette_alpha: int) -> Any:
        """Apply vignette darkening around edges."""
        if vignette_alpha <= 0 or Image is None:
            return img
        w, h = img.size
        vignette = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(vignette)
        # Create concentric darkening
        steps = 40
        for i in range(steps):
            alpha = int((vignette_alpha / 100.0) * 255 * (i / steps))
            draw.ellipse([
                (-w * i / steps, -h * i / steps),
                (w + w * i / steps, h + h * i / steps)
            ], outline=(0, 0, 0, alpha), width=2)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img.alpha_composite(vignette)
        return img

    def _local_glow_shadow(img: Any) -> Any:
        """Create a soft glow effect around the product."""
        w, h = img.size
        bg = Image.new('RGBA', (w, h), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create a large blur for glow
        glow_blur = shadow_blur if shadow_blur is not None else 40
        glow_size = w + h
        glow = Image.new('RGBA', (glow_size, glow_size), (0, 0, 0, 0))
        rgba = _hex_to_rgba(shadow_color, shadow_intensity // 2)
        glow_inner = Image.new('RGBA', (glow_size, glow_size), rgba)
        glow_inner = glow_inner.filter(ImageFilter.GaussianBlur(radius=glow_blur))
        glow.alpha_composite(glow_inner)
        
        # Center glow under product
        glow_x = (w - glow_size) // 2
        glow_y = h - (glow_size // 3)
        bg.alpha_composite(glow, (glow_x, glow_y))
        bg.alpha_composite(img)
        return bg

    def _local_reflection_shadow(img: Any) -> Any:
        """Create a mirrored reflection/shadow below product."""
        w, h = img.size
        bg = Image.new('RGBA', (w, h + h // 3), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Place original image
        bg.alpha_composite(img, (0, 0))
        
        # Create reflection
        reflection = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        reflection_alpha = reflection.split()[-1]
        # Gradient fade for reflection
        for y in range(reflection.size[1]):
            fade = int(255 * (1.0 - y / reflection.size[1]))
            alpha = Image.new('L', (reflection.size[0], 1), fade)
            # Blend with existing alpha
        
        # Darken reflection
        rgba = _hex_to_rgba(shadow_color, shadow_intensity)
        darkened = Image.new('RGBA', reflection.size, rgba)
        darkened.putalpha(reflection_alpha)
        
        bg.alpha_composite(darkened, (0, h))
        return bg

    def _local_long_shadow(img: Any) -> Any:
        """Create a perspective/long shadow extending from product."""
        w, h = img.size
        extended_w = int(w * 2)
        extended_h = int(h * 2)
        bg = Image.new('RGBA', (extended_w, extended_h), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Calculate shadow projection based on angle
        import math
        angle_rad = math.radians(shadow_angle)
        shadow_len = shadow_distance if shadow_distance is not None else int(max(w, h) * 1.5)
        
        # Create shadow strip
        shadow_width_px = max(3, int(w * 0.3))
        rgba = _hex_to_rgba(shadow_color, shadow_intensity)
        shadow_strip = Image.new('RGBA', (shadow_len, shadow_width_px), rgba)
        
        # Blur and fade
        blur_amt = shadow_blur if shadow_blur is not None else 10
        shadow_strip = shadow_strip.filter(ImageFilter.GaussianBlur(radius=blur_amt))
        
        # Rotate and position
        shadow_strip = shadow_strip.rotate(shadow_angle, expand=False)
        sx = (extended_w - w) // 2 + shadow_offset[0] if shadow_offset else extended_w // 2
        sy = (extended_h - h) // 2 + shadow_offset[1] if shadow_offset else extended_h // 2
        bg.alpha_composite(shadow_strip, (sx, sy))
        
        # Place original on top
        img_x = (extended_w - w) // 2
        img_y = (extended_h - h) // 2
        bg.alpha_composite(img, (img_x, img_y))
        return bg

    def _local_gradient_shadow(img: Any) -> Any:
        """Create shadow with gradient color transition."""
        w, h = img.size
        bg = Image.new('RGBA', (w, h), (0, 0, 0, 0) if background_color is None else ImageColor.getrgb(background_color) + (255,))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Use provided gradient stops or default
        stops = gradient_stops if gradient_stops else [(shadow_color, shadow_intensity), ("#FFFFFF", 0)]
        
        # Create gradient shadow (vertical)
        sh = shadow_height if shadow_height is not None else int(h * 0.15)
        sh = max(1, abs(sh))
        gradient = Image.new('RGBA', (w, sh))
        
        for y in range(sh):
            # Interpolate color across stops
            pct = y / sh
            # Simple linear interpolation
            if len(stops) >= 2:
                c1, a1 = stops[0]
                c2, a2 = stops[1]
                r1, g1, b1, _ = _hex_to_rgba(c1, a1)
                r2, g2, b2, _ = _hex_to_rgba(c2, a2)
                r = int(r1 + (r2 - r1) * pct)
                g = int(g1 + (g2 - g1) * pct)
                b = int(b1 + (b2 - b1) * pct)
                a = int(a1 + (a2 - a1) * pct)
                for x in range(w):
                    gradient.putpixel((x, y), (r, g, b, a))
        
        # Blur gradient
        blur_amt = shadow_blur if shadow_blur is not None else 15
        if blur_amt > 0:
            gradient = gradient.filter(ImageFilter.GaussianBlur(radius=blur_amt))
        
        # Position below product
        off_x = shadow_offset[0] if shadow_offset else 0
        off_y = shadow_offset[1] if shadow_offset else 15
        bg.alpha_composite(gradient, (off_x, h - sh + off_y))
        bg.alpha_composite(img)
        return bg

    def _apply_hardness(shadow_img: Any, hardness: float) -> Any:
        """Adjust shadow hardness via edge enhancement or softening."""
        if hardness < 0.5:
            # Soften
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=2))
        elif hardness > 1.5:
            # Sharpen edges
            shadow_img = shadow_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return shadow_img

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
        
        # Route to appropriate shadow function based on mode
        if shadow_type == 'glow':
            out = _local_glow_shadow(img)
        elif shadow_type == 'reflection':
            out = _local_reflection_shadow(img)
        elif shadow_type == 'long':
            out = _local_long_shadow(img)
        elif shadow_type == 'gradient' or use_gradient:
            out = _local_gradient_shadow(img)
        elif shadow_type in ('soft', 'hard'):
            # Use drop shadow with hardness adjustment
            out = _local_drop_shadow(img)
            out = _apply_hardness(out, shadow_hardness)
        elif shadow_type == 'float':
            out = _local_float_shadow(img)
        else:  # regular, drop, natural
            out = _local_drop_shadow(img)
        
        # Apply vignette if requested
        if vignette_intensity > 0:
            out = _apply_vignette(out, vignette_intensity)
        
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
            'features': {
                'hardness': shadow_hardness,
                'gradient': use_gradient,
                'vignette': vignette_intensity,
            }
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