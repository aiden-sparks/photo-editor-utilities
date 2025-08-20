"""Utility functions for editing images using Pillow"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2


def adjust_contrast(img, val):
    # Normal settings range is about 0.65 to 1.35
    return ImageEnhance.Contrast(img).enhance(val)


def adjust_color_balance(img, val):
    return ImageEnhance.Color(img).enhance(val)


def adjust_brightness(img, val):
    return ImageEnhance.Brightness(img).enhance(val)


def adjust_sharpness(img, val):
    return ImageEnhance.Sharpness(img).enhance(val)

def adjust_highlights(img, value):
    """
    Adjusts the brightness of the highlights in an image.
    
    Parameters:
        img   : np.ndarray (BGR image, uint8, 0–255)
        value : float in [-1.0, 1.0]
                -1.0 = darker highlights
                +1.0 = brighter highlights
    
    Returns:
        np.ndarray (adjusted image, uint8)
    """
    # Convert to float in [0,1]
    img_float = img.astype(np.float32) / 255.0
    
    # Convert to luminance (grayscale) for weighting
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    
    # Weight curve: pixels near white get weight closer to 1,
    # shadows get lower weight. Smooth with a power curve.
    weight = np.clip(luminance, 0, 1) ** 2.2   # gamma curve emphasizes highlights
    
    # Compute adjustment factor per pixel
    # Highlights get full "value", midtones get ~50%, shadows much less
    adjustment = value * (0.2 + 0.8 * weight)  # ensures shadows still shift slightly
    
    # Expand adjustment to 3 channels
    adjustment = cv2.merge([adjustment]*3)
    
    # Apply adjustment
    result = img_float + adjustment
    
    # Clip to [0,1] and convert back
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def adjust_shadows(img, value):
    """
    Adjusts the brightness of the shadows in an image.
    
    Parameters:
        img   : np.ndarray (BGR image, uint8, 0–255)
        value : float in [-1.0, 1.0]
                -1.0 = darker shadows
                +1.0 = brighter shadows
    
    Returns:
        np.ndarray (adjusted image, uint8)
    """
    # Convert to float in [0,1]
    img_float = img.astype(np.float32) / 255.0
    
    # Convert to luminance (grayscale) for weighting
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    
    # Weight curve: pixels near black get weight closer to 1,
    # highlights get lower weight. Smooth with a power curve.
    weight = np.clip(1.0 - luminance, 0, 1) ** 2.2  # gamma curve emphasizes shadows
    
    # Compute adjustment factor per pixel
    # Shadows get full "value", midtones get ~50%, highlights much less
    adjustment = value * (0.2 + 0.8 * weight)  # ensures highlights still shift slightly
    
    # Expand adjustment to 3 channels
    adjustment = cv2.merge([adjustment]*3)
    
    # Apply adjustment
    result = img_float + adjustment
    
    # Clip to [0,1] and convert back
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def adjust_midtones(img, value):
    """
    Adjusts the brightness of the midtones (exposure) in an image.

    Parameters:
        img   : np.ndarray (BGR image, uint8, 0-255)
        value : float in [-1.0, 1.0]
                -1.0 = darker midtones
                +1.0 = brighter midtones

    Returns:
        np.ndarray (adjusted image, uint8)
    """
    # Convert to float in [0,1]
    img_float = img.astype(np.float32) / 255.0

    # Convert to luminance (grayscale) for weighting
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # Weight curve centered at 0.5 luminance (midtones). Use a Gaussian falloff
    # so highlights and shadows are affected less.
    sigma = 0.20  # controls width of midtone influence
    diff = luminance - 0.5
    weight = np.exp(-(diff * diff) / (2.0 * sigma * sigma)).astype(np.float32)

    # Compute adjustment factor per pixel
    # Midtones get full "value", highlights/shadows much less
    adjustment = value * (0.2 + 0.8 * weight)

    # Expand adjustment to 3 channels
    adjustment = cv2.merge([adjustment] * 3)

    # Apply adjustment
    result = img_float + adjustment

    # Clip to [0,1] and convert back
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def adjust_blacks(img, value):
    """
    Adjusts the brightness of the deepest shadows (blacks) at the left edge of the histogram.

    Parameters:
        img   : np.ndarray (BGR image, uint8, 0-255)
        value : float in [-1.0, 1.0]
                -1.0 = crush blacks (darker)
                +1.0 = lift blacks (brighter)

    Returns:
        np.ndarray (adjusted image, uint8)
    """
    img_float = img.astype(np.float32) / 255.0
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # Strong emphasis on the very lowest luminance values, quick falloff.
    sigma = 0.10
    weight = np.exp(-(luminance * luminance) / (2.0 * sigma * sigma)).astype(np.float32)

    # Blacks get most of the effect; the rest of the histogram is affected only slightly.
    adjustment = value * (0.05 + 0.95 * weight)
    adjustment = cv2.merge([adjustment] * 3)

    result = img_float + adjustment
    result = np.clip(result, 0.0, 1.0)
    return (result * 255).astype(np.uint8)


def adjust_whites(img, value):
    """
    Adjusts the brightness of the brightest highlights (whites) at the right edge of the histogram.

    Parameters:
        img   : np.ndarray (BGR image, uint8, 0-255)
        value : float in [-1.0, 1.0]
                -1.0 = dim whites (darker)
                +1.0 = boost whites (brighter)

    Returns:
        np.ndarray (adjusted image, uint8)
    """
    img_float = img.astype(np.float32) / 255.0
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # Strong emphasis on the very highest luminance values, quick falloff.
    sigma = 0.10
    diff = 1.0 - luminance
    weight = np.exp(-(diff * diff) / (2.0 * sigma * sigma)).astype(np.float32)

    # Whites get most of the effect; the rest is affected only slightly.
    adjustment = value * (0.05 + 0.95 * weight)
    adjustment = cv2.merge([adjustment] * 3)

    result = img_float + adjustment
    result = np.clip(result, 0.0, 1.0)
    return (result * 255).astype(np.uint8)


def apply_point_lut(img, x, y):
    work = img.convert("RGBA")

    pt_lut_u8 = np.interp(np.arange(0, 256), x, y).astype(np.uint8)

    R, G, B, A = [0, 1, 2, 3]
    source = work.split()
    out = []
    for band in [R, G, B]:
        out.append(source[band].point(pt_lut_u8))
    out.append(source[A]) # Don't use LUT on the alpha band

    merged_img = Image.merge('RGBA', out)
    return merged_img


def apply_channel_lut(img, x, y, channel):
    work = img.convert("RGBA")

    R, G, B, A = [0, 1, 2, 3]
    source = work.split()
    out = []
    for band in [R, G, B]:
        out.append(source[band])
    out.append(source[A]) # Don't use LUT on the alpha band
    
    channels = {"R": R, "G": G, "B": B}
    chnl_lut_u8 = np.interp(np.arange(0, 256), x, y).astype(np.uint8)
    out[channels[channel]] = out[channels[channel]].point(chnl_lut_u8)

    merged_img = Image.merge('RGBA', out)
    return merged_img


def adjust_tint(image, tint_value, scale=0.15):
    """
    Adjusts green-magenta tint in an image.
    
    Parameters:
        image: numpy array (H, W, 3), float32 in [0,1]
        tint_value: float in [-1.0, 1.0], where
                    -1.0 = max green, +1.0 = max magenta
        scale: strength of the effect (default=0.15)
    """
    # derive k from slider value
    k = tint_value * scale
    
    # build per-channel multipliers
    r_mult = 1 + k
    g_mult = 1 - 2 * k
    b_mult = 1 + k
    
    # apply
    out = np.empty_like(image)
    out[..., 0] = np.clip(image[..., 0] * r_mult, 0, 1)
    out[..., 1] = np.clip(image[..., 1] * g_mult, 0, 1)
    out[..., 2] = np.clip(image[..., 2] * b_mult, 0, 1)
    
    return out


def adjust_temp(image, temp_value, scale=0.15):
    """
    
    """
    # derive k from slider value
    k = temp_value * scale
    
    # build per-channel multipliers
    r_mult = 1 + k
    g_mult = 1 + k
    b_mult = 1 - 2 * k
    
    # apply
    out = np.empty_like(image)
    out[..., 0] = np.clip(image[..., 0] * r_mult, 0, 1)
    out[..., 1] = np.clip(image[..., 1] * g_mult, 0, 1)
    out[..., 2] = np.clip(image[..., 2] * b_mult, 0, 1)
    
    return out


def apply_disposable_camera_filter(img, *, grain_strength=0.08, vignette_strength=0.35, light_leak_strength=0.08):
    """
    Return a new image styled like a disposable camera shot.

    Args:
    - img: PIL.Image — input image
    - grain_strength: float — 0..~0.1, amount of film-like noise
    - vignette_strength: float — 0..1, edge darkening
    - light_leak_strength: float — 0..1, intensity of warm edge leak
    """
    original_mode = img.mode
    work = img.convert("RGBA")

    # 1) Color curve shifts with green tint - boost green channel, reduce red and blue
    def _build_lut(x_points, y_points):
        x = np.asarray(x_points, dtype=np.float32)
        y = np.asarray(y_points, dtype=np.float32)
        return np.interp(np.arange(256), x, y).astype(np.uint8)

    r_lut = _build_lut([0, 235], [5, 245])  # Reduced red slightly
    g_lut = _build_lut([0, 245], [15, 255])  # Boosted green more
    b_lut = _build_lut([0, 255], [0, 240])  # Reduced blue slightly

    r, g, b, a = work.split()
    r = r.point(r_lut)
    g = g.point(g_lut)
    b = b.point(b_lut)
    work = Image.merge("RGBA", [r, g, b, a])

    # 2) Add green tint overlay
    arr = np.array(work).astype(np.float32)
    green_tint = np.array([0.0, 0.15, 0.0], dtype=np.float32)  # Green tint
    arr[..., :3] = np.clip(arr[..., :3] + green_tint * 255 * 0.1, 0, 255)
    work = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    # 3) Slightly reduce contrast for faded look and bump saturation up
    work = ImageEnhance.Contrast(work).enhance(0.93)
    work = ImageEnhance.Color(work).enhance(1.02)
    work = ImageEnhance.Brightness(work).enhance(1.00)

    # 4) Brighten highlights using luminance-based weighting
    arr = np.array(work).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:]
    
    # Calculate perceptual luminance
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    
    # Create weight mask that emphasizes highlights
    # Highlights (0.7-1.0): full effect, Midtones (0.4-0.7): moderate effect, Shadows (0.0-0.4): minimal effect
    weight = np.zeros_like(luminance)
    
    # Shadows: minimal effect (0.0-0.4)
    shadow_mask = luminance < 0.4
    weight[shadow_mask] = luminance[shadow_mask] * 0.1
    
    # Midtones: moderate effect (0.4-0.7)
    midtone_mask = (luminance >= 0.4) & (luminance < 0.7)
    weight[midtone_mask] = 0.04 + (luminance[midtone_mask] - 0.4) * 0.2
    
    # Highlights: full effect (0.7-1.0)
    highlight_mask = luminance >= 0.7
    weight[highlight_mask] = 0.1 + (luminance[highlight_mask] - 0.7) * 0.3
    
    weight = weight[..., None]
    
    # Apply highlights brightening
    highlight_boost = 0.25  # Amount to brighten highlights
    adjusted_rgb = rgb + weight * highlight_boost * (1.0 - rgb) * 0.6
    adjusted_rgb = np.clip(adjusted_rgb, 0.0, 1.0)
    
    # Reconstruct the image
    out_arr = np.concatenate([adjusted_rgb, alpha], axis=-1)
    work = Image.fromarray((out_arr * 255.0).astype(np.uint8), mode="RGBA")

    # Convert to numpy for spatial effects
    arr = np.array(work).astype(np.float32)
    h, w = arr.shape[0], arr.shape[1]

    # 5) Vignette (radial falloff)
    if vignette_strength > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rx, ry = cx, cy
        nx = (xx - cx) / (rx + 1e-6)
        ny = (yy - cy) / (ry + 1e-6)
        r2 = nx * nx + ny * ny
        vignette = 1.0 - vignette_strength * np.clip(r2, 0.0, 1.0)
        vignette = vignette[..., None]
        arr[..., :3] = np.clip(arr[..., :3] * vignette, 0, 255)

    # 6) Film grain (additive, gaussian noise)
    if grain_strength > 0:
        noise = np.random.normal(loc=0.0, scale=255.0 * grain_strength, size=(h, w, 1)).astype(np.float32)
        arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)

    # 7) Subtle warm light leak from one edge
    if light_leak_strength > 0:
        # Left-edge gradient leak (H, W, 1)
        x = np.linspace(1.0, 0.0, w, dtype=np.float32)
        leak_mask = np.clip(x, 0.0, 1.0)[None, :]
        leak_mask = np.repeat(leak_mask, h, axis=0)  # (h, w)
        leak_mask = (leak_mask ** 2.5) * light_leak_strength  # (h, w)
        leak_mask = leak_mask[..., None]  # (h, w, 1)
        leak_color = np.array([255.0, 110.0, 20.0], dtype=np.float32)  # softer warm
        arr[..., :3] = np.clip(arr[..., :3] + leak_mask * leak_color * 0.15, 0, 255)

    out = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    # Final tiny contrast trim to glue it together
    out = ImageEnhance.Contrast(out).enhance(0.98)

    # Return in the original mode
    return out.convert(original_mode)
