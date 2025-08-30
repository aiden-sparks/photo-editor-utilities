import cv2
import numpy as np

# ------------------------------
# Utility: Convert to Lab for luminance & color balance
# ------------------------------
def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)

def from_lab(lab):
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

# ------------------------------
# Step 1: Normalize image
# ------------------------------
def normalize_image(img):
    lab = to_lab(img)
    L, A, B = cv2.split(lab)

    # Exposure normalization → target midtone ~128
    median_l = np.median(L)
    exposure_scale = 128.0 / (median_l + 1e-6)
    L = L * exposure_scale

    # Contrast normalization → stretch histogram slightly
    p2, p98 = np.percentile(L, (2, 98))
    if p98 > p2:
        L = (L - p2) * (255.0 / (p98 - p2))

    # White balance normalization → gray world assumption
    mean_a, mean_b = np.mean(A), np.mean(B)
    A = A - (mean_a - 128)
    B = B - (mean_b - 128)

    # Clip and enforce float32 for all
    L = np.clip(L, 0, 255).astype(np.float32)
    A = np.clip(A, 0, 255).astype(np.float32)
    B = np.clip(B, 0, 255).astype(np.float32)

    lab_norm = cv2.merge([L, A, B])  # now consistent
    return from_lab(lab_norm)

# ------------------------------
# Step 2: Analyze image stats (for adaptive filter scaling)
# ------------------------------
def analyze_image(img):
    lab = to_lab(img)
    L, A, B = cv2.split(lab)

    stats = {
        "luminance_mean": np.mean(L),
        "luminance_std": np.std(L),   # contrast proxy
        "color_temp_offset": (np.mean(A) - 128, np.mean(B) - 128)
    }
    return stats

# ------------------------------
# Step 3: Film look filter
# ------------------------------
def apply_film_filter(img, stats):
    out = img.copy().astype(np.float32) / 255.0

    # Adaptive exposure bump
    target_mid = 0.5
    exposure_factor = (target_mid / (stats["luminance_mean"] / 255.0))
    exposure_factor = np.clip(exposure_factor, 0.8, 1.2)  # limit strength
    out = np.clip(out * exposure_factor, 0, 1)

    # Tone curve (gentle S-curve for film look)
    def s_curve(x, strength=0.2):
        return x + strength * (x - x**2)
    out = s_curve(out)

    # Color grading: push shadows cool, highlights warm
    shadows_mask = out < 0.4
    highlights_mask = out > 0.6
    out[..., 2][shadows_mask[..., 2]] *= 1.05  # boost blue in shadows
    out[..., 1][highlights_mask[..., 1]] *= 1.03  # boost green in highlights
    out[..., 0][highlights_mask[..., 0]] *= 1.05  # boost red in highlights

    # Gentle desaturation (film look)
    hsv = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.9
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Film grain
    grain_strength = 0.05
    noise = np.random.normal(0, grain_strength, out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 1)

    return (out * 255).astype(np.uint8)

# ------------------------------
# Step 4: Full pipeline
# ------------------------------
def film_pipeline(img):
    img_norm = normalize_image(img)
    stats = analyze_image(img_norm)
    out = apply_film_filter(img_norm, stats)
    return out


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    img = cv2.imread("./TEST_IMGS/IMG_6633.jpg")
    result = film_pipeline(img)
    cv2.imwrite("film_output.jpg", result)
