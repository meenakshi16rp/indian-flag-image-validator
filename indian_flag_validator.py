import io, json, urllib.request
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

ASPECT_RATIO = 3 / 2
ASPECT_TOL = 0.01
COLOR_TOL_PCT = 5.0
CENTER_TOL_FRAC = 0.01
SPOKES_IDEAL = 24
SPOKES_TOL = 2

TARGETS = {
    "saffron": (255, 153, 51),
    "white": (255, 255, 255),
    "green": (19, 136, 8),
    "chakra_blue": (0, 0, 128),
}

def load_image(path_or_url, max_mb=5):
    def _is_url(s): return s.startswith("http")
    def _read_bytes(src):
        if _is_url(src):
            with urllib.request.urlopen(src) as r: b = r.read()
        else:
            with open(src, "rb") as f: b = f.read()
        if len(b) > max_mb * 1024 * 1024:
            raise ValueError("Image > 5MB")
        return b

    raw = _read_bytes(path_or_url)
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=raw)
        im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(im)[:, :, ::-1], im.size

def rgb_dev_pct(actual_rgb, target_rgb):
    diff = np.abs(np.array(actual_rgb) - np.array(target_rgb))
    return float(np.max(diff) / 255 * 100)

def mean_rgb(img_bgr, mask=None):
    if mask is not None:
        sel = img_bgr[mask]
    else:
        sel = img_bgr.reshape(-1, 3)
    return tuple(sel.mean(axis=0)[::-1])

def check_aspect_ratio(w, h):
    actual = w / h
    ok = abs(actual - ASPECT_RATIO) <= ASPECT_RATIO * ASPECT_TOL
    return {"status": "pass" if ok else "fail", "actual": f"{actual:.2f}"}

def check_stripes(img_bgr):
    h, w, _ = img_bgr.shape
    x1, x2 = int(0.2 * w), int(0.8 * w)
    thirds = h // 3
    bounds = {
        "saffron": (0, thirds - 1),
        "white": (thirds, 2 * thirds - 1),
        "green": (2 * thirds, h - 1),
    }
    color_report = {}
    stripe_heights = {}
    for stripe in ["saffron", "white", "green"]:
        y1, y2 = bounds[stripe]
        region = img_bgr[y1:y2+1, x1:x2]
        if stripe == "white":
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask_blue = cv2.inRange(hsv, (90, 50, 20), (140, 255, 255))
            mask_non_blue = mask_blue == 0
            mean_col = mean_rgb(region, mask_non_blue)
        else:
            mean_col = mean_rgb(region)
        dev = rgb_dev_pct(mean_col, TARGETS[stripe])
        color_report[stripe] = {
            "status": "pass" if dev <= COLOR_TOL_PCT else "fail",
            "deviation": f"{dev:.0f}%"
        }
        stripe_heights[stripe] = (y2 - y1 + 1) / h
    stripe_status = all(abs(v - 1/3) <= 0.01 for v in stripe_heights.values())
    stripe_report = {
        "status": "pass" if stripe_status else "fail",
        "top": f"{stripe_heights['saffron']:.2f}",
        "middle": f"{stripe_heights['white']:.2f}",
        "bottom": f"{stripe_heights['green']:.2f}"
    }
    return bounds, color_report, stripe_report

def detect_chakra(img_bgr, white_bounds):
    wy1, wy2 = white_bounds
    white_band = img_bgr[wy1:wy2+1]
    hsv = cv2.cvtColor(white_band, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (90, 50, 20), (140, 255, 255))
    mask2 = cv2.inRange(hsv, (100, 30, 10), (130, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 100.0, 0
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cy += wy1
    h, w, _ = img_bgr.shape
    offset_x = int(cx - w / 2)
    offset_y = int(cy - h / 2)
    chakra_pixels = white_band[mask > 0]
    if len(chakra_pixels) > 0:
        mean_col = np.mean(chakra_pixels[:, ::-1], axis=0)
        dev_blue = rgb_dev_pct(mean_col, TARGETS["chakra_blue"])
    else:
        dev_blue = 100.0
    spoke_count = count_spokes_enhanced(mask, int(cx), int(cy - wy1), radius)
    return (offset_x, offset_y), dev_blue, spoke_count

def count_spokes_enhanced(mask, cx, cy, radius):
    spokes1 = count_spokes_radial_smooth(mask, cx, cy, radius)
    spokes2 = count_spokes_edges(mask, cx, cy, radius)
    spokes3 = count_spokes_structural(mask, cx, cy, radius)
    spokes4 = count_spokes_fourier(mask, cx, cy, radius)
    results = [spokes1, spokes2, spokes3, spokes4]
    valid_results = [r for r in results if r > 10]
    if not valid_results:
        return max(results) if results else 0
    best_result = min(valid_results, key=lambda x: abs(x - SPOKES_IDEAL))
    return best_result

def count_spokes_radial_smooth(mask, cx, cy, radius):
    try:
        if radius < 5:
            return 0
        sample_radius = radius * 0.7
        angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        intensity_profile = []
        for angle in angles:
            x = int(cx + sample_radius * np.cos(angle))
            y = int(cy + sample_radius * np.sin(angle))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                neighborhood = mask[max(0, y-1):min(mask.shape[0], y+2),
                                  max(0, x-1):min(mask.shape[1], x+2)]
                intensity = np.mean(neighborhood) / 255.0
            else:
                intensity = 0
            intensity_profile.append(intensity)
        intensity_profile = np.array(intensity_profile)
        smoothed = ndimage.gaussian_filter1d(intensity_profile, sigma=2, mode='wrap')
        threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
        peaks = []
        for i in range(len(smoothed)):
            prev_idx = (i - 1) % len(smoothed)
            next_idx = (i + 1) % len(smoothed)
            if smoothed[i] > threshold and smoothed[i] >= smoothed[prev_idx] and smoothed[i] >= smoothed[next_idx]:
                peaks.append(i)
        grouped_peaks = []
        min_separation = 360 // 30
        for peak in peaks:
            if not grouped_peaks or min(abs(peak - gp) % 360 for gp in grouped_peaks) >= min_separation:
                grouped_peaks.append(peak)
        return len(grouped_peaks)
    except:
        return 0

def count_spokes_edges(mask, cx, cy, radius):
    try:
        roi_size = int(radius * 2.5)
        y1 = max(0, cy - roi_size // 2)
        y2 = min(mask.shape[0], cy + roi_size // 2)
        x1 = max(0, cx - roi_size // 2)
        x2 = min(mask.shape[1], cx + roi_size // 2)
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(radius*0.2),
                               minLineLength=int(radius*0.3), maxLineGap=int(radius*0.1))
        if lines is None:
            return 0
        center_x = cx - x1
        center_y = cy - y1
        radial_lines = 0
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            A = y2_l - y1_l
            B = x1_l - x2_l
            C = x2_l * y1_l - x1_l * y2_l
            if A*A + B*B > 0:
                distance = abs(A * center_x + B * center_y + C) / np.sqrt(A*A + B*B)
                if distance < radius * 0.1:
                    radial_lines += 1
        return radial_lines
    except:
        return 0

def count_spokes_structural(mask, cx, cy, radius):
    try:
        h, w = mask.shape
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        circular_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
        focused_mask = mask & circular_mask
        dist_transform = cv2.distanceTransform(focused_mask.astype(np.uint8),
                                             cv2.DIST_L2, 3)
        from scipy.ndimage import maximum_filter
        local_maxima = maximum_filter(dist_transform, size=5) == dist_transform
        local_maxima = local_maxima & (dist_transform > np.max(dist_transform) * 0.3)
        ridge_points = np.argwhere(local_maxima)
        if len(ridge_points) == 0:
            return 0
        angles = []
        for y, x in ridge_points:
            angle = np.arctan2(y - cy, x - cx)
            angles.append(angle)
        angle_bins = np.linspace(-np.pi, np.pi, 48)
        hist, _ = np.histogram(angles, bins=angle_bins)
        non_empty_bins = np.sum(hist > 0)
        estimated_spokes = int(non_empty_bins * 24 / 48)
        return max(0, min(30, estimated_spokes))
    except:
        return 0

def count_spokes_fourier(mask, cx, cy, radius):
    try:
        sample_radius = radius * 0.8
        n_samples = 360
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        samples = []
        for angle in angles:
            x = int(cx + sample_radius * np.cos(angle))
            y = int(cy + sample_radius * np.sin(angle))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                samples.append(mask[y, x] / 255.0)
            else:
                samples.append(0.0)
        samples = np.array(samples)
        fft_result = np.fft.fft(samples)
        power_spectrum = np.abs(fft_result)**2
        freqs = np.fft.fftfreq(n_samples)
        valid_indices = (np.abs(freqs) > 0.01) & (np.abs(freqs) < 0.5)
        if np.any(valid_indices):
            valid_power = power_spectrum[valid_indices]
            valid_freqs = freqs[valid_indices]
            peak_idx = np.argmax(valid_power)
            peak_freq = abs(valid_freqs[peak_idx])
            estimated_spokes = int(round(peak_freq * n_samples))
            if 10 <= estimated_spokes <= 40:
                return estimated_spokes
        return 0
    except:
        return 0

def count_spokes(mask, cx, cy, radius):
    spokes = 0
    for angle in np.linspace(0, 2*np.pi, 360, endpoint=False):
        prev_pixel = 0
        transitions = 0
        for r in np.linspace(0, radius, int(radius)):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                pixel = mask[y, x] > 0
                if pixel and not prev_pixel:
                    transitions += 1
                prev_pixel = pixel
        if transitions > 0:
            spokes += 1
    return spokes

def validate_flag(path_or_url, print_json=True):
    img_bgr, (w_in, h_in) = load_image(path_or_url)
    h, w, _ = img_bgr.shape
    report = {}
    report["aspect_ratio"] = check_aspect_ratio(w, h)
    bounds, color_report, stripe_report = check_stripes(img_bgr)
    report["colors"] = color_report
    report["stripe_proportion"] = stripe_report
    result = detect_chakra(img_bgr, bounds["white"])
    if result[0] is None:
        report["colors"]["chakra_blue"] = {"status": "fail", "deviation": "100%"}
        report["chakra_position"] = {"status": "fail", "offset_x": "N/A", "offset_y": "N/A"}
        report["chakra_spokes"] = {"status": "fail", "detected": 0}
    else:
        (offset_x, offset_y), dev_blue, spokes = result
        report["colors"]["chakra_blue"] = {
            "status": "pass" if dev_blue <= COLOR_TOL_PCT else "fail",
            "deviation": f"{dev_blue:.0f}%"
        }
        tol_x = w * CENTER_TOL_FRAC
        tol_y = h * CENTER_TOL_FRAC
        report["chakra_position"] = {
            "status": "pass" if abs(offset_x) <= tol_x and abs(offset_y) <= tol_y else "fail",
            "offset_x": f"{offset_x}px",
            "offset_y": f"{offset_y}px"
        }
        report["chakra_spokes"] = {
            "status": "pass" if abs(spokes - SPOKES_IDEAL) <= SPOKES_TOL else "fail",
            "detected": spokes
        }
    if print_json:
        return report

# Replace 'your_flag_image.png' with the path to the image you want to check
IMAGE_PATH = "your_flag_image.png"
plt.imshow(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB))
validate_flag(IMAGE_PATH)
