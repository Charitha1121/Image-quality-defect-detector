import os
import random
import cv2
import numpy as np

def generate_procedural_image(image_id: int, size: tuple = (224, 224)) -> np.ndarray:
    """
    Generates a high-quality clean base image procedurally based on an ID.
    This ensures varied content (patterns, gradients, textures, shapes) without external dependencies.
    Returns a BGR image as a numpy array.
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Use image_id to seed local generation for consistency
    rng = np.random.default_rng(image_id)
    
    pattern_type = image_id % 5
    
    if pattern_type == 0:
        # Concentric circles and radial lines (Geometric calibration pattern)
        img.fill(240)  # Off-white background
        color = (20, 20, 20)
        cv2.circle(img, (w // 2, h // 2), min(h, w) // 3, color, 3)
        cv2.circle(img, (w // 2, h // 2), min(h, w) // 6, color, 2)
        cv2.circle(img, (w // 2, h // 2), min(h, w) // 12, color, 1)
        # Draw spokes
        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            x_end = int(w // 2 + (w // 3) * np.cos(rad))
            y_end = int(h // 2 + (h // 3) * np.sin(rad))
            cv2.line(img, (w // 2, h // 2), (x_end, y_end), color, 2)
            
    elif pattern_type == 1:
        # Checkerboard / Grid pattern
        img.fill(255)
        grid_size = rng.choice([16, 28, 32])
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                    # Random pastel color
                    c = (int(rng.integers(100, 220)), int(rng.integers(100, 220)), int(rng.integers(100, 220)))
                    cv2.rectangle(img, (x, y), (min(x + grid_size, w), min(y + grid_size, h)), c, -1)
                    
    elif pattern_type == 2:
        # Smooth multi-directional color gradient
        for y in range(h):
            for x in range(w):
                r = int(127 + 127 * np.sin(x / 30.0))
                g = int(127 + 127 * np.cos(y / 30.0))
                b = int(127 + 127 * np.sin((x + y) / 40.0))
                img[y, x] = [b, g, r]
                
    elif pattern_type == 3:
        # Composite sinusoids simulating a natural background
        for y in range(h):
            for x in range(w):
                val = 127 + 60 * np.sin(x / 15.0) * np.cos(y / 15.0) + 30 * np.sin(x / 5.0) + 30 * np.cos(y / 8.0)
                # Keep within bounds
                val = max(0, min(255, val))
                # Soft brown/green hue
                img[y, x] = [int(val * 0.8), int(val * 0.9), int(val)]
                
    elif pattern_type == 4:
        # Random colored overlapping polygons and shapes
        img.fill(235)
        for _ in range(12):
            shape_type = rng.choice(["rect", "circle", "triangle"])
            color = (int(rng.integers(50, 200)), int(rng.integers(50, 200)), int(rng.integers(50, 200)))
            if shape_type == "rect":
                x1, y1 = rng.integers(0, w - 40), rng.integers(0, h - 40)
                x2, y2 = x1 + rng.integers(20, 80), y1 + rng.integers(20, 80)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            elif shape_type == "circle":
                cx, cy = rng.integers(20, w - 20), rng.integers(20, h - 20)
                radius = rng.integers(15, 45)
                cv2.circle(img, (cx, cy), radius, color, -1)
            elif shape_type == "triangle":
                pts = rng.integers(0, w, size=(3, 2))
                cv2.drawContours(img, [pts.astype(np.int32)], 0, color, -1)
                
    # Add a bit of fine edge detail to all base images for realistic quality scoring
    cv2.putText(img, f"BASE-IMG {image_id}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    
    # Soft bilateral filter to ensure clean base image has no compression or sampling noise
    img = cv2.bilateralFilter(img, 5, 50, 50)
    return img

def apply_blur(img: np.ndarray, severity: str) -> np.ndarray:
    ksize = 7 if severity == "low" else 15
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_underexposure(img: np.ndarray, severity: str) -> np.ndarray:
    factor = 0.4 if severity == "low" else 0.15
    # Convert to float, multiply, clip, convert back
    return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)

def apply_overexposure(img: np.ndarray, severity: str) -> np.ndarray:
    factor = 1.6 if severity == "low" else 2.5
    return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)

def apply_noise(img: np.ndarray, severity: str) -> np.ndarray:
    std = 15 if severity == "low" else 40
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_corruption(img: np.ndarray, severity: str) -> np.ndarray:
    quality = 25 if severity == "low" else 5
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if result:
        return cv2.imdecode(encimg, 1)
    return img

def apply_defect(img: np.ndarray, defect_type: str, seed: int) -> np.ndarray:
    """
    Simulates physical anomalies (scratches, dark spot contamination, vertical sensor lines).
    """
    rng = np.random.default_rng(seed)
    h, w, c = img.shape
    defect_img = img.copy()
    
    if defect_type == "scratch":
        # Draw a thin, bright or dark scratch line across the image
        x1, y1 = rng.integers(10, w - 10), rng.integers(10, h - 10)
        x2 = int(np.clip(x1 + rng.integers(-80, 80), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(-80, 80), 0, h - 1))
        color = (220, 220, 220) if rng.random() > 0.5 else (20, 20, 20)
        thickness = rng.choice([1, 2])
        cv2.line(defect_img, (x1, y1), (x2, y2), color, thickness)
        
    elif defect_type == "spot":
        # Draw a dark smudge or spot (e.g. dust on sensor)
        cx, cy = rng.integers(20, w - 20), rng.integers(20, h - 20)
        rx, ry = rng.integers(4, 12), rng.integers(4, 12)
        angle = rng.integers(0, 180)
        # Create transparency overlay for soft smudge
        overlay = defect_img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, (15, 15, 15), -1)
        # Blend
        cv2.addWeighted(overlay, 0.6, defect_img, 0.4, 0, defect_img)
        
    elif defect_type == "sensor_line":
        # Vertical sensor pixel column failure
        col = rng.integers(20, w - 20)
        color = (0, 0, 255) if rng.random() > 0.5 else (0, 255, 0)  # Red or Green hot pixel column
        cv2.line(defect_img, (col, 0), (col, h), color, 1)
        
    return defect_img

def generate_synthetic_dataset(output_dir: str, num_base_images: int = 50) -> dict:
    """
    Generates a full synthetic dataset with controlled label/severities and writes to disk.
    Splits images into train/val/test splits grouped by base image to prevent data leakage.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_info = []
    
    # Define splits by base image IDs
    # 70% train (0-34), 15% val (35-41), 15% test (42-49)
    for base_id in range(num_base_images):
        if base_id < int(num_base_images * 0.7):
            split = "train"
        elif base_id < int(num_base_images * 0.85):
            split = "val"
        else:
            split = "test"
            
        # 1. Base image (ACCEPTABLE)
        base_img = generate_procedural_image(base_id)
        
        # We will create directories for classes inside split
        # Classes: acceptable, blurry, underexposed, overexposed, noisy, corrupted, defective
        categories = {
            "acceptable": [base_img.copy()],
            "blurry": [apply_blur(base_img, "low"), apply_blur(base_img, "high")],
            "underexposed": [apply_underexposure(base_img, "low"), apply_underexposure(base_img, "high")],
            "overexposed": [apply_overexposure(base_img, "low"), apply_overexposure(base_img, "high")],
            "noisy": [apply_noise(base_img, "low"), apply_noise(base_img, "high")],
            "corrupted": [apply_corruption(base_img, "low"), apply_corruption(base_img, "high")],
            "defective": [
                apply_defect(base_img, "scratch", base_id * 10),
                apply_defect(base_img, "spot", base_id * 20),
                apply_defect(base_img, "sensor_line", base_id * 30)
            ]
        }
        
        for category, img_list in categories.items():
            class_dir = os.path.join(output_dir, split, category)
            os.makedirs(class_dir, exist_ok=True)
            
            for idx, img in enumerate(img_list):
                img_name = f"base_{base_id}_copy_{idx}.png"
                img_path = os.path.join(class_dir, img_name)
                cv2.imwrite(img_path, img)
                
                # Determine severity and label
                severity = "none" if category == "acceptable" else ("low" if idx == 0 else "high")
                # Defective items have varying types
                if category == "defective":
                    defect_types = ["scratch", "spot", "sensor_line"]
                    severity = defect_types[idx]
                    
                images_info.append({
                    "path": img_path,
                    "split": split,
                    "category": category,
                    "base_id": base_id,
                    "severity": severity
                })
                
    print(f"Generated synthetic dataset with {len(images_info)} images in {output_dir}")
    return images_info
if __name__ == "__main__":
    generate_synthetic_dataset("data_synthetic", num_base_images=50)
