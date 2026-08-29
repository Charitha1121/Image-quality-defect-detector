import cv2
import numpy as np

def extract_features(img_path: str) -> dict:
    """
    Extracts raw numerical features from an image to analyze its quality.
    
    Features:
    1. Blur: Laplacian variance (higher = sharper, lower = blurrier).
    2. Exposure: Mean, std, and a simplified 10-bin histogram of pixel brightness.
    3. Contrast: Root-Mean-Square (RMS) contrast.
    4. Noise: Standard deviation of difference between image and its median-filtered version.
    5. FFT Energy: High-frequency energy ratio in the frequency domain.
    """
    # Read image in grayscale and color
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Sharpness/Blur (Laplacian Variance)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    
    # 2. Exposure/Brightness
    brightness_mean = float(np.mean(img_gray))
    brightness_std = float(np.std(img_gray))
    
    # Normalized histogram (10 bins)
    hist = cv2.calcHist([img_gray], [0], None, [10], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_features = {f"hist_bin_{i}": float(val) for i, val in enumerate(hist)}
    
    # 3. Contrast (RMS contrast)
    contrast_rms = float(np.std(img_gray))
    
    # 4. Noise estimation (Difference between image and median filtered version)
    # This filters out small noise and compares it with the original to isolate noise variance.
    median_filtered = cv2.medianBlur(img_gray, 3)
    noise_diff = cv2.absdiff(img_gray, median_filtered)
    noise_std = float(np.std(noise_diff))
    
    # 5. FFT spectral energy ratio (High-frequency ratio)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    
    dft = np.fft.fft2(img_gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    
    # Create mask to separate low and high frequencies
    # Low-frequency circle radius
    r = min(rows, cols) // 10  # 10% radius
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r, 1, -1)
    
    low_freq_energy = np.sum(magnitude_spectrum * mask)
    total_energy = np.sum(magnitude_spectrum)
    
    fft_high_freq_ratio = float((total_energy - low_freq_energy) / (total_energy + 1e-8))
    
    # Combine all features
    features = {
        "blur_laplacian_var": blur_score,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast_rms": contrast_rms,
        "noise_std": noise_std,
        "fft_high_freq_ratio": fft_high_freq_ratio
    }
    features.update(hist_features)
    
    return features
