import os
import cv2
import numpy as np
import torch
import joblib
from backend.app.models import AnalysisResult
from ml.features import extract_features
from ml.deep_learning import QualityCNN, GradCAM, overlay_heatmap_on_image, device

# Paths to serialized models
CLASSICAL_MODEL_PATH = os.getenv("CLASSICAL_MODEL_PATH", "backend/models/classical_rf.joblib")
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "backend/models/cnn_model.pth")

# Global variables for models
classical_model_data = None
cnn_model = None
grad_cam_generator = None

# Create folders for uploads and heatmaps
UPLOAD_DIR = "static/uploads"
HEATMAP_DIR = "static/heatmaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

def load_models():
    """
    Loads both classical and CNN models from disk.
    If models do not exist, runs in Heuristic/Fallback mode.
    """
    global classical_model_data, cnn_model, grad_cam_generator
    
    # 1. Load Classical Model
    if os.path.exists(CLASSICAL_MODEL_PATH):
        try:
            classical_model_data = joblib.load(CLASSICAL_MODEL_PATH)
            print(f"Loaded Classical Model from {CLASSICAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading classical model: {e}")
            classical_model_data = None
    else:
        print(f"Classical model not found at {CLASSICAL_MODEL_PATH}. Using heuristic features.")
        
    # 2. Load PyTorch CNN Model
    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_model = QualityCNN(num_classes=7)
            cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
            cnn_model.to(device)
            cnn_model.eval()
            
            # Setup Grad-CAM on conv4 layer
            grad_cam_generator = GradCAM(cnn_model, cnn_model.conv4)
            print(f"Loaded CNN Model and registered Grad-CAM from {CNN_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            cnn_model = None
            grad_cam_generator = None
    else:
        print(f"CNN model not found at {CNN_MODEL_PATH}. Using heuristic defect detection.")

# Initial load attempts
load_models()

def detect_defects_heuristically(img_path: str) -> list:
    """
    Fallback classical computer vision defect detection.
    Detects scratches using Hough lines and spots using contours.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    issues = []
    
    # 1. Scratch Detection using Canny + HoughLinesP
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=5)
    if lines is not None and len(lines) > 0:
        confidence = min(0.5 + 0.1 * len(lines), 0.85)
        issues.append({
            "type": "defect",
            "severity": "scratch",
            "confidence": float(confidence)
        })
        
    # 2. Spot / Dust detection (thresholding + contours)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    small_dark_spots = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 200:
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:
                    small_dark_spots += 1
                    
    if small_dark_spots > 2:
        confidence = min(0.4 + 0.1 * small_dark_spots, 0.75)
        issues.append({
            "type": "defect",
            "severity": "spot",
            "confidence": float(confidence)
        })
        
    return issues

def preprocess_for_cnn(img_path: str) -> torch.Tensor:
    """
    Loads and transforms an image to feed into the PyTorch CNN.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0).to(device)  # Add batch dimension

def analyze_image_pipeline(img_path: str, filename: str) -> dict:
    """
    Core pipeline running feature engineering, classical ML inference, CNN inference, 
    fuses the predictions, and generates Grad-CAM overlays.
    """
    # 1. Feature Engineering
    features = extract_features(img_path)
    
    # Extract values for readability
    blur_val = features["blur_laplacian_var"]
    mean_brightness = features["brightness_mean"]
    noise_val = features["noise_std"]
    contrast_val = features["contrast_rms"]
    fft_val = features["fft_high_freq_ratio"]
    
    issues = []
    
    # 2. Classical Heuristics & Decision Boundaries (Hard boundaries are highly reliable)
    # Blur
    blur_confidence = 0.0
    if blur_val < 150:
        # Scale confidence based on severity
        blur_confidence = float(min(1.0, max(0.1, (150 - blur_val) / 130)))
        issues.append({
            "type": "blur",
            "severity": "high" if blur_val < 50 else "low",
            "confidence": blur_confidence
        })
        
    # Exposure (Underexposure)
    underexposure_confidence = 0.0
    if mean_brightness < 60:
        underexposure_confidence = float(min(1.0, max(0.1, (60 - mean_brightness) / 50)))
        issues.append({
            "type": "underexposure",
            "severity": "high" if mean_brightness < 30 else "low",
            "confidence": underexposure_confidence
        })
        
    # Exposure (Overexposure)
    overexposure_confidence = 0.0
    if mean_brightness > 195:
        overexposure_confidence = float(min(1.0, max(0.1, (mean_brightness - 195) / 50)))
        issues.append({
            "type": "overexposure",
            "severity": "high" if mean_brightness > 230 else "low",
            "confidence": overexposure_confidence
        })
        
    # Noise
    noise_confidence = 0.0
    if noise_val > 8.0:
        noise_confidence = float(min(1.0, max(0.1, (noise_val - 8.0) / 30.0)))
        issues.append({
            "type": "noise",
            "severity": "high" if noise_val > 25.0 else "low",
            "confidence": noise_confidence
        })
        
    # 3. Model Predictions (if available)
    classical_pred = None
    cnn_pred_class = None
    cnn_probs = None
    
    # Classical inference
    if classical_model_data is not None:
        try:
            model = classical_model_data["model"]
            feat_names = classical_model_data["feature_names"]
            feat_vector = [[features[k] for k in feat_names]]
            classical_pred = model.predict(feat_vector)[0]
        except Exception as e:
            print(f"Error running classical inference: {e}")
            
    # CNN inference and Grad-CAM Saliency Map
    heatmap_filename = None
    heatmap_path = None
    
    if cnn_model is not None:
        try:
            input_tensor = preprocess_for_cnn(img_path)
            outputs = cnn_model(input_tensor)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
            cnn_probs = probs.tolist()
            
            classes = ["acceptable", "blurry", "underexposed", "overexposed", "noisy", "corrupted", "defective"]
            cnn_pred_idx = probs.argmax()
            cnn_pred_class = classes[cnn_pred_idx]
            
            # If CNN detects defect or corruption with high confidence, append to issues
            if cnn_pred_class == "defective" and probs[cnn_pred_idx] > 0.35:
                # Add defect issue
                issues.append({
                    "type": "defect",
                    "severity": "high" if probs[cnn_pred_idx] > 0.7 else "low",
                    "confidence": float(probs[cnn_pred_idx])
                })
            elif cnn_pred_class == "corrupted" and probs[cnn_pred_idx] > 0.35:
                issues.append({
                    "type": "corruption",
                    "severity": "high" if probs[cnn_pred_idx] > 0.7 else "low",
                    "confidence": float(probs[cnn_pred_idx])
                })
                
            # Generate Grad-CAM Heatmap
            if grad_cam_generator is not None:
                heatmap = grad_cam_generator.generate_heatmap(input_tensor, cnn_pred_idx)
                overlay = overlay_heatmap_on_image(img_path, heatmap)
                heatmap_filename = f"heatmap_{os.path.basename(img_path)}"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                cv2.imwrite(heatmap_path, overlay)
        except Exception as e:
            print(f"Error running CNN inference/Grad-CAM: {e}")
    else:
        # Fallback to classical defect detection (Hough lines / spots)
        heuristic_defects = detect_defects_heuristically(img_path)
        issues.extend(heuristic_defects)
        
        # If fallback, we create a pseudo Grad-CAM heatmap highlighting detected lines/spots
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # Create a blank overlay
                overlay = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                # Dilate edges and color them red
                kernel = np.ones((5,5), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=1)
                overlay[dilated > 0] = [0, 0, 255] # Red highlight for anomalies
                
                blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
                heatmap_filename = f"heatmap_{os.path.basename(img_path)}"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                cv2.imwrite(heatmap_path, blended)
        except Exception as e:
            print(f"Error generating fallback heatmap: {e}")

    # 4. Score Fusion & Decision Logic
    # Start with base score of 100
    score = 100.0
    
    # Calculate deductions
    # Deduct for blur (up to 40 pts)
    if blur_val < 150:
        blur_deduct = min(40.0, (150 - blur_val) * 0.35)
        score -= blur_deduct
        
    # Deduct for exposure (up to 50 pts)
    if mean_brightness < 60:
        score -= min(50.0, (60 - mean_brightness) * 1.2)
    elif mean_brightness > 195:
        score -= min(50.0, (mean_brightness - 195) * 1.2)
        
    # Deduct for noise (up to 40 pts)
    if noise_val > 8.0:
        score -= min(40.0, (noise_val - 8.0) * 1.6)
        
    # Deduct for defects (up to 60 pts)
    defect_issues = [iss for iss in issues if iss["type"] == "defect"]
    if defect_issues:
        max_defect_conf = max(iss["confidence"] for iss in defect_issues)
        score -= min(60.0, max_defect_conf * 60.0)
        
    # Deduct for corruption (up to 50 pts)
    corruption_issues = [iss for iss in issues if iss["type"] == "corruption"]
    if corruption_issues:
        max_corr_conf = max(iss["confidence"] for iss in corruption_issues)
        score -= min(50.0, max_corr_conf * 50.0)
        
    # Ensure score bounds
    score = float(np.clip(score, 0, 100))
    
    # Determine final quality label
    if score >= 80 and len(issues) == 0:
        quality_label = "ACCEPTABLE"
    elif score >= 50 and not any(iss["type"] == "defect" for iss in issues):
        quality_label = "DEGRADED"
    else:
        quality_label = "DEFECTIVE"
        
    # Format issues to ensure unique entries
    unique_issues = {}
    for iss in issues:
        key = iss["type"]
        if key not in unique_issues or iss["confidence"] > unique_issues[key]["confidence"]:
            unique_issues[key] = iss
            
    final_issues_list = list(unique_issues.values())
    
    return {
        "quality_score": round(score, 1),
        "quality_label": quality_label,
        "issues": final_issues_list,
        "features": features,
        "heatmap_relative_path": f"/static/heatmaps/{heatmap_filename}" if heatmap_filename else None,
        "original_relative_path": f"/static/uploads/{os.path.basename(img_path)}"
    }
