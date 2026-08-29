import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageQualityDataset(Dataset):
    """
    PyTorch Dataset for image quality classification.
    Loads and resizes BGR images, converting them to PyTorch tensors.
    """
    def __init__(self, dataset_info: list, split: str, size: tuple = (224, 224), transform=None):
        self.samples = [info for info in dataset_info if info["split"] == split]
        self.size = size
        self.transform = transform
        
        # Categorical labels mapping
        self.classes = ["acceptable", "blurry", "underexposed", "overexposed", "noisy", "corrupted", "defective"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["path"]
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        img = cv2.resize(img, self.size)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Simple standardization
        # Mean/std for ImageNet or standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        label_idx = self.class_to_idx[sample["category"]]
        
        return img_tensor, label_idx, img_path

class QualityCNN(nn.Module):
    """
    Lightweight, self-contained custom CNN for image quality analysis.
    Avoids external weights dependency and trains fast on CPU.
    """
    def __init__(self, num_classes: int = 7):
        super(QualityCNN, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Conv block 4 (Target for Grad-CAM)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classifier
        # Input size: 128 channels * (224 / 2^4 = 14) * 14 = 128 * 14 * 14 = 25088
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GradCAM:
    """
    Grad-CAM tool to generate heatmaps for model explainability.
    Targets the final convolutional layer of the model.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates raw saliency heatmap values for input image tensor.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward()
        
        # Get gradient and activation
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted sum of feature maps
        cam = torch.sum(weights * activations, dim=0)  # [H, W]
        
        # Apply ReLU (keep positive influences)
        cam = F.relu(cam)
        
        # Normalize cam to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_heatmap_on_image(img_path: str, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays heatmaps from Grad-CAM onto the original image using OpenCV JET colormap.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image for Grad-CAM overlay: {img_path}")
        
    h, w, c = img.shape
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to 8-bit [0, 255]
    heatmap_255 = np.uint8(255 * heatmap_resized)
    
    # Apply Jet color map
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    
    # Blend image and heatmap
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay

def train_cnn_model(dataset_info: list, model_output_path: str = "cnn_model.pth", epochs: int = 12, batch_size: int = 16) -> dict:
    """
    Trains and saves the lightweight CNN quality classifier.
    """
    train_dataset = ImageQualityDataset(dataset_info, split="train")
    val_dataset = ImageQualityDataset(dataset_info, split="val")
    test_dataset = ImageQualityDataset(dataset_info, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = QualityCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"Training CNN on {device}...")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            torch.save(model.state_dict(), model_output_path)
            
    # Load best model for testing
    model.load_state_dict(torch.load(model_output_path))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    report = classification_report(test_targets, test_preds, target_names=train_dataset.classes, output_dict=True)
    conf_mat = confusion_matrix(test_targets, test_preds)
    
    print("\n--- CNN Deep Learning Model Evaluation ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return {
        "accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "classes": train_dataset.classes
    }
