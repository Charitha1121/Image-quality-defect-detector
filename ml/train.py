import os
import json
from ml.dataset import generate_synthetic_dataset
from ml.classical import train_classical_model
from ml.deep_learning import train_cnn_model

def run_pipeline():
    print("--- Starting AI-Powered Image Quality & Defect Detection Training Pipeline ---")
    
    # 1. Dataset Generation
    print("\n[Step 1/3] Generating synthetic degradations and dataset splits...")
    dataset_info = generate_synthetic_dataset("data_synthetic", num_base_images=50)
    
    # 2. Train Classical Model
    print("\n[Step 2/3] Extracting features and training Classical Random Forest model...")
    classical_model_path = os.path.join("backend", "models", "classical_rf.joblib")
    classical_results = train_classical_model(dataset_info, model_output_path=classical_model_path)
    
    # 3. Train Deep Learning CNN Model
    print("\n[Step 3/3] Training deep learning CNN branch for defect/anomaly detection...")
    cnn_model_path = os.path.join("backend", "models", "cnn_model.pth")
    cnn_results = train_cnn_model(dataset_info, model_output_path=cnn_model_path, epochs=12, batch_size=16)
    
    # Save training metrics to metrics.json
    metrics = {
        "classical": {
            "accuracy": classical_results["accuracy"],
            "classification_report": classical_results["classification_report"],
            "confusion_matrix": classical_results["confusion_matrix"],
            "feature_importances": classical_results["feature_importances"][:15]  # Top 15 features
        },
        "cnn": {
            "accuracy": cnn_results["accuracy"],
            "classification_report": cnn_results["classification_report"],
            "confusion_matrix": cnn_results["confusion_matrix"]
        }
    }
    
    metrics_path = os.path.join("backend", "models", "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Pipeline Completed Successfully! ---")
    print(f"Classical RF saved to: {classical_model_path}")
    print(f"Deep CNN saved to: {cnn_model_path}")
    print(f"Pipeline training metrics logged to: {metrics_path}")

if __name__ == "__main__":
    run_pipeline()
