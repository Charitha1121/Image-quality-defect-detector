import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ml.features import extract_features

def build_dataset_matrix(dataset_info: list) -> tuple:
    """
    Given lists of image paths and classes, extracts features for each and
    returns matrices ready for scikit-learn training.
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # Store feature names to check importance later
    feature_names = None
    
    for idx, info in enumerate(dataset_info):
        try:
            feats = extract_features(info["path"])
            if feature_names is None:
                feature_names = list(feats.keys())
            
            feat_vector = [feats[k] for k in feature_names]
            
            if info["split"] == "train":
                X_train.append(feat_vector)
                y_train.append(info["category"])
            elif info["split"] == "val":
                X_val.append(feat_vector)
                y_val.append(info["category"])
            elif info["split"] == "test":
                X_test.append(feat_vector)
                y_test.append(info["category"])
        except Exception as e:
            print(f"Error processing {info['path']}: {e}")
            
    return (
        np.array(X_train), np.array(y_train),
        np.array(X_val), np.array(y_val),
        np.array(X_test), np.array(y_test),
        feature_names
    )

def train_classical_model(dataset_info: list, model_output_path: str = "classical_rf.joblib") -> dict:
    """
    Trains a Random Forest Classifier on engineered features.
    """
    print("Extracting features for classical ML model...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = build_dataset_matrix(dataset_info)
    
    print(f"Training shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Feature importances
    importances = rf.feature_importances_
    feat_importances = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n--- Classical Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nFeature Importances:")
    for name, imp in feat_importances[:10]:
        print(f"  {name}: {imp:.4f}")
        
    # Serialize model and metadata
    model_data = {
        "model": rf,
        "feature_names": feature_names,
        "classes": rf.classes_.tolist()
    }
    
    # Ensure folder path exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_data, model_output_path)
    print(f"Model saved to {model_output_path}")
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "classes": rf.classes_.tolist(),
        "feature_importances": feat_importances
    }
