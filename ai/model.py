import joblib, os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ai.features import extract_features

AI_DATA = "./data/ai/"
HUMAN_DATA = "./data/human/"

def load_dataset():
    X = []
    y = []

    # AI voices → label = 1
    for f in os.listdir(AI_DATA):
        if f.endswith(".wav") or f.endswith(".mp3"):
            path = os.path.join(AI_DATA, f)
            feat = extract_features(path)
            X.append(feat)
            y.append(1)

    # Human voices → label = 0
    for f in os.listdir(HUMAN_DATA):
        if f.endswith(".wav") or f.endswith(".mp3"):
            path = os.path.join(HUMAN_DATA, f)
            feat = extract_features(path)
            X.append(feat)
            y.append(0)

    return np.array(X), np.array(y)

def train_model():

    X, y_labels = load_dataset()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print(f"\nTotal training samples: {len(X)} (AI: {y_labels.count(1)}, Human: {y_labels.count(0)})")
    print(f"\nTotal training samples: {len(X)} (AI: {int(np.sum(y_labels == 1))}, Human: {int(np.sum(y_labels == 0))})")

    # Train model optimized for small datasets
    model = RandomForestClassifier( # 0 - no max depth and 1 lowest depth
        n_estimators=50,
        max_depth=5, # low value - underfitting, high value - overfitting
        min_samples_split=2, # high value - underfitting, low value - overfitting
        min_samples_leaf=1, # high value - underfitting, low value - overfitting
        random_state=42
    )
    model.fit(X_scaled, y_labels)

    # Save trained model + scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "./pickle_storage/model.pkl")
    joblib.dump(scaler, "./pickle_storage/scaler.pkl")
    

    print("Training complete. Files saved in /pickle_storage/")


    train_score = model.score(X_scaled, y_labels)
    print(f"Model Accuracy: {train_score*100:.1f}%")
    print(f"Model is ready for predictions!\n")

    return model, scaler

if __name__ == "__main__":
    train_model()