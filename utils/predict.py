import numpy as np
import joblib

from ai.features import extract_features
from core import MODEL_PATH, SCALER_PATH


_model = None
_scaler = None

def load_model_and_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError("Model Load Failed") from e
        
        try:
            _scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            raise RuntimeError("Scaler Load Failed") from e
        
    return _model, _scaler

def predict_from_file(file_path):

    model, scaler = load_model_and_scaler()

    feat = extract_features(file_path)
    feat_arr = np.array(feat).reshape(1, -1)
    feat_scaled = scaler.transform(feat_arr)

    prediction = model.predict(feat_scaled)[0]             # 0 or 1
    probability = model.predict_proba(feat_scaled)[0]      # [prob_human, prob_ai]

    prediction_label = "AI_GENERATED" if prediction == 1 else "HUMAN"
    confidence = float(probability[prediction])                  # prob of chosen label

    return {
        "label": prediction_label,
        "prediction_index": int(prediction),
        "probabilities": {
            "human": float(probability[0]),
            "ai": float(probability[1])
        },
        "confidence": confidence
    }
