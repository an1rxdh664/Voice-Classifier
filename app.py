from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from dotenv import load_dotenv

import uvicorn, os

from utils.predict import predict_from_file, load_model_and_scaler
from utils.audio_parser import get_audio_from_request, error_response
from utils.security import verify_api_key
from core import get_request_body

load_dotenv()

app = FastAPI(
    title = "AI Voice Classifier",
    description = "",
    version = "1.0"
)


@app.get("/")
def info():
    return {
        "name": "API Based AI Voice Classifier",
        "version": "1.0",
        "description": "Detects if a voice is AI-generated or human",
        "features": {
            "MFCC": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "ZCR": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "Spectral Centroid": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "RMS": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "Pitch": ["Mean", "Std Dev", "Variance", "Mean Difference"]
        },
        "total_features": 20,
        "model": "Random Forest Classifier (50 trees)",
        "endpoints": {
            "predict": "/predict",
            "upload": "/upload",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    model_ready = True
    scaler_ready = True
    try:
        load_model_and_scaler()
    except Exception:
        model_ready = False
        scaler_ready = False

    return {
        "status": "ok",
        "model_loaded": model_ready,
        "scaler_loaded": scaler_ready
    }


@app.post("/predict")
async def predict(request: Request, _verify: bool = Depends(verify_api_key)):
    body = await get_request_body(request)
    temp_path, err = await get_audio_from_request(body)

    if err:
        return err

    try:
        result = predict_from_file(temp_path)

        classification = "AI_GENERATED" if result["label"] == "AI_GENERATED" else "HUMAN"
        confidence = result["confidence"]

        return {
            "status": "success",
            "language": body.get("language", "Unknown"),
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": f"Model confidence: {confidence * 100:.2f}%"
        }

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Prediction failed", str(e), 500)
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/upload")
async def predict_upload(
    request: Request,
    file: UploadFile = File(...),
    _verify: bool = Depends(verify_api_key)
):
    body = await get_request_body(request)
    temp_path, err = await get_audio_from_request(body, file)

    if err:
        return err
    
    try:

        result = predict_from_file(temp_path)

        return {
            "prediction": result["label"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "confidence_score": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:
        return error_response("Prediction failed", str(e), 500)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)