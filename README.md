# AI Voice Classifier

A FastAPI-based audio classifier that predicts whether a voice recording is AI-generated or human spoken.

## Overview

This project exposes a REST API for audio classification. It accepts:
- `base64` audio in JSON
- remote audio URLs
- multipart file uploads

The backend extracts audio features using `librosa`, scales them, and predicts with a pre-trained Random Forest model.

## Features

- `/predict` for JSON requests with `audioBase64`, `audioFormat`, or `audio_url`
- `/upload` for multipart audio uploads
- `/health` to verify model/scaler availability
- `/docs` for interactive API documentation
- `.env` API key authentication

## Project structure

```
ai-voice-detection/
├── ai/                 # training and feature extraction code
│   ├── features.py
│   └── model.py
├── pickle_storage/     # saved model and scaler artifacts
├── utils/              # audio parsing, prediction, security helpers
│   ├── audio_parser.py
│   ├── predict.py
│   └── security.py
├── app.py              # FastAPI application entrypoint
├── core.py             # shared constants and helpers
├── .env                # environment variables (API key)
└── README.md           # this file
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install fastapi uvicorn numpy librosa scikit-learn joblib python-multipart soundfile numba python-dotenv requests
```

3. Ensure `.env` contains:

```text
API_KEY="your_api_key_here"
```

## Running the API

Start the server:

```bash
cd /run/media/an1rxdh/Drive D/Environment/voice-ai-detection
source venv/bin/activate
python app.py
```

Open the docs at `http://127.0.0.1:8000/docs`.

## API Endpoints

### `POST /predict`

Request body options:

- `audioBase64` with `audioFormat`
- `audio_url`

Example JSON:

```json
{
  "audioFormat": "wav",
  "audioBase64": "<base64 string>"
}
```

Response:

```json
{
  "status": "success",
  "language": "Unknown",
  "classification": "HUMAN",
  "confidenceScore": 0.63,
  "explanation": "Model confidence: 63.00%"
}
```

### `POST /upload`

Upload a file with form field `file`:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "x-api-key: $API_KEY" \
  -F "file=@path/to/audio.wav"
```

Response:

```json
{
  "prediction": "HUMAN",
  "confidence": "62.97%",
  "confidence_score": 0.6297,
  "probabilities": {
    "human": 0.6297,
    "ai": 0.3703
  }
}
```

### `GET /health`

Returns model readiness status.

## Training the model

If you need to retrain or rebuild the model, add audio files under `data/ai/` and `data/human/` and run:

```bash
python ai/model.py
```

This saves `model.pkl` and `scaler.pkl` into `pickle_storage/`.

## Notes

- The app loads `.env` automatically on startup.
- The API uses `x-api-key` for authentication.
- If port `8000` is in use, start the app on another port or stop the existing server.
