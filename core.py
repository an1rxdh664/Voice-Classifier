from urllib.parse import urlparse
from fastapi import Request

MODEL_PATH = "./pickle_storage/model.pkl"
SCALER_PATH = "./pickle_storage/scaler.pkl"

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

async def get_request_body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}