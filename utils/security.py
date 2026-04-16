from fastapi import HTTPException, Header
import os

async def verify_api_key(x_api_key: str = Header(None)):
    valid_key = os.getenv("API_KEY")
    if not valid_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if x_api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True