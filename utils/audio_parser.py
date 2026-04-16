from fastapi import UploadFile
from fastapi.responses import JSONResponse

import base64, tempfile, requests, os

from core import is_valid_url

def error_response(message, details=None, code=400):
    payload = {
        "status": "error",
        "message": message
    }
    if details:
        payload["details"] = details
    return JSONResponse(status_code=code, content=payload)


async def get_audio_from_request(body: dict, file: UploadFile = None):
    # If base64 audio
    if "audioBase64" in body:
        audio_b64 = body["audioBase64"]
        file_ext = (body.get("audioFormat") or body.get("audioformat") or "mp3").strip(".")
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return None, error_response("Invalid base64 audio", str(e), 400)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name, None

    # If audio url
    if "audio_url" in body:
        url = body["audio_url"]
        if not is_valid_url(url):
            return None, error_response("Invalid audio_url", code=400)
        
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
        
        except Exception as e:
            return None, error_response("Failed to download audio from the url", str(e), 400)
        
        content_type = res.headers.get("content-type", "")
        
        if "audio" not in content_type and "octet-stream" not in content_type:
            return None, error_response("URL did not return audio content", code=400)
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(res.content)
        tmp.close()
        return tmp.name, None
    
    # Multipart file upload
    if file:
        try:
            contents = await file.read()
            suffix = os.path.splitext(file.filename)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(contents)
            tmp.close()
            return tmp.name, None
        except Exception as e:
            return None, error_response("Failed to process uploaded file", str(e), 400)
        
    return None, error_response("No audioBase64, audio_url or file provided", code=400)