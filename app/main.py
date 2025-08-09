import base64
import io
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from consul import Consul, Check
import uvicorn
from PIL import Image
from deepface import DeepFace
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="FaceService")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Response Model
# -----------------------------
class BaseResponse(BaseModel):
    status_code: int
    message: str
    data: Any

# -----------------------------
# Helpers
# -----------------------------
def decode_base64(data: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(data)))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image")

def check_face_in_image(data: str) -> bool:
    img = decode_base64(data)
    try:
        result = DeepFace.detectFace(img)
        return True
    except ValueError:
        return False

def verify_faces(img1: str, img2: str) -> Dict[str, Any]:
    try:
        result = DeepFace.verify(img1, img2, model_name="ArcFace")
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/verify-face", response_model=BaseResponse)
def verify_face(payload: Dict[str, str]):
    img1 = payload.get("image1")
    img2 = payload.get("image2")
    if not img1 or not img2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing image1 or image2")

    if not check_face_in_image(img1):
        return BaseResponse(status_code=status.HTTP_400_BAD_REQUEST, message="No face detected in image1", data=None)
    if not check_face_in_image(img2):
        return BaseResponse(status_code=status.HTTP_400_BAD_REQUEST, message="No face detected in image2", data=None)

    result = verify_faces(img1, img2)
    return BaseResponse(status_code=status.HTTP_200_OK, message="Verification result", data=result)

@app.post("/detect-face", response_model=BaseResponse)
def detect_face(payload: Dict[str, str]):
    img = payload.get("image")
    if not img:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing image")

    face_detected = check_face_in_image(img)
    return BaseResponse(status_code=status.HTTP_200_OK, message="Detection result", data={"face_detected": face_detected})

# -----------------------------
# Consul registration
# -----------------------------
def register_to_consul():
    c = Consul(host=os.getenv("CONSUL_HOST", "127.0.0.1"), port=8500)
    c.agent.service.register(
        name="face-service",
        service_id="face-service-1",
        address=os.getenv("SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVICE_PORT", 5001)),
        check=Check.http(
            f"http://{os.getenv('SERVICE_HOST', '127.0.0.1')}:{os.getenv('SERVICE_PORT', 5001)}/docs",
            interval="10s"
        ),
    )

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    register_to_consul()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("SERVICE_PORT", 5001)),
        reload=False
    )