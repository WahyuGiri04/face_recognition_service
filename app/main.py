# app/main.py
import cv2
import numpy as np
import base64
import os
import requests
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from deepface import DeepFace
import logging
from dotenv import load_dotenv

# ---------- LOAD ENVIRONMENT VARIABLES ----------
# Get the root directory (parent of app folder)
ROOT_DIR = Path(__file__).parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load .env file from root directory
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Try to load from current directory as fallback
    load_dotenv()

# ---------- LOGGING SETUP ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ENVIRONMENT VARIABLES ----------
SERVICE_HOST = os.getenv("SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "5001"))
CONSUL_HOST = os.getenv("CONSUL_HOST", "127.0.0.1")
CONSUL_PORT = int(os.getenv("CONSUL_PORT", "8500"))

# Log loaded environment variables
logger.info(f"Service Configuration:")
logger.info(f"  HOST: {SERVICE_HOST}")
logger.info(f"  PORT: {SERVICE_PORT}")
logger.info(f"  CONSUL_HOST: {CONSUL_HOST}")
logger.info(f"  CONSUL_PORT: {CONSUL_PORT}")
logger.info(f"  ROOT_DIR: {ROOT_DIR}")
logger.info(f"  ENV_PATH: {ENV_PATH}")

# ---------- FASTAPI SETUP ----------
app = FastAPI(title="Face Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- BASE RESPONSE MODEL ----------
class BaseResponse(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

# ---------- REQUEST MODELS ----------
class ImageRequest(BaseModel):
    img1: str
    img2: str

class FaceDetectionRequest(BaseModel):
    image: str

# ---------- RESPONSE MODELS ----------
class FaceVerificationData(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str

class FaceDetectionData(BaseModel):
    has_face: bool
    face_count: int
    faces: List[Dict[str, int]]

# ---------- MODEL ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- CONSUL INTEGRATION ----------
class ConsulService:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        logger.info(f"Consul service initialized: {self.base_url}")
    
    def register_service(self):
        """Register service to Consul"""
        service_data = {
            "ID": "face-recognition-api",
            "Name": "face-recognition-api",
            "Tags": ["face-recognition", "api", "python"],
            "Address": SERVICE_HOST,
            "Port": SERVICE_PORT,
            "Check": {
                "HTTP": f"http://{SERVICE_HOST}:{SERVICE_PORT}/health",
                "Interval": "10s"
            }
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/v1/agent/service/register",
                json=service_data
            )
            if response.status_code == 200:
                logger.info("Service registered to Consul successfully")
            else:
                logger.error(f"Failed to register service: {response.status_code}")
        except Exception as e:
            logger.error(f"Error registering service to Consul: {e}")
    
    def deregister_service(self):
        """Deregister service from Consul"""
        try:
            response = requests.put(
                f"{self.base_url}/v1/agent/service/deregister/face-recognition-api"
            )
            if response.status_code == 200:
                logger.info("Service deregistered from Consul successfully")
            else:
                logger.error(f"Failed to deregister service: {response.status_code}")
        except Exception as e:
            logger.error(f"Error deregistering service from Consul: {e}")

consul_service = ConsulService(CONSUL_HOST, CONSUL_PORT)

# ---------- UTILITY FUNCTIONS ----------
def decode_base64_image(base64_string: str):
    """Decode base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Decode error: {e}")
        return None

def detect_faces(image: np.ndarray) -> tuple:
    """Detect faces in image using OpenCV"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        
        return len(faces) > 0, len(faces), face_list
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return False, 0, []

def create_response(code: int, message: str, data: Any = None) -> BaseResponse:
    """Create standardized response"""
    return BaseResponse(code=code, message=message, data=data)

# ---------- STARTUP/SHUTDOWN EVENTS ----------
@app.on_event("startup")
async def startup_event():
    """Register service to Consul on startup"""
    consul_service.register_service()

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister service from Consul on shutdown"""
    consul_service.deregister_service()

# ---------- HEALTH CHECK ENDPOINT ----------
@app.get("/health")
async def health_check():
    """Health check endpoint for Consul"""
    return {"status": "healthy", "service": "face-recognition-api"}

# ---------- FACE DETECTION ENDPOINT ----------
@app.post("/detect", response_model=BaseResponse)
async def detect_face(data: FaceDetectionRequest):
    """Detect faces in base64 image"""
    try:
        image = decode_base64_image(data.image)
        if image is None:
            return create_response(
                code=400,
                message="Failed to decode base64 image"
            )
        
        has_face, face_count, faces = detect_faces(image)
        
        detection_data = FaceDetectionData(
            has_face=has_face,
            face_count=face_count,
            faces=faces
        )
        
        return create_response(
            code=200,
            message="Face detection completed successfully",
            data=detection_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return create_response(
            code=500,
            message=f"Internal server error: {str(e)}"
        )

# ---------- FACE VERIFICATION ENDPOINT ----------
@app.post("/verify", response_model=BaseResponse)
async def verify_faces(data: ImageRequest):
    """Verify if two faces belong to the same person"""
    try:
        img1 = decode_base64_image(data.img1)
        img2 = decode_base64_image(data.img2)
        
        if img1 is None or img2 is None:
            return create_response(
                code=400,
                message="Failed to decode one or both base64 images"
            )
        
        # Check if both images contain faces
        has_face1, count1, _ = detect_faces(img1)
        has_face2, count2, _ = detect_faces(img2)
        
        if not has_face1:
            return create_response(
                code=400,
                message="No face detected in the first image"
            )
        
        if not has_face2:
            return create_response(
                code=400,
                message="No face detected in the second image"
            )
        
        # Perform face verification
        result = DeepFace.verify(
            img1, img2,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )
        
        verification_data = FaceVerificationData(
            verified=result["verified"],
            distance=result["distance"],
            threshold=result["threshold"],
            model=result["model"]
        )
        
        return create_response(
            code=200,
            message="Face verification completed successfully",
            data=verification_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return create_response(
            code=500,
            message=f"Internal server error: {str(e)}"
        )

# ---------- INFO ENDPOINT ----------
@app.get("/info", response_model=BaseResponse)
async def get_service_info():
    """Get service information"""
    info_data = {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "host": SERVICE_HOST,
        "port": SERVICE_PORT,
        "consul_host": CONSUL_HOST,
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/detect", "method": "POST", "description": "Detect faces in image"},
            {"path": "/verify", "method": "POST", "description": "Verify two faces"},
            {"path": "/info", "method": "GET", "description": "Service information"}
        ]
    }
    
    return create_response(
        code=200,
        message="Service information retrieved successfully",
        data=info_data
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)