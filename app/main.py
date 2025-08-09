# backend/main.py
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from deepface import DeepFace

# ---------- FASTAPI SETUP ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODEL ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- UTIL ----------
def decode_base64_image(base64_string: str):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Decode error:", e)
        return None

# ---------- SCHEMA ----------
class ImageRequest(BaseModel):
    img1: str
    img2: str

# ---------- ENDPOINT VERIFY ----------
@app.post("/verify")
async def verify_faces(data: ImageRequest):
    img1 = decode_base64_image(data.img1)
    img2 = decode_base64_image(data.img2)
    if img1 is None or img2 is None:
        raise HTTPException(400, "Decode error")
    try:
        res = DeepFace.verify(img1, img2, model_name="ArcFace",
                              detector_backend="opencv",
                              enforce_detection=False, silent=True)
        return {"verified": res["verified"], "distance": res["distance"],
                "threshold": res["threshold"], "model": res["model"]}
    except Exception as e:
        raise HTTPException(500, str(e))
