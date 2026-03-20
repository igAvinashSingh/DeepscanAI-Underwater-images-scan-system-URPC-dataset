#!/usr/bin/env python3
"""
DeepScan AI — URPC2020 Underwater Object Detection Backend
YOLOv8 + FastAPI + Claude AI Integration

URPC2020 Classes:
  0: holothurian (sea cucumber)
  1: echinus (sea urchin)
  2: scallop
  3: starfish
  4: waterweeds
"""

import os
import io
import base64
import time
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import google.generativeai as genai

# ── Try importing ultralytics; fallback to mock if not installed ──
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Running in mock mode.")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish', 'waterweeds']
CLASS_COLORS = {
    'holothurian': (0, 191, 165),
    'echinus':     (0, 229, 255),
    'scallop':     (179, 157, 219),
    'starfish':    (255, 138, 101),
    'waterweeds':  (129, 199, 132),
}

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/urpc2020_yolov8.pt")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini AI configured")
else:
    gemini_model = None
    print("⚠️  GEMINI_API_KEY not set. AI descriptions will use fallback.")

app = FastAPI(
    title="DeepScan AI",
    description="Real-Time Underwater Object Detection using YOLOv8 + Claude AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

model = None

@app.on_event("startup")
async def load_model():
    global model
    if YOLO_AVAILABLE and Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        print(f"✅ YOLOv8 model loaded: {MODEL_PATH}")
    else:
        model = None
        print("⚠️  Using mock detection (no model found)")

# ─────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────

class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]        # [x, y, w, h] normalized 0-1
    bbox_pixels: List[int]   # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    annotated_image_b64: str
    inference_time_ms: float
    image_size: List[int]
    total_objects: int
    unique_species: int

# ─────────────────────────────────────────────────────────────
#  DETECTION ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Run YOLOv8 object detection on an uploaded underwater image.
    Returns annotated image + detection data.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    orig_h, orig_w = img_array.shape[:2]

    start_time = time.perf_counter()

    if model is not None:
        detections = run_yolo(img_array, conf_threshold, iou_threshold, orig_w, orig_h)
    else:
        detections = mock_detections(orig_w, orig_h)

    inference_ms = (time.perf_counter() - start_time) * 1000

    annotated = draw_annotations(img_array.copy(), detections)
    annotated_b64 = encode_image_b64(annotated)

    unique_species = len(set(d.class_id for d in detections))

    return DetectionResponse(
        success=True,
        detections=detections,
        annotated_image_b64=annotated_b64,
        inference_time_ms=round(inference_ms, 2),
        image_size=[orig_w, orig_h],
        total_objects=len(detections),
        unique_species=unique_species
    )


def run_yolo(img_array, conf, iou, w, h) -> List[Detection]:
    results = model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        verbose=False
    )[0]

    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        detections.append(Detection(
            id=i,
            class_id=cls_id,
            class_name=cls_name,
            confidence=confidence,
            bbox=[(x1/w), (y1/h), ((x2-x1)/w), ((y2-y1)/h)],
            bbox_pixels=[int(x1), int(y1), int(x2), int(y2)]
        ))
    return detections


def mock_detections(w, h) -> List[Detection]:
    """Realistic mock detections for demo/testing."""
    raws = [
        (0, 0.12, 0.42, 0.22, 0.54, 0.934),
        (1, 0.61, 0.18, 0.76, 0.38, 0.871),
        (3, 0.35, 0.71, 0.52, 0.90, 0.812),
        (2, 0.15, 0.28, 0.29, 0.46, 0.756),
        (4, 0.73, 0.62, 0.92, 0.90, 0.643),
    ]
    dets = []
    for i, (cls_id, nx1, ny1, nx2, ny2, conf) in enumerate(raws):
        x1, y1 = int(nx1*w), int(ny1*h)
        x2, y2 = int(nx2*w), int(ny2*h)
        dets.append(Detection(
            id=i,
            class_id=cls_id,
            class_name=CLASS_NAMES[cls_id],
            confidence=conf,
            bbox=[nx1, ny1, nx2-nx1, ny2-ny1],
            bbox_pixels=[x1, y1, x2, y2]
        ))
    return dets

# ─────────────────────────────────────────────────────────────
#  ANNOTATION DRAWING
# ─────────────────────────────────────────────────────────────

def draw_annotations(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """Draw bounding boxes with YOLOv8-style labels on the image."""
    for det in dets:
        x1, y1, x2, y2 = det.bbox_pixels
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
        bgr = (color[2], color[1], color[0])

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

        # Corner highlights
        cs = 14
        cv2.line(img, (x1, y1), (x1+cs, y1), bgr, 3)
        cv2.line(img, (x1, y1), (x1, y1+cs), bgr, 3)
        cv2.line(img, (x2, y1), (x2-cs, y1), bgr, 3)
        cv2.line(img, (x2, y1), (x2, y1+cs), bgr, 3)
        cv2.line(img, (x1, y2), (x1+cs, y2), bgr, 3)
        cv2.line(img, (x1, y2), (x1, y2-cs), bgr, 3)
        cv2.line(img, (x2, y2), (x2-cs, y2), bgr, 3)
        cv2.line(img, (x2, y2), (x2, y2-cs), bgr, 3)

        # Label
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(img, (x1-1, y1-th-8), (x1+tw+12, y1), (10, 20, 35), -1)
        cv2.rectangle(img, (x1-1, y1-th-8), (x1+tw+12, y1), bgr, 1)
        cv2.putText(img, label, (x1+5, y1-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img


def encode_image_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()

# ─────────────────────────────────────────────────────────────
#  CLAUDE AI SPECIES ENDPOINT
# ─────────────────────────────────────────────────────────────

FALLBACK_INFO = {
    'holothurian': {
        'common_name': 'Sea Cucumber', 'scientific_name': 'Holothuroidea spp.',
        'habitat': 'Seafloor / Benthic zone', 'depth_range': '0 – 6000m',
        'danger_level': 1, 'size': '10 – 30 cm', 'diet': 'Detritus, algae',
        'conservation_status': 'Varies by species',
        'description': 'Sea cucumbers are echinoderms that recycle nutrients from seafloor sediments. They play a vital ecological role and can expel their internal organs as defense.',
        'fun_fact': 'Some species breathe through their rear end by pumping water in and out!'
    },
    'echinus': {
        'common_name': 'Sea Urchin', 'scientific_name': 'Echinoidea spp.',
        'habitat': 'Rocky reefs', 'depth_range': '0 – 1000m',
        'danger_level': 2, 'size': '5 – 15 cm', 'diet': 'Algae, kelp',
        'conservation_status': 'Least Concern',
        'description': 'Sea urchins are keystone species on coral reefs, controlling algae growth. Their spines can pierce skin and break off, causing painful wounds.',
        'fun_fact': "Sea urchins have five teeth in a structure called Aristotle's Lantern!"
    },
    'scallop': {
        'common_name': 'Scallop', 'scientific_name': 'Pectinidae spp.',
        'habitat': 'Sandy / Gravel seabed', 'depth_range': '0 – 400m',
        'danger_level': 1, 'size': '5 – 20 cm', 'diet': 'Plankton, algae',
        'conservation_status': 'Least Concern',
        'description': 'Scallops are free-swimming bivalves with up to 200 simple eyes. They escape predators by rapidly clapping their shells.',
        'fun_fact': 'Scallops can swim by jet propulsion, clapping shells up to 5 times per second!'
    },
    'starfish': {
        'common_name': 'Starfish (Sea Star)', 'scientific_name': 'Asteroidea spp.',
        'habitat': 'Rocky / Sandy seabed', 'depth_range': '0 – 6000m',
        'danger_level': 1, 'size': '12 – 50 cm', 'diet': 'Mussels, clams, oysters',
        'conservation_status': 'Varies by species',
        'description': 'Starfish are apex predators of intertidal zones. They digest prey outside their body by pushing their stomach out through their mouth.',
        'fun_fact': 'A starfish can regenerate a lost arm — and that arm can grow a whole new body!'
    },
    'waterweeds': {
        'common_name': 'Aquatic Seaweed', 'scientific_name': 'Various marine flora',
        'habitat': 'Coastal / Reef zones', 'depth_range': '0 – 50m',
        'danger_level': 0, 'size': 'Varies', 'diet': 'Photosynthesis',
        'conservation_status': 'Stable',
        'description': 'Marine aquatic plants form the foundation of underwater food chains. They produce oxygen, provide habitat for juvenile fish, and absorb CO2.',
        'fun_fact': 'Kelp forests can grow up to 60cm per day — the fastest-growing organisms on Earth!'
    },
}


@app.get("/species/{species_name}")
async def get_species_info(species_name: str):
    """Query Gemini AI for detailed species information."""

    if gemini_model is None:
        info = FALLBACK_INFO.get(species_name.lower())
        if info:
            return info
        raise HTTPException(404, f"Species '{species_name}' not found")

    prompt = f"""You are a marine biologist. Provide info about: {species_name}
Return ONLY valid JSON (no markdown, no backticks):
{{
  "common_name": "string",
  "scientific_name": "string",
  "habitat": "string",
  "depth_range": "string",
  "danger_level": 1,
  "size": "string",
  "diet": "string",
  "conservation_status": "string",
  "description": "2-3 sentences",
  "fun_fact": "one surprising fact"
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        info = FALLBACK_INFO.get(species_name.lower())
        if info:
            return info
        raise HTTPException(500, "AI query failed")

# ─────────────────────────────────────────────────────────────
#  HEALTH & INFO
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "DeepScan AI",
        "version": "1.0.0",
        "model": "YOLOv8",
        "dataset": "URPC2020",
        "classes": CLASS_NAMES,
        "ai_provider": "Google Gemini",
        "ai_enabled": gemini_model is not None,
    }

@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)