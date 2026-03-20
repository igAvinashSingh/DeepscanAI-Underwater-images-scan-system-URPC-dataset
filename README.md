# 🌊 DeepScan AI — Underwater Object Detection System

**Real-time underwater species detection using YOLOv8 + Claude AI**  
Dataset: URPC2020 (National Underwater Robotics Professional Competition 2020)

---

## 🎯 Overview

DeepScan AI is a full-stack underwater object detection system that:
1. **Detects** 5 underwater species using YOLOv8
2. **Annotates** images with bounding boxes and confidence scores
3. **Explains** detected species using Claude AI (habitat, behavior, ecology)

### Detected Classes (URPC2020)
| ID | Class | Description |
|----|-------|-------------|
| 0  | holothurian | Sea cucumber |
| 1  | echinus | Sea urchin |
| 2  | scallop | Bivalve mollusc |
| 3  | starfish | Sea star |
| 4  | waterweeds | Aquatic plants |

---

## 📁 Project Structure

```
underwater_detection/
├── index.html              # ← Frontend UI (open this in browser)
├── backend/
│   ├── app.py              # FastAPI server (YOLOv8 + Claude AI)
│   ├── train.py            # Training & evaluation pipeline
│   └── requirements.txt    # Python dependencies
├── models/
│   └── urpc2020_yolov8.pt  # Trained model (you provide)
└── README.md
```

---

## 🚀 Quick Start

### Option A: Frontend Only (No Backend Needed)
Just open `index.html` in your browser. It includes:
- Simulated YOLOv8 detections
- Real Claude AI species descriptions (needs API key in browser)
- Full interactive UI

### Option B: Full Stack with Real YOLOv8

#### 1. Install dependencies
```bash
pip install fastapi uvicorn ultralytics anthropic pillow opencv-python pyyaml
```

#### 2. Get the URPC2020 Dataset
Download from: https://github.com/xiaoDetection/Learning-Heavily-Degraded-Prior
Or from the official competition: http://www.urpc.com.cn/

#### 3. Prepare dataset structure
```
urpc2020/
├── images/
│   ├── train/   ← training images (.jpg)
│   └── val/     ← validation images (.jpg)
└── labels/
    ├── train/   ← YOLO labels (.txt)
    └── val/     ← YOLO labels (.txt)
```

#### 4. Convert XML annotations to YOLO format
```bash
python backend/train.py convert \
  --xml-dir urpc2020/annotations/train \
  --out-dir urpc2020/labels/train
```

#### 5. Train YOLOv8
```bash
# Medium model (recommended)
python backend/train.py train --model yolov8m.pt --epochs 150 --batch 16

# Lightweight for faster training
python backend/train.py train --model yolov8s.pt --epochs 100 --batch 32
```

#### 6. Evaluate
```bash
python backend/train.py eval --weights runs/detect/urpc2020_yolov8/weights/best.pt
```

#### 7. Start backend server
```bash
export ANTHROPIC_API_KEY="your-key-here"
export YOLO_MODEL_PATH="runs/detect/urpc2020_yolov8/weights/best.pt"

python backend/app.py
# Server starts at http://localhost:8000
```

#### 8. Open the frontend
Open `index.html` in your browser, or serve it:
```bash
python -m http.server 3000
# Visit http://localhost:3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Upload image → get detections + annotated image |
| GET | `/species/{name}` | Get Claude AI species info |
| GET | `/health` | Health check |
| GET | `/` | API info |

### Example: Detect Objects
```python
import requests

with open('underwater.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        params={'conf_threshold': 0.25}
    )

data = response.json()
print(f"Found {data['total_objects']} objects in {data['inference_time_ms']:.1f}ms")
for det in data['detections']:
    print(f"  {det['class_name']}: {det['confidence']:.1%}")
```

---

## 🤖 YOLOv8 Architecture

```
Input (640×640)
    ↓
CSPDarknet53 Backbone  ← Feature extraction
    ↓
PANNet Neck            ← Multi-scale feature fusion
    ↓
Detection Head         ← 3 scales (80×80, 40×40, 20×20)
    ↓
NMS Post-processing
    ↓
Detections [class, confidence, bbox]
```

### Recommended Models by Use Case
| Model | Size | mAP | Speed | Use Case |
|-------|------|-----|-------|----------|
| YOLOv8n | 3.2M | ~45% | 80fps | Edge devices |
| YOLOv8s | 11M | ~50% | 60fps | Real-time |
| YOLOv8m | 25M | ~55% | 40fps | **Recommended** |
| YOLOv8l | 43M | ~58% | 25fps | High accuracy |

---

## 🌊 Underwater-Specific Training Tips

1. **Color jitter**: Heavy HSV augmentation to simulate varying water conditions
2. **Brightness variation**: Underwater light varies significantly with depth
3. **Mosaic augmentation**: Helps with small/occluded objects
4. **Mixed precision**: Use `amp=True` for faster GPU training
5. **Pre-training**: Start from COCO weights, fine-tune on URPC2020

### Expected Performance (YOLOv8m, 150 epochs)
- mAP@0.5: ~72-78%
- mAP@0.5:0.95: ~52-58%
- Inference: ~18ms on GPU

---

## 🔑 Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required for AI species descriptions
YOLO_MODEL_PATH=models/best.pt  # Path to trained .pt file
PORT=8000                        # Server port (default: 8000)
```

---

## 📊 Dataset Stats (URPC2020)

- ~5,543 underwater images
- ~30,000+ annotated instances
- Images from South China Sea, Yellow Sea
- Resolution: 800×600 to 1920×1080
- Captured at depths: 0–50m

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8 (Ultralytics) |
| Backend | FastAPI + Python |
| AI Descriptions | Claude claude-sonnet-4-20250514 (Anthropic) |
| Frontend | Vanilla HTML/CSS/JS |
| Image Processing | OpenCV + Pillow |
| Deployment | Docker / Uvicorn |

---

## 📄 License
MIT License — Free for research and educational use.

Dataset: URPC2020 — Please cite the original competition organizers.
