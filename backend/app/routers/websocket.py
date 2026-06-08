"""
WebSocket endpoint for real-time webcam detection.

Flow:
    1. Frontend captures camera frames via getUserMedia
    2. Sends frames as base64 JPEG via WebSocket
    3. Backend decodes → runs YOLOv8 detection → returns JSON results
    4. Frontend draws bounding boxes on a canvas overlay

This keeps the model on the server and works even when deployed remotely.
"""

import base64
import json
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])
logger = logging.getLogger(__name__)

# Shared pipeline reference (set by main.py lifespan)
_pipeline = None


def set_pipeline(pipeline):
    """Called from main.py to inject the detection pipeline."""
    global _pipeline
    _pipeline = pipeline


@router.websocket("/detect")
async def websocket_detect(ws: WebSocket):
    """
    Real-time detection over WebSocket.

    Client sends:  {"image": "<base64 JPEG>", "conf": 0.25}
    Server replies: {
        "detections": [...],
        "count": N,
        "fps": X,
        "class_counts": {"car": 2, "minivan": 1},
        "inference_time_ms": Y
    }
    """
    await ws.accept()
    logger.info("WebSocket client connected for real-time detection")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Receive frame from client
            data = await ws.receive_text()
            msg = json.loads(data)

            if "image" not in msg:
                await ws.send_json({"error": "No image data"})
                continue

            # Decode base64 JPEG → numpy array
            img_b64 = msg["image"]
            # Strip data URL prefix if present
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_json({"error": "Failed to decode image"})
                continue

            # Run detection
            conf = msg.get("conf", 0.25)
            t0 = time.time()

            if _pipeline is None:
                await ws.send_json({"error": "Model not loaded"})
                continue

            result = _pipeline._detector.detect(frame, confidence=conf)
            inference_ms = (time.time() - t0) * 1000

            # Build response
            detections = []
            for det in result.detections:
                detections.append({
                    "x1": int(det.x1),
                    "y1": int(det.y1),
                    "x2": int(det.x2),
                    "y2": int(det.y2),
                    "class_name": det.class_name,
                    "confidence": round(det.confidence, 3),
                    "track_id": det.track_id,
                })

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            response = {
                "detections": detections,
                "count": len(detections),
                "class_counts": result.class_counts,
                "inference_time_ms": round(inference_ms, 1),
                "fps": round(fps, 1),
                "frame": frame_count,
            }

            await ws.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected after {frame_count} frames")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
