from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import tempfile
import os
import shutil
import argparse
import re
from dataclasses import dataclass
from inference import UnifiedInference

import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import io
from PIL import Image

from utils import show_masks, read_image


MODEL_ID = "BAAI/RoboBrain2.0-7B"
DEVICE_MAP = "auto"


# @dataclass
class InferenceRequest(BaseModel):
    text: str
    image: str
    task: str = "general"
    plot: bool = False
    enable_thinking: Optional[bool] = None
    do_sample: bool = True
    temperature: float = 0.7
    
# @dataclass
class InferenceResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None
    # Structured data based on task type
    points: Optional[List[List[int]]] = None  # For pointing task: [[x1, y1], [x2, y2], ...]
    trajectory: Optional[List[List[int]]] = None  # For trajectory task: [[x1, y1], [x2, y2], ...]
    bounding_boxes: Optional[List[List[int]]] = None  # For affordance/grounding: [[x1, y1, x2, y2], ...]
    mask: Optional[List[List[List[int]]]] = None  # For mask task

# @dataclass
class ModelConfig(BaseModel):
    model_id: str = "BAAI/RoboBrain2.0-7B"
    device_map: str = "auto"

inference_model = None
sam2_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_model, sam2_predictor
    inference_model = UnifiedInference(
        model_id=MODEL_ID,
        device_map=DEVICE_MAP
    )

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    yield
    inference_model = None

app = FastAPI(title="RoboBrain2.0 Inference Service", version="1.0.0", lifespan=lifespan)

def parse_structured_data(answer_text: str, task: str):
    """Extract structured data from answer text based on task type"""
    points = None
    trajectory = None
    bounding_boxes = None
    
    if task == "pointing":
        # Extract points in format [(x1, y1), (x2, y2), ...]
        point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(point_pattern, answer_text)
        print(f"Matches for points: {matches}")
        if matches:
            points = [[int(x), int(y)] for x, y in matches]
    
    elif task == "trajectory":
        # Extract trajectory points in format [[x1, y1], [x2, y2], ...] or (x1, y1), (x2, y2)
        trajectory_pattern = r'(\d+),\s*(\d+)'
        matches = re.findall(trajectory_pattern, answer_text)
        if matches:
            trajectory = [[int(x), int(y)] for x, y in matches]
    
    elif task in ["affordance", "grounding"]:
        # Extract bounding boxes in format [x1, y1, x2, y2]
        box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(box_pattern, answer_text)
        if matches:
            bounding_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in matches]
    
    return points, trajectory, bounding_boxes

@app.post("/initialize", response_model=dict)
async def initialize_model(config: ModelConfig):
    global inference_model
    try:
        inference_model = UnifiedInference(
            model_id=config.model_id,
            device_map=config.device_map
        )
        return {"status": "Model initialized successfully", "model_id": config.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    image = request.image
    
    brain_task = request.task if request.task != 'mask' else "pointing"
    result = inference_model.inference(
        text=request.text,
        image=image,
        task=brain_task,
        plot=request.plot,
        enable_thinking=request.enable_thinking,
        do_sample=request.do_sample,
        temperature=request.temperature
    )
    # Parse structured data from the answer
    points, trajectory, bounding_boxes = parse_structured_data(result["answer"], brain_task)
    print(request.task, result["answer"], points, trajectory, bounding_boxes)
    
    # get masks
    if request.task == 'mask':
        assert points is not None, "Points must be provided for mask task. Got answer: " + result["answer"]
        sam2_predictor.set_image(read_image(image))
        input_point = np.array(points)
        input_label = np.array([1] * len(points)) # all positive points
        masks, scores, _ = sam2_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        # scores = scores[sorted_ind]
        masks = masks.tolist()
    
    response = InferenceResponse(
        answer=result["answer"],
        thinking=result.get("thinking"),
        points=points,
        trajectory=trajectory,
        bounding_boxes=bounding_boxes,
        mask=masks if request.task == 'mask' else None
    )
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": inference_model is not None}

@app.get("/model_info")
async def get_model_info():
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return {
        "model_id": inference_model.model_id,
        "supports_thinking": inference_model.supports_thinking
    }


def run():
    parser = argparse.ArgumentParser(description="RoboBrain2.0 Inference Service")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload")

    config_args = parser.parse_args()
    
    uvicorn.run("service:app", host=config_args.host, port=config_args.port, reload=config_args.reload)
    
    
if __name__ == "__main__":
    run()