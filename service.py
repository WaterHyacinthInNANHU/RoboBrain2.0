from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import tempfile
import os
import shutil
import argparse
from inference import UnifiedInference

class InferenceRequest(BaseModel):
    text: str
    image_urls: Optional[List[str]] = None
    task: str = "general"
    plot: bool = False
    enable_thinking: Optional[bool] = None
    do_sample: bool = True
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None
    plot_path: Optional[str] = None

class ModelConfig(BaseModel):
    model_id: str = "BAAI/RoboBrain2.0-7B"
    device_map: str = "auto"

inference_model = None
config_args = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_model
    inference_model = UnifiedInference(
        model_id=config_args.model_id,
        device_map=config_args.device_map
    )
    yield
    inference_model = None

app = FastAPI(title="RoboBrain2.0 Inference Service", version="1.0.0", lifespan=lifespan)

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
    
    try:
        if request.image_urls:
            images = request.image_urls
        else:
            raise HTTPException(status_code=400, detail="No images provided")
        
        result = inference_model.inference(
            text=request.text,
            image=images,
            task=request.task,
            plot=request.plot,
            enable_thinking=request.enable_thinking,
            do_sample=request.do_sample,
            temperature=request.temperature
        )
        
        response = InferenceResponse(
            answer=result["answer"],
            thinking=result.get("thinking"),
            plot_path=None
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/inference_with_upload", response_model=InferenceResponse)
async def run_inference_with_upload(
    text: str = Form(...),
    task: str = Form("general"),
    plot: bool = Form(False),
    enable_thinking: Optional[bool] = Form(None),
    do_sample: bool = Form(True),
    temperature: float = Form(0.7),
    files: List[UploadFile] = File(...)
):
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        temp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_paths.append(tmp.name)
        
        result = inference_model.inference(
            text=text,
            image=temp_paths,
            task=task,
            plot=plot,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
            temperature=temperature
        )
        
        for temp_path in temp_paths:
            os.unlink(temp_path)
        
        response = InferenceResponse(
            answer=result["answer"],
            thinking=result.get("thinking"),
            plot_path=None
        )
        
        return response
    except Exception as e:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboBrain2.0 Inference Service")
    parser.add_argument("--model_id", type=str, default="BAAI/RoboBrain2.0-7B", 
                       help="Model ID or path (default: BAAI/RoboBrain2.0-7B)")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping strategy (default: auto)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload")
    
    config_args = parser.parse_args()
    
    print(f"Starting RoboBrain2.0 service with:")
    print(f"  Model ID: {config_args.model_id}")
    print(f"  Device Map: {config_args.device_map}")
    print(f"  Host: {config_args.host}")
    print(f"  Port: {config_args.port}")
    
    uvicorn.run("service:app", host=config_args.host, port=config_args.port, reload=config_args.reload)