import requests
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

class RoboBrainClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def initialize_model(self, model_id: str = "BAAI/RoboBrain2.0-7B", device_map: str = "auto") -> Dict[str, Any]:
        data = {
            "model_id": model_id,
            "device_map": device_map
        }
        response = self.session.post(f"{self.base_url}/initialize", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/model_info")
        response.raise_for_status()
        return response.json()
    
    def inference(
        self,
        text: str,
        image: List[str],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        data = {
            "text": text,
            "image": image,
            "task": task,
            "plot": plot,
            "enable_thinking": enable_thinking,
            "do_sample": do_sample,
            "temperature": temperature
        }
        response = self.session.post(f"{self.base_url}/inference", json=data)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    client = RoboBrainClient()
    
    print("Health check:", client.health_check())
    
    print("Initializing model...")
    init_result = client.initialize_model()
    print("Init result:", init_result)
    
    print("Model info:", client.get_model_info())
    
    prompt = "What is shown in this image?"
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    print("Running inference...")
    result = client.inference(
        text=prompt,
        image=[image_url],
        task="general",
        enable_thinking=True
    )
    print("Result:", result)