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
    
    def inference_with_urls(
        self,
        text: str,
        image_urls: List[str],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        data = {
            "text": text,
            "image_urls": image_urls,
            "task": task,
            "plot": plot,
            "enable_thinking": enable_thinking,
            "do_sample": do_sample,
            "temperature": temperature
        }
        response = self.session.post(f"{self.base_url}/inference", json=data)
        response.raise_for_status()
        return response.json()
    
    def inference_with_files(
        self,
        text: str,
        image_paths: List[Union[str, Path]],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        files = []
        for path in image_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            files.append(("files", (path.name, open(path, "rb"), f"image/{path.suffix[1:]}")))
        
        data = {
            "text": text,
            "task": task,
            "plot": plot,
            "enable_thinking": enable_thinking,
            "do_sample": do_sample,
            "temperature": temperature
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/inference_with_upload",
                data=data,
                files=files
            )
            response.raise_for_status()
            return response.json()
        finally:
            for _, file_tuple in files:
                file_tuple[1].close()
    
    def inference(
        self,
        text: str,
        images: Union[List[str], List[Path]],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        if not images:
            raise ValueError("At least one image must be provided")
        
        first_image = images[0]
        if isinstance(first_image, (str, Path)):
            if isinstance(first_image, str) and first_image.startswith(("http://", "https://")):
                return self.inference_with_urls(
                    text=text,
                    image_urls=images,
                    task=task,
                    plot=plot,
                    enable_thinking=enable_thinking,
                    do_sample=do_sample,
                    temperature=temperature
                )
            else:
                return self.inference_with_files(
                    text=text,
                    image_paths=images,
                    task=task,
                    plot=plot,
                    enable_thinking=enable_thinking,
                    do_sample=do_sample,
                    temperature=temperature
                )
        else:
            raise ValueError("Images must be file paths or URLs")

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
        images=[image_url],
        task="general",
        enable_thinking=True
    )
    print("Result:", result)