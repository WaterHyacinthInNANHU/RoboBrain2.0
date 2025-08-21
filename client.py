import requests
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import base64
import mimetypes

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
        image: Union[str, Path],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run inference with either image path/URL or file upload
        
        Args:
            text: Input text prompt
            image: Can be:
                - Local file path (str or Path)
                - URL (str starting with http:// or https://)
            task: Task type
            plot: Whether to plot results
            enable_thinking: Enable thinking mode
            do_sample: Sampling flag
            temperature: Sampling temperature
        """
        image_str = str(image)
        
        # Check if image is a URL
        if image_str.startswith(('http://', 'https://')):
            # Use original endpoint for URLs
            data = {
                "text": text,
                "image": image_str,
                "task": task,
                "plot": plot,
                "enable_thinking": enable_thinking,
                "do_sample": do_sample,
                "temperature": temperature
            }
            response = self.session.post(f"{self.base_url}/inference", json=data)
        else:
            # Use file upload endpoint for local files
            image_path = Path(image_str)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Prepare form data
            data = {
                "text": text,
                "task": task,
                "plot": plot,
                "do_sample": do_sample,
                "temperature": temperature
            }
            
            # Add enable_thinking only if it's not None
            if enable_thinking is not None:
                data["enable_thinking"] = enable_thinking
            
            # Prepare file for upload
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if mime_type is None:
                mime_type = 'image/jpeg'  # Default fallback
            
            with open(image_path, 'rb') as f:
                files = {
                    'image': (image_path.name, f, mime_type)
                }
                response = self.session.post(
                    f"{self.base_url}/inference_upload",
                    data=data,
                    files=files
                )
        
        response.raise_for_status()
        return response.json()
    
    def inference_with_file_upload(
        self,
        text: str,
        image_path: Union[str, Path],
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Explicit method for file upload inference
        """
        return self.inference(
            text=text,
            image=image_path,
            task=task,
            plot=plot,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
            temperature=temperature
        )
    
    def inference_with_url(
        self,
        text: str,
        image_url: str,
        task: str = "general",
        plot: bool = False,
        enable_thinking: Optional[bool] = None,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Explicit method for URL-based inference
        """
        data = {
            "text": text,
            "image": image_url,
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
    
    # Test with URL (original functionality)
    prompt = "What is shown in this image?"
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    print("Running inference with URL...")
    result = client.inference(
        text=prompt,
        image=image_url,
        task="general",
        enable_thinking=True
    )
    print("Result:", result)
    
    # Test with local file upload
    # local_image_path = "./test_image.jpg"  # Uncomment and provide actual path
    # print("Running inference with file upload...")
    # result = client.inference(
    #     text=prompt,
    #     image=local_image_path,
    #     task="general",
    #     enable_thinking=True
    # )
    # print("Result:", result)