#!/usr/bin/env python3
import os
import time
import subprocess
import sys
from client import RoboBrainClient
from utils import show_masks, read_image
import numpy as np
import cv2

def test_service():
    print("Testing RoboBrain2.0 Service and Client...")
    
    client = RoboBrainClient('http://100.79.185.61:8000')
    
    print("\n1. Health check...")
    try:
        health = client.health_check()
        print(f"✅ Health check passed: {health}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    print("\n2. Getting model info...")
    try:
        info = client.get_model_info()
        print(f"✅ Model info: {info}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    print("\n3. Testing with URL (original functionality)...")
    try:
        url_result = client.inference_with_url(
            text="What is shown in this image?",
            image_url="http://images.cocodataset.org/val2017/000000039769.jpg",
            task="general",
            enable_thinking=False
        )
        print(f"✅ URL inference: {url_result['answer'][:50]}...")
    except Exception as e:
        print(f"❌ URL inference failed: {e}")
    
    print("\n4. Testing different tasks with file upload...")
    tasks_to_test = ["mask"]
    # Test with local image file
    # image_path = './assets/demo_maniskill/rabbit.jpg'
    image_path = './assets/demo_maniskill/peg_assemble_human_view.png'
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        print("Please make sure the test image exists")
        return False
    
    for task in tasks_to_test:
        try:
            if task == "pointing":
                prompt = "Point to the pegs in this image"
            elif task == "grounding":
                prompt = "pegs"
            elif task == "mask":
                # positive
                # prompt = "Identify surfaces or regions in the image that are unsuitable for supporting the pink peg in a stable manner."
                # negative
                prompt = "Mark regions where the peg cannot stand without falling or not rigid enough to support the object."
            else:
                prompt = "Describe this image"
            
            print(f"\n   Testing task: {task}")
            result = client.inference_with_file_upload(
                text=prompt,
                image_path=image_path,
                task=task,
                enable_thinking=False
            )
            
            print(f"   Answer: {result['answer'][:100]}...")
            
            if result.get('points'):
                print(f"   Points: {result['points']}")
            if result.get('trajectory'):
                print(f"   Trajectory: {result['trajectory']}")
            if result.get('bounding_boxes'):
                print(f"   Bounding boxes: {result['bounding_boxes']}")
            if result.get('mask'):
                os.makedirs('./test_results', exist_ok=True)
                show_masks(
                    save_to=f"./test_results/{task}_mask.png",
                    image=read_image(image_path),
                    masks=np.array(result['mask']),
                    scores=np.array([1.0] * len(result['mask'])),
                    point_coords=np.array(result.get('points')) if result.get('points') else None,
                    box_coords=np.array(result.get('bounding_boxes')) if result.get('bounding_boxes') else None,
                    input_labels=np.array([1] * len(result.get('points'))) if result.get('points') else None,
                    borders=True
                )
                print(f"   Mask saved to: ./test_results/{task}_mask.png")
            
            print(f"✅ Task '{task}' completed successfully")
            
        except Exception as e:
            print(f"❌ Task '{task}' failed: {e}")
    
    print("\n5. Testing automatic detection (URL vs file)...")
    try:
        # Should automatically detect this is a URL and use URL endpoint
        auto_url_result = client.inference(
            text="What is shown in this image?",
            image="http://images.cocodataset.org/val2017/000000039769.jpg",
            task="general"
        )
        print(f"✅ Auto URL detection: {auto_url_result['answer'][:50]}...")
        
        # Should automatically detect this is a file and use upload endpoint
        auto_file_result = client.inference(
            text="Describe this image",
            image=image_path,
            task="general"
        )
        print(f"✅ Auto file detection: {auto_file_result['answer'][:50]}...")
        
    except Exception as e:
        print(f"❌ Auto detection failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Starting service test...")
    print("Make sure to run 'python service.py' in another terminal first!")
    print("Waiting 3 seconds for you to start the service...")
    time.sleep(3)
    
    success = test_service()
    
    if success:
        print("\n✅ All tests completed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)