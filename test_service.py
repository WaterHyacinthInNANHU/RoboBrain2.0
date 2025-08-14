#!/usr/bin/env python3

import time
import subprocess
import sys
from client import RoboBrainClient

def test_service():
    print("Testing RoboBrain2.0 Service and Client...")
    
    client = RoboBrainClient()
    
    print("\n1. Health check...")
    try:
        health = client.health_check()
        print(f"✓ Health check passed: {health}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    print("\n2. Getting model info...")
    try:
        info = client.get_model_info()
        print(f"✓ Model info: {info}")
    except Exception as e:
        print(f"✗ Model info failed: {e}")
    
    print("\n3. Testing inference with URL...")
    try:
        prompt = "What is shown in this image?"
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
        result = client.inference(
            text=prompt,
            images=[image_url],
            task="general",
            enable_thinking=True
        )
        print(f"✓ Inference successful!")
        print(f"Answer: {result['answer'][:100]}...")
        if result.get('thinking'):
            print(f"Thinking: {result['thinking'][:100]}...")
        if result.get('points'):
            print(f"Points: {result['points']}")
        if result.get('trajectory'):
            print(f"Trajectory: {result['trajectory']}")
        if result.get('bounding_boxes'):
            print(f"Bounding boxes: {result['bounding_boxes']}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
    
    print("\n4. Testing different tasks...")
    tasks_to_test = ["general", "pointing", "grounding"]
    
    for task in tasks_to_test:
        try:
            if task == "pointing":
                prompt = "Point to the cats in this image"
            elif task == "grounding":
                prompt = "cats"
            else:
                prompt = "Describe this image"
                
            result = client.inference(
                text=prompt,
                images=[image_url],
                task=task,
                enable_thinking=False
            )
            print(f"✓ Task '{task}' completed: {result['answer'][:50]}...")
            if result.get('points'):
                print(f"  Points: {result['points']}")
            if result.get('trajectory'):
                print(f"  Trajectory: {result['trajectory']}")
            if result.get('bounding_boxes'):
                print(f"  Bounding boxes: {result['bounding_boxes']}")
        except Exception as e:
            print(f"✗ Task '{task}' failed: {e}")
    
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