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
    
    print("\n4. Testing different tasks...")
    # tasks_to_test = ["general", "pointing", "grounding", "mask"]
    tasks_to_test = ["mask"]
    image_path = './assets/demo_maniskill/peg_assemble_human_view.png'
    
    for task in tasks_to_test:
        if task == "pointing":
            prompt = "Point to the pegs in this image"
        elif task == "grounding":
            prompt = "pegs"
        elif task == "mask":
            # prompt = "Please point out the best place to insert the pink peg"
            # prompt = "There is a pink peg in the image. Please point out PLACES where the pink peg can NOT be placed STABLY"
            prompt = "There is a pink peg in the image. Please point out FIXTURES that can be used to hold the peg for reorientating it"
            # prompt = "There is a pink peg in the image. Please point out intermediate places to suitable to reorientate the peg "
        else:
            prompt = "Describe this image"
            
        result = client.inference(
            text=prompt,
            image=image_path,
            task=task,
            enable_thinking=False
        )
        if result.get('points'):
            print(f"  Points: {result['points']}")
        if result.get('trajectory'):
            print(f"  Trajectory: {result['trajectory']}")
        if result.get('bounding_boxes'):
            print(f"  Bounding boxes: {result['bounding_boxes']}")
        if result.get('mask'):
            # Convert byte mask(s) to numpy array(s)
            # print(f"  Mask: ")
            os.makedirs('./test_results', exist_ok=True)
            show_masks(
                save_to=f"./test_results/{task}_mask.png",
                image=read_image(image_path),
                masks=np.array(result['mask']),
                scores=np.array([1.0] * len(result['mask'])),  # Dummy scores for visualization
                point_coords=np.array(result.get('points')),
                box_coords=np.array(result.get('bounding_boxes')) if result.get('bounding_boxes') else None,
                input_labels=np.array([1] * len(result.get('points'))) if result.get('points') else None,  # Assuming no labels for this test
                borders=True
            )
        print(f"✓ Task '{task}' completed: {result['answer'][:50]}...")

    
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