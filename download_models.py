#!/usr/bin/env python3
"""
Script to download YOLO models for person detection
"""

import os
import urllib.request
import zipfile

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    try:
        # Create SSL context that doesn't verify certificates
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create opener with SSL context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        # Try alternative method
        try:
            import requests
            response = requests.get(url, verify=False)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename} (alternative method)")
        except Exception as e2:
            print(f"Failed to download {filename} with alternative method: {e2}")
            raise

def download_yolo_models():
    """Download YOLO v3 models for person detection."""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # YOLO v3 files
    yolo_files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filename, url in yolo_files.items():
        if not os.path.exists(filename):
            try:
                download_file(url, filename)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping...")
    
    print("\nYOLO models download completed!")
    print("You can now run the burglar detection system with person detection enabled.")

if __name__ == "__main__":
    download_yolo_models()
