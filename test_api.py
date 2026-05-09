import requests
import base64
import os

# --- CONFIGURATION ---
# Replace this with the link from your Google Colab box
BACKEND_URL = "https://cold-teeth-enjoy.loca.lt"

# Path to any image file on your computer (jpg or png)
IMAGE_PATH = "cuss-todo.png" 

def test_inference():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find {IMAGE_PATH}. Please put an image in this folder first!")
        return

    print(f"Reading image: {IMAGE_PATH}...")
    with open(IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    print(f"Sending request to Colab GPU...")
    
    payload = {
        "image_base64": f"data:image/jpeg;base64,{encoded_string}",
        "prompt": "Read the question in this image and answer it accurately."
    }
    
    headers = {
        "bypass-tunnel-reminder": "true" # This skips the Localtunnel warning page
    }

    try:
        response = requests.post(f"{BACKEND_URL}/api/analyze", json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*40)
            print("AI RESPONSE:")
            print(result["result"])
            print("="*40)
        else:
            print(f"Failed! Status Code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_inference()
