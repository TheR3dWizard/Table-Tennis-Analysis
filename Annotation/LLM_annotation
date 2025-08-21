import requests
import base64
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi:latest"  # or your chosen model
IMAGE_PATH = "/home/dwarkesh/Desktop/project work 1/image.png"  # your image path

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def ask_ollama(prompt, image_path):
    image_base64 = encode_image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    end_time = time.time()

    if response.status_code == 200:
        print("Ollama Response:", response.json()["response"])
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    ask_ollama("In one short sentence, describe only the main activity in the attached table tennis game.", IMAGE_PATH)
