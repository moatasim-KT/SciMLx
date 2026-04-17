import os
import urllib.request
import json
api_key = os.environ.get("HF_TOKEN")
req = urllib.request.Request("https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0", method="POST")
req.add_header("Authorization", f"Bearer {api_key}")
req.add_header("Content-Type", "application/json")
data = json.dumps({"inputs": "A scientific architectural schematic of a neural network model, beautiful, 4k"}).encode('utf-8')
from urllib.error import HTTPError
try:
    with urllib.request.urlopen(req, data=data) as f:
        print("Success:", f.status)
        with open("hf_test.png", "wb") as img:
            img.write(f.read())
except HTTPError as e:
    print("Error:", e.code, e.read().decode('utf-8'))
