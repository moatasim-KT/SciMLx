import os
import urllib.request
import json
api_key = os.environ.get("OPENAI_API_KEY")
req = urllib.request.Request("https://api.openai.com/v1/images/generations", method="POST")
req.add_header("Authorization", f"Bearer {api_key}")
req.add_header("Content-Type", "application/json")
data = json.dumps({
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024",
    "model": "dall-e-3"
}).encode('utf-8')
from urllib.error import URLError, HTTPError
try:
    with urllib.request.urlopen(req, data=data) as f:
        print("Success:", f.status)
        resp = json.loads(f.read().decode('utf-8'))
        print(resp['data'][0]['url'])
except HTTPError as e:
    print("Error:", e.read().decode('utf-8'))
