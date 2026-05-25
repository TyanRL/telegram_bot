import requests
import json
import time
import dotenv
import os

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Step 1: Submit video generation request
response = requests.post(
  url="https://openrouter.ai/api/v1/videos",
  headers={
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "x-ai/grok-imagine-video",
    "prompt": "A serene mountain landscape at sunset with clouds drifting by"
  })
)
print(response.status_code)
result = response.json()
print(result)
response.raise_for_status()

if "id" not in result:
    raise RuntimeError(f"Unexpected response: {result}")

job_id = result["id"]
polling_url = result["polling_url"]
print(f"Job submitted: {job_id}")

# Step 2: Poll for completion
while True:
  poll_response = requests.get(
    url=polling_url,
    headers={
      "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
  )
  status_data = poll_response.json()
  print(f"Status: {status_data['status']}")

  if status_data["status"] == "completed":
    for url in status_data.get("unsigned_urls", []):
      print(f"Video URL: {url}")
    break
  elif status_data["status"] == "failed":
    print(f"Error: {status_data.get('error', 'Unknown error')}")
    break

  time.sleep(5)