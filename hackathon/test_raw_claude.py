import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")

def test_claude_raw():
    print("--- Testing Claude Raw ---")
    if not api_key:
        print("❌ No API Key")
        return
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_claude_raw()
