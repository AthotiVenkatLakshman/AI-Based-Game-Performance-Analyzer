import os
import requests
from dotenv import load_dotenv

load_dotenv()
hf_key = os.getenv("HF_API_KEY")

def test_hf_qa():
    print("--- Testing Hugging Face QA ---")
    if not hf_key or hf_key == "your_hf_api_key_here":
        print("❌ No valid HF API Key found")
        return
    try:
        url = "https://router.huggingface.co/models/deepset/roberta-base-squad2"
        headers = {"Authorization": f"Bearer {hf_key}"}
        payload = {"inputs": {"question": "What is the capital of France?", "context": "The capital of France is Paris."}}
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"❌ HF QA Error: {e}")

if __name__ == "__main__":
    test_hf_qa()
