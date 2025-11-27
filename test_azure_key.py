"""Azure Speech Key Test"""
import os
from dotenv import load_dotenv
import requests

# Load .env
load_dotenv()

key = os.getenv('AZURE_SPEECH_KEY')
region = os.getenv('AZURE_SPEECH_REGION')

print(f"[OK] Key loaded: {key[:15]}...{key[-15:]}")
print(f"[OK] Region: {region}")
print()

# Test Azure Speech Token issuance
print("[TEST] Testing Azure Speech token issuance...")
url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
headers = {
    "Ocp-Apim-Subscription-Key": key
}

try:
    response = requests.post(url, headers=headers, timeout=10)
    print(f"[RESPONSE] Status: {response.status_code}")

    if response.status_code == 200:
        token = response.text
        print(f"[SUCCESS] Token issued successfully!")
        print(f"Token (first 50 chars): {token[:50]}...")
    else:
        print(f"[FAILED] Token issuance failed: {response.status_code}")
        print(f"Error: {response.text}")

except Exception as e:
    print(f"[ERROR] Request failed: {str(e)}")
