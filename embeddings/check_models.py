"""
Run this to see which Gemini models are available with your API key.
    python check_models.py
"""
import requests, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("❌ No GEMINI_API_KEY found in .env"); sys.exit(1)

print(f"Using key: {GEMINI_API_KEY[:12]}...")

# List all available models
url  = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
resp = requests.get(url, timeout=15)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    models = resp.json().get("models", [])
    print(f"\nAvailable models ({len(models)} total):\n")
    embed_models = []
    for m in models:
        name       = m.get("name", "")
        methods    = m.get("supportedGenerationMethods", [])
        print(f"  {name}  →  {methods}")
        if "embedContent" in methods:
            embed_models.append(name)
    print(f"\n✅ Models supporting embedContent: {embed_models}")
else:
    print(f"❌ Error: {resp.text}")