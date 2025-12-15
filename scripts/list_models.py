import google.generativeai as genai
import os
from dotenv import load_dotenv

try:
    load_dotenv()
except:
    pass

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in env.")
    exit(1)

genai.configure(api_key=api_key)

print("Listing available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
