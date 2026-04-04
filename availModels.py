# temp_list.py - put this in root, run it
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    print(m.name)