
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Select model
MODEL_NAME = "gemini-3.1-flash-lite"

# Connect to Gemini
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="Explain artificial intelligence in two simple sentences."
    )

    print("\nGEMINI API CONNECTION SUCCESSFUL!")
    print("\nAI RESPONSE:")
    print(response.text)

except Exception as e:
    print("\nAPI REQUEST FAILED!")
    print("Error:", str(e))

