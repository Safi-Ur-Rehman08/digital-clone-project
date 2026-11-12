


import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

# IMPORTANT:
# Copy the exact working model name from llm_test.py
MODEL_NAME = "gemini-3.1-flash-lite"

client = genai.Client(api_key=API_KEY)


def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I don't have that in my notes."

    context = "\n\n".join(
        f"Source: {chunk.get('source', 'Unknown')}\n"
        f"{chunk.get('text', '')}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI digital clone of the document owner.

Rules:
1. Answer in first person.
2. Use ONLY the provided personal knowledge.
3. Never invent personal information.
4. If the answer is missing, say:
   "I don't have that in my notes."
5. Give clear and natural answers.
6. Never claim to be the real person.

PERSONAL KNOWLEDGE:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:

        print("\nSending request to Gemini...")
        print("Using model:", MODEL_NAME)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        print("Gemini response received successfully!")

        return response.text.strip()

    except Exception as e:

        print("\n========== GEMINI ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("==================================\n")

        raise



