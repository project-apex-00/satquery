"""
gemini_client.py

Thin wrapper around the Gemini API. Reads the API key from an
environment variable (GEMINI_API_KEY) -- never hardcode it in code.

On Railway: set GEMINI_API_KEY in your project's Variables tab.
Locally: put it in a .env file (see .env.example) and it loads automatically.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # loads .env if present (local dev only -- Railway injects vars directly)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Locally: add it to a .env file. "
        "On Railway: add it under Project -> Variables."
    )

genai.configure(api_key=API_KEY)
_model = genai.GenerativeModel("gemini-2.0-flash")  # fast + cheap, good for this use case


def ask_gemini(user_question: str, specialist_result: dict) -> str:
    """
    Takes the user's original question + the specialist model's structured
    output, and asks Gemini to turn it into a natural-language answer.

    This is the "specialist model output -> Gemini reasoning" combo that
    makes the answer RS-adapted rather than a bare Gemini guess.
    """
    prompt = f"""You are a remote-sensing analysis assistant.

A specialist AI model (fine-tuned on real Sentinel-2 satellite imagery)
analyzed the user's uploaded image and produced this structured result:

Predicted land-cover class: {specialist_result.get('predicted_class')}
Confidence: {specialist_result.get('confidence')}
All class probabilities: {specialist_result.get('all_probs')}

The user's question was: "{user_question}"

Using ONLY the specialist model's result above as ground truth, write a
clear, natural-language answer to the user's question. If the question
asks something the classifier result can't answer, say so honestly
rather than guessing.
"""
    response = _model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    # quick manual test
    fake_result = {
        "predicted_class": "Forest",
        "confidence": 0.996,
        "all_probs": {"Forest": 0.996, "Pasture": 0.003},
    }
    print(ask_gemini("What kind of land is this?", fake_result))
