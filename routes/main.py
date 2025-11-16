import os
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Put it in a .env file or env var.")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

# Choose your models
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image-preview"

def enforce_sentence_limit(text, max_sentences):
    sentences = []
    current = ""

    for char in text:
        current += char
        if char == ".":
            sentences.append(current.strip())
            current = ""

        if len(sentences) == max_sentences:
            break

    if not sentences:
        return text

    return " ".join(sentences)


@app.route("/generate/text", methods=["POST"])
def generate_text_route():
    """
    JSON body:
    {
      "prompt": "your text prompt here"
    }
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
        )

        # Concatenate all text parts
        text_parts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)

        full_text = enforce_sentence_limit("\n".join(text_parts).strip(), 3)

        return jsonify({"text": full_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"Error generating image: {e}")  # This will show in your terminal
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)