from flask import Flask, request, jsonify
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_model():
    data = request.get_json()
    prompt = data.get("prompt", "")
    model_name = data.get("model", "")

    if not prompt or not model_name:
        return jsonify({
            "success": False,
            "message": "Both 'prompt' and 'model' fields are required."
        }), 400

    try:
        # Initialize the selected model
        chat = ChatOllama(base_url=OLLAMA_URL, model=model_name, timeout=60)

        # Send the prompt to the model
        response = chat.invoke(prompt)

        return jsonify({
            "success": True,
            "model": model_name,
            "prompt": prompt,
            "response": response.content
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
