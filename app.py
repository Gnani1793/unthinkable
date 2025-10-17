import os
import google.generativeai as genai
from flask import Flask, request, render_template, redirect
from dotenv import load_dotenv

# For local transcription
import torch
import librosa
from transformers import pipeline

# --- Load API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# ✅ Configure the Gemini API correctly
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# --- Load Local Whisper Model ---
try:
    print("Loading local Whisper model...")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
    print("✅ Local Whisper model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local Whisper model: {e}")
    transcriber = None

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)

        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            transcript_text = ""
            summary_text = ""

            try:
                # --- Step 1: Transcription using Whisper ---
                if not transcriber:
                    raise RuntimeError("Local transcription model not loaded.")
                
                print("🎙 Starting local transcription...")
                audio_input, _ = librosa.load(filepath, sr=16000)
                transcription_result = transcriber(audio_input, return_timestamps=True)
                transcript_text = transcription_result.get('text', "Transcription failed.")
                print("✅ Local transcription finished.")

                # --- Step 2: Summarization using Google Gemini ---
                if transcript_text and "failed" not in transcript_text.lower():
                    print("🤖 Sending transcript to Gemini API for summarization...")
                    # ✅ Correct model name and API usage
                    model = genai.GenerativeModel('gemini-flash-latest')
                    prompt = f"""
                    You are a professional meeting assistant. Based on the following meeting transcript, please provide:
                    1. A concise, easy-to-read summary of the key discussion points and decisions.
                    2. A bulleted list of all action items.

                    Transcript:
                    ---
                    {transcript_text}
                    ---
                    """
                    response = model.generate_content(prompt)
                    summary_text = response.text.strip() if response.text else "No summary generated."
                    print("✅ Received summary from Gemini API.")

            except Exception as e:
                print(f"❌ Error during processing: {e}")
                transcript_text = f"An error occurred: {str(e)}"
                summary_text = "Processing failed."

            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

            return render_template('index.html', transcript=transcript_text, summary=summary_text)

    return render_template('index.html', transcript=None, summary=None)


if __name__ == '__main__':
    app.run(debug=True)
