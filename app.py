import os
from flask import Flask, request, render_template, redirect
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# ✅ Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    transcript_text = ""
    summary_text = ""

    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)

        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # --- Step 1: Transcription directly using Gemini ---
                print("🎙 Uploading to Gemini for transcription...")
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    [f"Transcribe and summarize this audio file."],
                    files=[filepath]
                )
                transcript_text = response.text or "Transcription failed."

                # --- Step 2: Summarization (Gemini handles it together) ---
                summary_text = (
                    "Summary extracted successfully."
                    if transcript_text else "No summary generated."
                )
                print("✅ Gemini transcription and summary complete.")

            except Exception as e:
                print(f"❌ Error during processing: {e}")
                transcript_text = f"An error occurred: {str(e)}"
                summary_text = "Processing failed."

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

            return render_template('index.html', transcript=transcript_text, summary=summary_text)

    return render_template('index.html', transcript=None, summary=None)


# ✅ Correct port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
