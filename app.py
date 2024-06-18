from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("medium")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(audio_path)
    
    # Transcribe audio to text
    result = model.transcribe(audio_path)
    transcription = result["text"]
    
    os.remove(audio_path)  # Clean up the saved file
    
    return jsonify({"transcription": transcription})

@app.route('/translate', methods=['POST'])
def translate():
    if 'audio' not in request.files or 'target_language' not in request.form:
        return jsonify({"error": "Audio file and target language must be provided"}), 400

    audio_file = request.files['audio']
    target_language = request.form['target_language']
    audio_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(audio_path)
    
    # Transcribe audio to text
    result = model.transcribe(audio_path)
    transcription = result["text"]
    
    # Translate text to target language
    translation_result = model.translate(transcription, to_lang=target_language)
    translation = translation_result["text"]
    
    os.remove(audio_path)  # Clean up the saved file
    
    return jsonify({"transcription": transcription, "translation": translation})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9000)

