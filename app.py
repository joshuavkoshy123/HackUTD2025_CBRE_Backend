from flask import Flask, jsonify, request
from flask_cors import CORS
import nemotron
import pdf_conversion
import os

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")  # allows requests from your React frontend

# Folder where uploads are stored
UPLOAD_FOLDER = os.path.join(os.getcwd(), "pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

@app.route('/api/hello')
def hello():
    response = nemotron.nemotron()
    return jsonify(message=response)

image_paths = []
@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file to uploads directory
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    trimmed = file_path.split("backend\\", 1)[1]
    print(trimmed)

    image_paths = pdf_conversion.pdf_to_images(trimmed)
    return jsonify(message=image_paths)

# gives a summary of each uploaded file
@app.route('/api/summary')
def summary():
    response = pdf_conversion.send_to_nvidia_model(image_paths, "Please give a summary of this document.")
    return jsonify(message=response)

@app.route('/api/query', methods=['POST'])
def query():
    query = request
    response = pdf_conversion.send_to_nvidia_model(image_paths, request)
    return jsonify(message=response)

if __name__ == '__main__':
    app.run(debug=True)
