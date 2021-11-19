import os
import json

from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.detector.ner import NERDetector
from settings import SERVER_PORT, SERVER_HOST, LOCAL, UPLOAD_DIR

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)
ner_detector = NERDetector()


@app.route('/api/detection/', methods=['POST'])
def get_model_info():
    text_file = request.files['file']
    full_filename = secure_filename(text_file.filename)
    text_file_path = os.path.join(UPLOAD_DIR, full_filename)
    text_file.save(text_file_path)
    ner_results = ner_detector.run(txt_file_path=text_file_path)

    return json.dumps({"response": ner_results})


if __name__ == '__main__':
    if LOCAL:
        app.run(debug=True, host=SERVER_HOST, port=SERVER_PORT)
    else:
        app.run(debug=False, host=SERVER_HOST, port=SERVER_PORT)
