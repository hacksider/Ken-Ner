import os

from utils.file_tool import make_directory_if_not_exists

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = make_directory_if_not_exists(os.path.join(BASE_DIR, 'utils', 'model'))
UPLOAD_DIR = make_directory_if_not_exists(os.path.join(BASE_DIR, 'upload'))

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
LOCAL = True

JSON_FILE_PATH = "2015di.json"
EPOCHS = 200
