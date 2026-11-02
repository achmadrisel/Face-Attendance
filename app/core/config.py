from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    FACE_DETECTION: str = os.getenv('FACE_DETECTION', './models/scrfd_0.5G.onnx')
    FACE_RECOGNITION: str = os.getenv('FACE_RECOGNITION', './models/glintr100.onnx')
    DATASET_PATH: str = os.getenv('DATASET_PATH', './facedata')

    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', 5432))
    DB_NAME: str = os.getenv('DB_NAME', 'attendance_db')
    DB_USER: str = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'postgres')
    DB_URL: str = os.getenv('DB_URL', f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    class config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()