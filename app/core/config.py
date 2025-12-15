import os
import logging
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path=".env")

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"
    MODEL_NAME = "gemini-1.5-flash"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SCHOOL_API_URL = os.getenv("SCHOOL_API_URL", "http://localhost:8080/api/common-data")

settings = Settings()

if not settings.GOOGLE_API_KEY:
    logging.warning("Thiếu GOOGLE_API_KEY. Chức năng chat sẽ không hoạt động.")
