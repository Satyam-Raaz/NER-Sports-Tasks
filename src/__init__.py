import logging 
import os
from src.constant import ARTIFACTS_DIR, LOGS_DIR, LOGS_FILE_NAME


logs_path = os.path.join(os.getcwd(), ARTIFACTS_DIR, LOGS_DIR)
os.makedirs(logs_path, exist_ok=True)

LOGS_FILE_PATH = os.path.join(logs_path, LOGS_FILE_NAME) 

logging.basicConfig(
    filename=LOGS_FILE_PATH,
    level=logging.DEBUG,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
)