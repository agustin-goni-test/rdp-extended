import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Config parameters
LOG_DIR = os.getenv("LOG_DIR", "log")
LEVEL = logging.DEBUG if os.getenv("DEBUG") == "true" else logging.INFO

# Logger class, singleton implementation
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logging()
        return cls._instance
            

    def _setup_logging(self):
            # Crear archivo de log si no existe
        os.makedirs(LOG_DIR, exist_ok=True)

        # Crear archivo de log con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{LOG_DIR}/process_{timestamp}.log"

        logging.basicConfig(
            level=LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ],
            force=True
        )

        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warninig(self, message):
        self.logger.warning(message)

    def ddebug(self, message):
        self.logger.debug(message)
    
