"""
Mərkəzi loglama konfiqurasiyası.

Bütün modullar bunu istifadə edir:
    from logging_config import get_logger
    logger = get_logger(__name__)

`print()` əvəzinə `logger.info/warning/error` — beləliklə hər sətirdə
timestamp + səviyyə (INFO/WARNING/ERROR) olur və LOG_LEVEL ilə filtərlənə
bilir (məs. production-da yalnız WARNING+ görmək istəsən).

.env dəyişənləri (opsional):
    LOG_LEVEL=INFO        # DEBUG / INFO / WARNING / ERROR
    LOG_FILE=app.log      # defolt: app.log (Streamlit UI-dəki log paneli
                           # bunu oxuyur — boş sətir versən, fayla yazılmır)
    MONGO_URI=mongodb://localhost:27017   # boş buraxılsa, Mongo-ya yazılmır
    MONGO_DB=agent_logs
    MONGO_COLLECTION=logs
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "app.log")

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "agent_logs")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "logs")

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class MongoHandler(logging.Handler):
    """Hər log sətrini LibreChat-in Mongo-suna toxunmadan, ayrı bir
    DB/collection-a (.env-dəki MONGO_DB/MONGO_COLLECTION) yazır.
    Mongo əlçatan olmasa belə tətbiqi çökdürməsin deyə emit() içində
    xəta udulur — yalnız console-a xəbərdarlıq yazılır."""

    def __init__(self, uri: str, db_name: str, collection_name: str):
        super().__init__()
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._collection = self._client[db_name][collection_name]
        self._warned = False

    def emit(self, record: logging.LogRecord):
        try:
            self._collection.insert_one({
                "timestamp": datetime.now(timezone.utc),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:
            if not self._warned:
                self._warned = True
                logging.getLogger(__name__).warning(
                    "MongoHandler: Mongo-ya yazıla bilmədi, bundan sonrakı "
                    "xətalar susdurulur (yalnız fayl/console loglanacaq).",
                    exc_info=True,
                )


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if MONGO_URI:
        try:
            mongo_handler = MongoHandler(MONGO_URI, MONGO_DB, MONGO_COLLECTION)
            root.addHandler(mongo_handler)
        except Exception:
            logging.getLogger(__name__).warning(
                "MongoHandler quraşdırıla bilmədi — yalnız fayl/console loglanacaq.",
                exc_info=True,
            )

    # Üçüncü tərəf kitabxanaların həddindən artıq DEBUG/INFO log-larını sakitləşdir
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Hər modulun başında: logger = get_logger(__name__)"""
    _configure_once()
    return logging.getLogger(name)


def truncate(text: str, max_len: int = 200) -> str:
    """Log sətirlərində uzun mətnləri (sual, cavab) qısaltmaq üçün köməkçi."""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_len else text[:max_len] + "…"