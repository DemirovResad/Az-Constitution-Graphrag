import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from logging_config import get_logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

logger = get_logger(__name__)


# ============================================================
# Konfiqurasiya
# ============================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
OUTPUT_DIM = int(os.getenv("EMBED_OUTPUT_DIM", "768"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))

# Defolt dəyərlər community-nin bildirdiyi free-tier rəqəmləridir.
# AI Studio-da (Settings -> Usage and billing) öz layihənə uyğun yoxla.
RPM_LIMIT = int(os.getenv("EMBED_RPM_LIMIT", "100"))
TPM_LIMIT = int(os.getenv("EMBED_TPM_LIMIT", "30000"))
RPD_LIMIT = int(os.getenv("EMBED_RPD_LIMIT", "1000"))

_raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

CHECKPOINT_PATH = Path(os.getenv("EMBED_CHECKPOINT_PATH", "embeddings_checkpoint.jsonl"))
PDF_PATH = os.getenv("PDF_PATH", "")

_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s")
_DAILY_QUOTA_PATTERN = re.compile(r"per[\s_-]?day|daily|PerDay", re.IGNORECASE)


# ============================================================
# Xətalar
# ============================================================
class DailyQuotaExceeded(Exception):
    """Cari API key-in günlük (RPD) limiti bitdikdə atılır."""


# ============================================================
# Bir neçə API key arasında keçid
# ============================================================
class KeyPool:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError(
                "Heç bir API key tapılmadı. .env-də GEMINI_API_KEYS "
                "(vergüllə ayrılmış) və ya GOOGLE_API_KEY təyin et."
            )
        self.keys = keys
        self.index = 0
        self.client = genai.Client(api_key=self.keys[self.index])

    @property
    def current_key_masked(self) -> str:
        k = self.keys[self.index]
        return f"{k[:6]}...{k[-4:]}" if len(k) > 10 else "***"

    def rotate(self) -> bool:
        """Növbəti key-ə keçir. Uğurlu olsa True, key qalmayıbsa False qaytarır."""
        if self.index + 1 >= len(self.keys):
            return False
        self.index += 1
        self.client = genai.Client(api_key=self.keys[self.index])
        logger.info(
            "KeyPool: növbəti key-ə keçilir (%d/%d): %s",
            self.index + 1, len(self.keys), self.current_key_masked,
        )
        return True


# ============================================================
# Sadə RPM / TPM / RPD limiter
# ============================================================
class RateLimiter:
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._minute_start = time.time()
        self._req_this_minute = 0
        self._tok_this_minute = 0
        self._day_start = time.time()
        self._req_today = 0

    def _reset_if_needed(self):
        now = time.time()
        if now - self._minute_start >= 60:
            self._minute_start = now
            self._req_this_minute = 0
            self._tok_this_minute = 0
        if now - self._day_start >= 86400:
            self._day_start = now
            self._req_today = 0

    def wait_if_needed(self, est_tokens: int):
        self._reset_if_needed()
        while (
            self._req_this_minute + 1 > self.rpm
            or self._tok_this_minute + est_tokens > self.tpm
        ):
            sleep_for = max(1.0, 60 - (time.time() - self._minute_start))
            logger.warning(
                "RateLimiter: RPM/TPM limitinə yaxın — %.0fs gözlənilir...",
                sleep_for,
            )
            time.sleep(sleep_for)
            self._reset_if_needed()
        if self._req_today + 1 > self.rpd:
            raise DailyQuotaExceeded()

    def record(self, tokens_used: int):
        self._req_this_minute += 1
        self._tok_this_minute += tokens_used
        self._req_today += 1


def estimate_tokens(text: str) -> int:
    """Kobud qiymətləndirmə: ~4 simvol = 1 token."""
    return max(1, len(text) // 4)

