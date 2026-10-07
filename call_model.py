import os
import re
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from logging_config import get_logger

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)

logger = get_logger(__name__)


DEFAULT_PROVIDER = "google_genai"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8192


@lru_cache(maxsize=None)
def get_model(
    model_name: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> BaseChatModel:

    provider = provider or os.getenv(
        "MODEL_PROVIDER",
        DEFAULT_PROVIDER
    )

    model_name = model_name or os.getenv(
        "MODEL_NAME",
        DEFAULT_MODEL
    )

    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv(
            "MODEL_TEMPERATURE",
            DEFAULT_TEMPERATURE
        ))
    )

    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv(
            "MODEL_MAX_TOKENS",
            DEFAULT_MAX_TOKENS
        ))
    )

    base_url = base_url or os.getenv(
        "MODEL_BASE_URL",
        ""
    )

    
    if base_url and provider != "openai":
        logger.warning(
            "MODEL_BASE_URL təyin olunub, amma provider '%s' üçün nəzərə "
            "alınmır (yalnız provider=openai üçün istifadə olunur). Native "
            "'%s' inteqrasiyası öz endpoint-ini özü bilir — .env-dən "
            "MODEL_BASE_URL sətrini silə bilərsən.",
            provider, provider,
        )
        base_url = ""

    logger.info(
        "Model konfiqurasiyası — provider=%s model=%s base_url=%s "
        "temperature=%s max_tokens=%s",
        provider, model_name, base_url or "default", temperature, max_tokens,
    )

    kwargs = {
        "model": model_name,
        "model_provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Yalnız URL veriləndə (və provider=openai olanda) əlavə et
    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(**kwargs)


# ============================================================
# Langfuse — bütün pipeline-ı (plan → search → extract) tək trace
# kimi izləmək üçün. .env-də bunlar olmalıdır:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
#   LANGFUSE_BASE_URL=https://cloud.langfuse.com   (🇪🇺 EU; 🇺🇸 US üçün https://us.cloud.langfuse.com, ya da öz self-host ünvanın)
# ============================================================
_langfuse_client = None
_langfuse_handler = None


def get_langfuse_client():
    """Langfuse client-i (env-dən konfiqurasiya oxuyur) tək dəfə yaradır."""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = get_client()
    return _langfuse_client


def get_langfuse_handler() -> CallbackHandler:
    """LangChain/LangGraph .invoke()-a config={'callbacks': [...]} kimi
    ötürüləcək Langfuse callback handler-i qaytarır."""
    global _langfuse_handler
    if _langfuse_handler is None:
        get_langfuse_client()  # client əvvəlcə init olunmalıdır
        _langfuse_handler = CallbackHandler()
    return _langfuse_handler


# ============================================================
# Rate-limit (429 RESOURCE_EXHAUSTED) retry — Google-un pulsuz tier-i
# dəqiqəlik token limitinə çatanda "retryDelay"-i mesajdan oxuyub
# gözləyir, sonra yenidən cəhd edir. Bütün structured_model.invoke()
# çağırışları bunun üzərindən keçməlidir (plan.py, extract.py).
# ============================================================
_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s")


def invoke_with_retry(structured_model, prompt: str, config=None, max_retries: int = 3):
    """structured_model.invoke()-u işə salır; 429/RESOURCE_EXHAUSTED xətası
    gələrsə Google-un tövsiyə etdiyi retryDelay qədər (+ kiçik buffer)
    gözləyib yenidən cəhd edir. Digər xətalarda birbaşa raise edir."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return structured_model.invoke(prompt, config=config)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str
            if not is_rate_limit or attempt == max_retries:
                raise
            last_error = e
            match = _RETRY_DELAY_PATTERN.search(error_str)
            wait_seconds = float(match.group(1)) + 3 if match else 30.0
            logger.warning(
                "invoke_with_retry: rate limit (429) — %.0fs gözlənilir "
                "(cəhd %d/%d)...",
                wait_seconds, attempt + 1, max_retries,
            )
            time.sleep(wait_seconds)
    if last_error:
        raise last_error