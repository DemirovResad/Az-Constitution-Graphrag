"""
Gemini Embedding modulu — böyük PDF/mətn korpusunu (məs. Konstitusiya
maddələri) chunk-lara bölüb gemini-embedding-001 ilə embed edir.

Ortaq konfiqurasiya (model adı, API key-lər, KeyPool, RateLimiter)
config.py-dədir — retriever.py da eyni modulu istifadə edir ki, iki
yerdə təkrarlanmasın.

Xüsusiyyətlər:
  - Checkpoint: hər embed olunan chunk dərhal .jsonl faylına yazılır.
    Skript kəsilsə/rate-limit-ə düşsə, yenidən işə salanda artıq
    bitmiş id-lər avtomatik skip olunur.
  - Rate limit idarəsi: RPM/TPM-ə yaxınlaşanda gözləyir; 429 xətasında
    Google-un tövsiyə etdiyi retryDelay-i oxuyub bir daha cəhd edir.
  - Çox API key: .env-də GEMINI_API_KEYS="key1,key2,key3" kimi verilsə,
    bir key-in günlük limiti (RPD) bitəndə avtomatik növbətiyə keçir.
  - Batch: bir sorğuda bir neçə mətni birlikdə embed edir.

.env nümunəsi config.py-nin başında yazılıb.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pdfplumber
from google.genai import types

from config import (
    API_KEYS,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EMBED_MODEL,
    OUTPUT_DIM,
    PDF_PATH,
    RPD_LIMIT,
    RPM_LIMIT,
    TPM_LIMIT,
    _DAILY_QUOTA_PATTERN,
    _RETRY_DELAY_PATTERN,
    DailyQuotaExceeded,
    KeyPool,
    RateLimiter,
    estimate_tokens,
)
from logging_config import get_logger

logger = get_logger(__name__)

# Maddələri ayırmaq üçün pattern — Azərbaycan Konstitusiyasında maddələr
# "Maddə 1." formatında başlayır. Sənin PDF-in fərqli formatdadırsa
# (məs. "Maddə I." və ya nömrə fərqli yerdədirsə) bu regex-i dəyiş.
ARTICLE_PATTERN = re.compile(r"(Maddə\s+\d+[\.\-–])", re.IGNORECASE)

# Bir chunk-ın maksimum uzunluğu (simvol). Maddə bundan qısadırsa TAM maddə
# tək chunk kimi qalır (defolt və üstünlük verilən hal — maddə bölünməsin ki,
# kontekst itməsin). Yalnız bundan uzun maddələr (məs. Prezidentin səlahiyyətləri,
# torpaq/mülkiyyət maddələri) aşağıdakı üsulla hissələrə bölünür.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1500"))

# Maddə daxilindəki bəndlər Konstitusiyada Roma rəqəmi ilə başlayır: "I.", "II." və s.
# Uzun maddəni bölərkən ƏVVƏLCƏ bu sərhədlərdən istifadə olunur (mətni ortasından
# kəsmək əvəzinə) — köhnə sadə simvol-sayı bölməsi bunu etmirdi və nəticədə bəzi
# chunk-lar cümlənin ortasından kəsilirdi, hətta iki fərqli maddənin mətni qarışırdı.
_CLAUSE_PATTERN = re.compile(r"(?m)(^[IVXLCDM]{1,6}\.\s)")
# Bəndlərə görə bölmək mümkün olmayanda (bənd yoxdur, ya da tək bənd özü
# MAX_CHUNK_CHARS-dan uzundur) son çarə: cümlə sərhədləri.
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-ZƏÖÜŞÇĞİ0-9])")


# ============================================================
# Checkpoint (nəticələrin saxlanması və bərpası)
# ============================================================
def _load_done_ids(checkpoint_path: Path) -> set:
    done = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def _append_result(checkpoint_path: Path, row: dict):
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# PDF-dən mətn çıxarma
# ============================================================
def extract_pdf_text(pdf_path: str) -> str:
    """PDF-in bütün səhifələrindən mətni çıxarıb tək string kimi qaytarır."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            else:
                logger.warning(
                    "extract_pdf_text: səhifə %d-dən mətn çıxmadı (skan ola bilər).",
                    page_num,
                )
    return full_text


def _pack_sentences(header: str, sentences: list[str], max_chars: int) -> list[str]:
    """Cümlələri (və ya bəndləri) ardıcıl yığır, MAX_CHUNK_CHARS-a çatanda yeni
    hissə başlayır. Hər hissənin əvvəlinə `header` (məs. 'Maddə 29. Mülkiyyət
    hüququ') qoyulur ki, hissə tək başına (kontekstsiz) oxunanda da hansı
    maddəyə aid olduğu bəlli olsun."""
    parts: list[str] = []
    current = header
    current_len = len(header)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        extra = len(sent) + 1
        if current_len > len(header) and current_len + extra > max_chars:
            parts.append(current.strip())
            current = header
            current_len = len(header)
        current += "\n" + sent if current_len == len(header) else " " + sent
        current_len += extra
    if current.strip() != header.strip():
        parts.append(current.strip())
    return parts or [header.strip()]


def _split_long_article(header: str, body: str, max_chars: int) -> list[str]:
    """Uzun maddəni hissələrə bölür: əvvəlcə Roma rəqəmli bəndlərə (I., II., ...)
    görə, tapılmasa (və ya tək bənd özü max_chars-dan uzundursa) cümlə
    sərhədlərinə görə. Nəticədə heç bir hissə cümlənin ortasından kəsilmir."""
    clause_parts = _CLAUSE_PATTERN.split(body)
    clauses = [
        (clause_parts[i] + clause_parts[i + 1]).strip()
        for i in range(1, len(clause_parts) - 1, 2)
    ] if len(clause_parts) > 2 else []
    if clause_parts and clause_parts[0].strip() and not clauses:
        clauses = [clause_parts[0].strip()]  # bənd markeri heç yoxdur — bütün body bir "bənd"

    units: list[str] = []
    for clause in clauses:
        if len(clause) <= max_chars:
            units.append(clause)
        else:  # tək bənd özü çox uzundur — cümlələrə böl
            units.extend(s for s in _SENTENCE_PATTERN.split(clause) if s.strip())

    return _pack_sentences(header, units, max_chars)


def chunk_by_article(
    text: str,
    pattern: re.Pattern = ARTICLE_PATTERN,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """
    Mətni 'Maddə N.' başlıqlarına görə chunk-lara bölür — DEFOLT olaraq HƏR
    MADDƏ TAM bir chunk-dır (id: 'madde_N'), kontekst itməsin deyə.

    Yalnız maddə `max_chunk_chars`-dan uzundursa (məs. Prezidentin
    səlahiyyətləri), həmin maddə bəndlərinə görə (I., II., ... — Konstitusiyanın
    öz strukturu) hissələrə bölünür: 'madde_N_p1', 'madde_N_p2', ... Hər hissənin
    əvvəlində maddənin başlığı təkrarlanır ki, hissə tək başına oxunanda da
    hansı maddəyə aid olduğu bilinsin. Bənd markeri yoxdursa (nadir hal) və ya
    tək bənd özü uzundursa, son çarə olaraq cümlə sərhədlərinə görə bölünür —
    heç bir hal mətni sözün/cümlənin ortasından kəsmir.

    Əgər sənin PDF-in fərqli formatdadırsa (məs. fəsillərə görə bölünübsə,
    ya da "Maddə" sözü yoxdursa), modulun yuxarısındakı ARTICLE_PATTERN-i
    öz formatına uyğun dəyiş.
    """
    parts = pattern.split(text)
    chunks = []

    if parts and parts[0].strip():
        chunks.append({"id": "preambula", "text": parts[0].strip()})

    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip()
        match = re.search(r"\d+", header)
        article_num = match.group() if match else str(len(chunks) + 1)
        full_text = f"{header} {body}".strip()

        if len(full_text) <= max_chunk_chars:
            chunks.append({"id": f"madde_{article_num}", "text": full_text})
        else:
            sub_parts = _split_long_article(header, body, max_chunk_chars)
            for p_no, sub_text in enumerate(sub_parts, start=1):
                chunks.append({"id": f"madde_{article_num}_p{p_no}", "text": sub_text})
            logger.info(
                "chunk_by_article: Maddə %s uzundur (%d simvol) — %d hissəyə bölündü",
                article_num, len(full_text), len(sub_parts),
            )
        i += 2

    return chunks


# ============================================================
# Əsas funksiya
# ============================================================
def embed_chunks(
    chunks: list[dict],
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_retries_per_batch: int = 5,
) -> None:
    """
    chunks: [{"id": "madde_1", "text": "..."}, ...] formasında siyahı.
    Hər id unikal olmalıdır — checkpoint bərpası bunun üzərində işləyir.

    Artıq embeddinq olunmuş chunk-lar avtomatik skip edilir (checkpoint
    faylında olan id-lərə görə) — yəni bu funksiyanı təkrar çağırmaq
    mövcud .jsonl-i sıfırdan yazmır, üstünə əlavə edir.

    Nəticələr checkpoint_path-a append olunur:
      {"id": ..., "text": ..., "embedding": [float, ...]}
    """
    pool = KeyPool(API_KEYS)
    limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)

    done_ids = _load_done_ids(checkpoint_path)
    pending = [c for c in chunks if c["id"] not in done_ids]

    logger.info(
        "Embedding başlayır — model=%s toplam=%d artıq_bitib=%d qalan=%d api_key_sayı=%d",
        EMBED_MODEL, len(chunks), len(done_ids), len(pending), len(API_KEYS),
    )

    if not pending:
        logger.info("Bütün chunk-lar artıq embed olunub, edilməli iş yoxdur.")
        return

    i = 0
    while i < len(pending):
        batch = pending[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        est_tokens = sum(estimate_tokens(t) for t in texts)

        try:
            limiter.wait_if_needed(est_tokens)
        except DailyQuotaExceeded:
            if pool.rotate():
                limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)
                continue
            logger.error(
                "Bütün key-lərin günlük (RPD) limiti bitdi. Checkpoint "
                "saxlanıldı (%s) — sabah və ya yeni key ilə skripti "
                "yenidən işə sal.",
                checkpoint_path,
            )
            return

        attempt = 0
        while True:
            try:
                result = pool.client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=OUTPUT_DIM,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                for chunk, emb in zip(batch, result.embeddings):
                    _append_result(
                        checkpoint_path,
                        {"id": chunk["id"], "text": chunk["text"], "embedding": emb.values},
                    )
                limiter.record(est_tokens)
                i += BATCH_SIZE
                logger.info("[%d/%d] embed olundu.", min(i, len(pending)), len(pending))
                break

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str
                is_daily = is_rate_limit and _DAILY_QUOTA_PATTERN.search(error_str)

                if is_daily:
                    logger.warning("[%s] Günlük limit bitdi.", pool.current_key_masked)
                    if pool.rotate():
                        limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)
                        continue
                    logger.error(
                        "Bütün key-lər bitib. Checkpoint saxlanıldı — "
                        "sabah davam etdirə bilərsən."
                    )
                    return

                if is_rate_limit and attempt < max_retries_per_batch:
                    attempt += 1
                    match = _RETRY_DELAY_PATTERN.search(error_str)
                    wait_seconds = float(match.group(1)) + 3 if match else 30.0
                    logger.warning(
                        "Rate limit (429) — %.0fs gözlənilir (cəhd %d/%d)...",
                        wait_seconds, attempt, max_retries_per_batch,
                    )
                    time.sleep(wait_seconds)
                    continue

                logger.error(
                    "Gözlənilməz xəta, bu batch skip edilir: %s",
                    error_str[:200],
                )
                i += BATCH_SIZE
                break

    logger.info("BİTDİ — bütün chunk-lar embed olundu (%s)", checkpoint_path)


# ============================================================
# CLI: python embed_gemini.py --pdf konstitusiya.pdf
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF-i chunk edib Gemini ilə embed edir.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=PDF_PATH,
        help="Embed olunacaq PDF-in yolu (.env-də PDF_PATH ilə də verilə bilər).",
    )
    args = parser.parse_args()

    if not args.pdf:
        raise SystemExit(
            "PDF yolu verilməyib. --pdf konstitusiya.pdf ilə çağır, "
            "ya da .env-də PDF_PATH=... təyin et."
        )

    print(f"PDF oxunur: {args.pdf}")
    raw_text = extract_pdf_text(args.pdf)
    print(f"Çıxarılan simvol sayı: {len(raw_text)}")

    chunks = chunk_by_article(raw_text)
    print(f"Tapılan chunk sayı: {len(chunks)}")
    if chunks:
        print(f"Nümunə (ilk chunk, ilk 200 simvol): {chunks[0]['text'][:200]}")

    embed_chunks(chunks)