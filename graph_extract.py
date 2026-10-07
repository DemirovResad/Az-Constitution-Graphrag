"""
GraphRAG-ın 1-ci mərhələsi: hər chunk-dan (Konstitusiya maddəsindən)
entity-lər (Prezident, Milli Məclis, vətəndaş hüquqları və s.) və
onlar arasındakı əlaqələr LLM ilə çıxarılır.

Giriş: embeddings_checkpoint.jsonl (config.py-dəki CHECKPOINT_PATH) —
yalnız id+text istifadə olunur, embedding-lərə ehtiyac yoxdur.

Çıxış: graph_extraction_checkpoint.jsonl — hər sətir bir chunk-ın
çıxarılan entity/relation siyahısıdır. Checkpoint-lidir — artıq
emal olunmuş chunk-lar təkrar göndərilmir.

call_model.py-dəki get_model() və invoke_with_retry() eyni qaydada
istifadə olunur (rate-limit/429 idarəsi ordan gəlir).

İstifadə:
    python graph_extract.py
"""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

import unicodedata
from call_model import get_model, invoke_with_retry
from config import CHECKPOINT_PATH

GRAPH_CHECKPOINT_PATH = Path("graph_extraction_checkpoint.jsonl")

# Bundan qısa chunk-lar (məs. yalnız başlıq olan "I fəsil ...") LLM-ə
# göndərilmir — onlardan entity çıxmır, yalnız boş çağırışdır.
MIN_CHUNK_CHARS = 10

# Gleaning: chunk-dan ilk çıxarışdan sonra modelə "nəyisə qaçırmısan?"
# deyə əlavə sorğu göndərmək. 0 = söndürülü (hər chunk 1 dəfə göndərilir),
# 1 = hər chunk üçün 1 əlavə çağırış (xərc ~2 qat) və s.
# Kodu dəyişmədən .env-də və ya terminalda GLEANING_ROUNDS ilə verilə bilər.
# Model yeni heç nə tapmayanda dövr erkən dayanır.
GLEANING_ROUNDS = int(os.getenv("GLEANING_ROUNDS", "1"))

GLEANING_PROMPT = """\
Əvvəlki çıxarışda bu mətndən bəzi entity və əlaqələr qaçırılmış ola bilər.
Mətnə yenidən bax və YALNIZ əvvəl çıxarmadıqlarını ver. Əvvəlki
cavabdakıları təkrarlama. Əvvəlki qaydalar eynilə keçərlidir (yalnız
mətndə açıq yazılan, uydurma yox). Əlavə heç nə yoxdursa, boş siyahılar
qaytar.
"""


# ============================================================
# Struktur (Pydantic) — model bu formatda JSON qaytaracaq
# ============================================================
class Entity(BaseModel):
    name: str = Field(description="Entity-nin adı (məs. 'Prezident', 'Milli Məclis', 'söz azadlığı')")
    type: str = Field(description="Kateqoriya: INSTITUTION, ROLE, PERSON, RIGHT, CONCEPT, PROCEDURE və s.")
    description: str = Field(description="Bu chunk-a əsasən entity-nin qısa təsviri (1 cümlə)")


class Relation(BaseModel):
    source: str = Field(description="Əlaqənin başladığı entity adı")
    target: str = Field(description="Əlaqənin bitdiyi entity adı")
    relation: str = Field(description="Əlaqənin təsviri (məs. 'təyin edir', 'hüququna malikdir')")
    strength: int = Field(ge=1, le=10, description="Əlaqənin gücü/vacibliyi (1=zəif, 10=çox güclü)")


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


EXTRACTION_PROMPT_TEMPLATE = """\
Aşağıda Azərbaycan Respublikası Konstitusiyasından bir maddənin mətni verilib.
Bu mətndən bilik qrafı üçün ENTITY-lər və onlar arasındakı ƏLAQƏLƏRİ çıxar.

Entity nümunələri: dövlət orqanları (Prezident, Milli Məclis, Nazirlər
Kabineti, Konstitusiya Məhkəməsi, Ali Məhkəmə, bələdiyyələr), rollar
(vətəndaş, əcnəbi, deputat), hüquq/azadlıqlar (söz azadlığı, mülkiyyət
hüququ, seçki hüququ), prosedurlar (referendum, seçki, təyinat).

Əlaqə nümunələri: "təyin edir", "seçir", "səlahiyyətlidir", "hüququna
malikdir", "məsuliyyət daşıyır", "tabedir".

Yalnız BU MƏTNDƏ birbaşa göstərilən entity və əlaqələri çıxar, uydurma.
Mətndə heç bir aydın entity/əlaqə yoxdursa, boş siyahılar qaytar.

Qaydalar:
- Konkret şəxslərin (məs. dövlət xadimləri) tipi PERSON olsun, ROLE yox.
  ROLE yalnız vəzifə/status üçündür (Prezident, deputat, vətəndaş).
- Əlaqədəki source və target adları entity siyahısındakı adlarla DƏQİQ
  eyni olmalıdır. Əlaqədə işlətdiyin hər ad entity siyahısında da olsun.
- Eyni anlayışı hər yerdə eyni qısa adla yaz, mötərizəsiz və artıq
  sözsüz (məs. həmişə "referendum", "ümumxalq səsverməsi (referendum)"
  yox; həmişə "Prezident", "Azərbaycan Respublikasının Prezidenti" yox).
- Mətnin əvvəli və ya sonu kəsilmiş ola bilər (yarımçıq söz və ya
  cümlə). Yarımçıq hissədən entity və ya əlaqə çıxarma.

Maddə ID: {chunk_id}
Mətn:
{text}
"""


def load_chunks(checkpoint_path: Path = CHECKPOINT_PATH) -> list[dict]:
    chunks = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append({"id": row["id"], "text": row["text"]})
    return chunks


def _load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["chunk_id"])
                except Exception:
                    continue
    return done


def _append_result(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Gleaning
# ============================================================
def _norm(s: str) -> str:
    # graph_build._normalize ilə eyni məntiq (Azərbaycan hərfləri üçün)
    s = unicodedata.normalize("NFC", s.strip())
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return " ".join(s.split())


def _merge(base: ExtractionResult, extra: ExtractionResult) -> tuple[ExtractionResult, int]:
    """extra-dakı YENİ entity/relation-ları base-ə əlavə edir.
    Qaytarır: (birləşmiş nəticə, yeni əlavə olunanların sayı)."""
    seen_ents = {_norm(e.name) for e in base.entities}
    new_ents = []
    for e in extra.entities:
        k = _norm(e.name)
        if k in seen_ents:
            continue
        seen_ents.add(k)
        new_ents.append(e)

    seen_rels = {(_norm(r.source), _norm(r.target), _norm(r.relation)) for r in base.relations}
    new_rels = []
    for r in extra.relations:
        k = (_norm(r.source), _norm(r.target), _norm(r.relation))
        if k in seen_rels:
            continue
        seen_rels.add(k)
        new_rels.append(r)

    merged = ExtractionResult(
        entities=base.entities + new_ents,
        relations=base.relations + new_rels,
    )
    return merged, len(new_ents) + len(new_rels)


def extract_chunk(structured_model, chunk: dict) -> ExtractionResult:
    """Bir chunk üçün ilkin çıxarış + (varsa) gleaning dövrləri."""
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_id=chunk["id"], text=chunk["text"])

    # İlk çıxarış — burada xəta olsa yuxarı atılır, chunk skip olunur
    result = invoke_with_retry(structured_model, prompt)

    # Gleaning: əvvəlki cavabı söhbət tarixçəsi kimi verib "qaçırdığını tap" deyirik
    for round_no in range(1, GLEANING_ROUNDS + 1):
        messages = [
            HumanMessage(content=prompt),
            AIMessage(content=result.model_dump_json()),
            HumanMessage(content=GLEANING_PROMPT),
        ]
        try:
            extra = invoke_with_retry(structured_model, messages)
        except Exception as e:
            # Gleaning uğursuz olsa ilk nəticəni itirmirik
            print(f"    gleaning #{round_no} xəta, mövcud nəticə saxlanır: {str(e)[:100]}")
            break

        result, added = _merge(result, extra)
        if added == 0:  # model artıq yeni heç nə tapmır — dayan, əlavə xərc etmə
            break

    return result


def extract_all(chunks: list[dict], checkpoint_path: Path = GRAPH_CHECKPOINT_PATH) -> None:
    model = get_model()
    structured_model = model.with_structured_output(ExtractionResult)

    done_ids = _load_done_ids(checkpoint_path)
    pending = [
        c for c in chunks
        if c["id"] not in done_ids and len(c["text"].strip()) >= MIN_CHUNK_CHARS
    ]

    print("=" * 60)
    print("ENTITY/RELATION ÇIXARILMASI BAŞLAYIR")
    print("=" * 60)
    print(f"Toplam chunk:    {len(chunks)}")
    print(f"Artıq bitib:     {len(done_ids)}")
    print(f"Qalan:           {len(pending)}")
    print(f"Gleaning dövrü:  {GLEANING_ROUNDS}")
    print("=" * 60)

    for i, chunk in enumerate(pending, start=1):
        try:
            result = extract_chunk(structured_model, chunk)
        except Exception as e:
            print(f"[{i}/{len(pending)}] {chunk['id']} — xəta, skip edilir: {str(e)[:150]}")
            continue

        _append_result(checkpoint_path, {
            "chunk_id": chunk["id"],
            "entities": [e.model_dump() for e in result.entities],
            "relations": [r.model_dump() for r in result.relations],
        })
        print(f"[{i}/{len(pending)}] {chunk['id']} — {len(result.entities)} entity, {len(result.relations)} əlaqə")

    print("=" * 60)
    print(f"BİTDİ — nəticələr {checkpoint_path}-də")
    print("=" * 60)


if __name__ == "__main__":
    chunks = load_chunks()
    extract_all(chunks)