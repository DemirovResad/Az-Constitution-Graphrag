"""
GraphRAG-ın summarize_descriptions mərhələsi: bir entity/əlaqənin bir neçə
chunk-dan gələn təsvirləri (graph_build.py onları ' | ' ilə birləşdirir) LLM
ilə tək, ardıcıl təsvirə çevrilir.

Giriş:  entity_graph.json
Çıxış:  eyni fayl (description sahələri yenilənir)
Cache:  description_summaries.jsonl — açar: node/edge + təsvirlərin hash-i.
        graph_build.py-ni təkrar işlətsən də, təsvirlər dəyişməyibsə LLM
        yenidən çağırılmır.

Ardıcıllıq: graph_build.py -> graph_summarize.py -> community_report.py

İstifadə:
    python graph_summarize.py
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from call_model import get_model, invoke_with_retry
from GraphRAG.graph_build import GRAPH_OUTPUT_PATH, load_graph, save_graph
from logging_config import get_logger

logger = get_logger(__name__)

SUMMARY_CACHE_PATH = Path("description_summaries.jsonl")
SEPARATOR = " | "  # graph_build.py-dəki birləşdirmə ilə eyni olmalıdır
MAX_DESCRIPTIONS = 20


class DescriptionSummary(BaseModel):
    description: str = Field(description="Bütün təsvirləri birləşdirən tək, ardıcıl təsvir")


SUMMARY_PROMPT_TEMPLATE = """\
Sən aşağıdakı məlumatın hərtərəfli xülasəsini hazırlayan köməkçisən.
Bir entity (və ya iki entity arasındakı əlaqə) üçün Azərbaycan
Konstitusiyasının müxtəlif hissələrindən çıxarılmış bir neçə təsvir verilib.

Bütün təsvirlərdəki məlumatı birləşdirən TƏK bir təsvir yaz.
- Təsvirlər bir-biri ilə ziddiyyət təşkil edərsə, ziddiyyəti həll et və
  ardıcıl bir xülasə ver.
- Üçüncü şəxsdə yaz.
- Kontekst üçün entity adlarını daxil et.
- Yalnız verilən təsvirlərə əsaslan, uydurma əlavə etmə.

Mövzu: {subject}
Təsvirlər:
{descriptions}
"""


def _hash(subject: str, descs: list[str]) -> str:
    raw = json.dumps([subject] + descs, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _load_cache(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    cache[row["id"]] = row
                except Exception:
                    continue
    return cache


def _process(structured_model, cache: dict, cache_path: Path, item_id: str, subject: str, data: dict) -> bool:
    """data['description']-i yerində yeniləyir. LLM çağırılıbsa True qaytarır."""
    descs = [d.strip() for d in data["description"].split(SEPARATOR) if d.strip()]
    if len(descs) < 2:
        return False
    descs = descs[:MAX_DESCRIPTIONS]
    h = _hash(subject, descs)

    cached = cache.get(item_id)
    if cached and cached["hash"] == h:
        data["description"] = cached["description"]
        return False

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        subject=subject, descriptions="\n".join(f"- {d}" for d in descs),
    )
    try:
        result = invoke_with_retry(structured_model, prompt)
    except Exception as e:
        logger.error("%s — xəta, orijinal təsvir saxlanır: %s", subject, str(e)[:150])
        return False

    data["description"] = result.description
    row = {"id": item_id, "hash": h, "description": result.description}
    cache[item_id] = row
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def summarize_all(cache_path: Path = SUMMARY_CACHE_PATH) -> None:
    graph = load_graph(GRAPH_OUTPUT_PATH)
    cache = _load_cache(cache_path)

    model = get_model()
    structured_model = model.with_structured_output(DescriptionSummary)

    calls = 0

    for key, data in graph.nodes(data=True):
        if _process(structured_model, cache, cache_path, f"node:{key}", data["name"], data):
            calls += 1
            logger.info("entity xülasələndi: %s", data["name"])

    for u, v, data in graph.edges(data=True):
        subject = f"{graph.nodes[u]['name']} → {graph.nodes[v]['name']}"
        if _process(structured_model, cache, cache_path, f"edge:{u}\t{v}", subject, data):
            calls += 1
            logger.info("əlaqə xülasələndi: %s", subject)

    save_graph(graph, GRAPH_OUTPUT_PATH)
    logger.info("BİTDİ — %d yeni LLM çağırışı, qraf %s-də yeniləndi", calls, GRAPH_OUTPUT_PATH)


if __name__ == "__main__":
    summarize_all()
    