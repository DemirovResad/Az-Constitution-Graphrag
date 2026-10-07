"""
GraphRAG-ın community report mərhələsi: graph_build.py-nin təyin etdiyi hər
community üçün (Louvain klasterləri) LLM bir xülasə hesabat yazır — title +
summary. Bu, Global Search-in (map-reduce) əsas girişi olacaq, local
search-ə də əlavə kontekst kimi qoşula bilər.

Giriş:  entity_graph.json (graph_build.py-nin çıxışı, community atributu ilə)
Çıxış:  community_reports.jsonl — hər sətir bir community-nin hesabatı:
        {"community_id", "title", "summary", "size", "entity_keys"}
        Checkpoint-lidir — artıq hesabatı yazılmış community-lər skip olunur.

call_model.py-dəki get_model() və invoke_with_retry() ilə eyni rate-limit
idarəsi istifadə olunur.

İstifadə:
    python community_report.py
    python community_report.py --min-size 2   # 1 node-luq community-ləri keç
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from call_model import get_model, invoke_with_retry
from GraphRAG.graph_build import GRAPH_OUTPUT_PATH, load_graph
from logging_config import get_logger

logger = get_logger(__name__)

REPORTS_PATH = Path(__file__).resolve().parent / "community_reports.jsonl"

# Bir community-dəki entity/əlaqə sayı bundan çoxdursa, prompt həddindən
# artıq böyüməsin deyə yalnız ilk MAX_ITEMS-i (əlaqə sayına görə ən
# "mərkəzi" entity-lər əvvəldə) göndəririk.
MAX_ITEMS = 40


# deep_agent.py-dəki _ID_PATTERN/_article_of ilə eyni məntiq — burada təkrarlanır
# ki, community_report.py deep_agent.py-dən asılı olmasın (dairəvi import olmasın).
_ID_PATTERN = re.compile(r"^(?:.*__)?(madde_(\d+)|preambula)(?:_p(\d+))?$")


def _article_label(chunk_id: str) -> str | None:
    m = _ID_PATTERN.match(chunk_id)
    if not m:
        return None
    return "Preambula" if m.group(1) == "preambula" else f"Maddə {m.group(2)}"


def _articles_of_community(graph, members: list[str]) -> list[str]:
    """Community-dəki entity-lərin mənbə chunk-larından (LLM-dən DEYİL, birbaşa
    qrafın özündən) maddə siyahısını çıxarır — halüsinasiya riski yoxdur."""
    labels = set()
    for key in members:
        for cid in graph.nodes[key].get("source_chunks", []):
            label = _article_label(cid)
            if label:
                labels.add(label)

    def sort_key(label: str) -> int:
        return 0 if label == "Preambula" else int(label.split()[1])

    return sorted(labels, key=sort_key)


class CommunityReport(BaseModel):
    title: str = Field(description="Community-ni 3-6 sözlə ümumiləşdirən qısa başlıq")
    summary: str = Field(description="Community-nin əhatə etdiyi mövzunu izah edən 3-5 cümləlik xülasə")


REPORT_PROMPT_TEMPLATE = """\
Aşağıda bir bilik qrafının bir hissəsi (community) verilib — bir-biri ilə
sıx bağlı entity-lər və aralarındakı əlaqələr. Bu, Azərbaycan Respublikası
Konstitusiyasından avtomatik çıxarılıb.

Bu community-ni ÜMUMİLƏŞDİRƏN bir başlıq və xülasə yaz. Xülasə yalnız
aşağıda verilən entity/əlaqələrə əsaslansın, uydurma əlavə etmə.

ENTITY-LƏR:
{entities}

ƏLAQƏLƏR:
{relations}
"""


def _format_community(graph, members: list[str]) -> tuple[str, str]:
    member_set = set(members)
    sorted_members = sorted(members, key=lambda k: -graph.degree[k])[:MAX_ITEMS]

    ent_lines = [
        f"- {graph.nodes[k]['name']} ({graph.nodes[k]['type']}): {graph.nodes[k]['description'][:200]}"
        for k in sorted_members
    ]

    rel_lines = []
    for u, v, d in graph.edges(data=True):
        if u in member_set and v in member_set:
            rel_lines.append(f"- {graph.nodes[u]['name']} → {graph.nodes[v]['name']}: {d['description'][:150]}")
        if len(rel_lines) >= MAX_ITEMS:
            break

    return "\n".join(ent_lines), "\n".join(rel_lines) if rel_lines else "(bu community daxilində əlaqə yoxdur)"


def _load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["community_id"])
                    except Exception:
                        continue
    return done


def build_all_reports(min_size: int = 1, checkpoint_path: Path = REPORTS_PATH) -> None:
    graph = load_graph(GRAPH_OUTPUT_PATH)

    groups: dict[int, list[str]] = defaultdict(list)
    for key, data in graph.nodes(data=True):
        groups[data["community"]].append(key)

    done_ids = _load_done_ids(checkpoint_path)
    pending = {cid: members for cid, members in groups.items()
               if cid not in done_ids and len(members) >= min_size}

    logger.info(
        "community report — toplam=%d artıq_bitib=%d qalan=%d (min_size=%d)",
        len(groups), len(done_ids), len(pending), min_size,
    )

    model = get_model()
    structured_model = model.with_structured_output(CommunityReport)

    for i, (cid, members) in enumerate(pending.items(), start=1):
        entities_text, relations_text = _format_community(graph, members)
        prompt = REPORT_PROMPT_TEMPLATE.format(entities=entities_text, relations=relations_text)

        try:
            result = invoke_with_retry(structured_model, prompt)
        except Exception as e:
            logger.error("community %d — xəta, skip edilir: %s", cid, str(e)[:150])
            continue

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "community_id": cid,
                "title": result.title,
                "summary": result.summary,
                "size": len(members),
                "entity_keys": members,
                "articles": _articles_of_community(graph, members),
            }, ensure_ascii=False) + "\n")

        logger.info("[%d/%d] community %d (%d entity) — %s", i, len(pending), cid, len(members), result.title)

    logger.info("BİTDİ — nəticələr %s-də", checkpoint_path)


def load_community_reports(path: Path = REPORTS_PATH) -> dict[int, dict]:
    """graph_query.py üçün: community_id -> {"title", "summary", "size", "entity_keys"}."""
    reports = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    reports[row["community_id"]] = row
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=1, help="Bundan kiçik community-lər üçün hesabat yazılmır")
    args = parser.parse_args()
    build_all_reports(min_size=args.min_size)
    