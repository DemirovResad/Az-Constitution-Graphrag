"""
GraphRAG-ın 2-ci mərhələsi: graph_extract.py-nin çıxardığı entity/əlaqələri
oxuyub, bir networkx qrafı qurur. Community detection/report YOXDUR
(LightRAG-tərzi sadələşdirmə) — birbaşa entity-based qraf, sonrakı
addımda (graph_query.py) local search üçün istifadə olunacaq.

Giriş: graph_extraction_checkpoint.jsonl (graph_extract.py-nin çıxışı)
Çıxış: entity_graph.json (networkx node_link_data formatında)

İstifadə:
    python graph_build.py
"""
import sys
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
GRAPH_EXTRACTION_PATH = "graph_extraction_checkpoint.jsonl"
GRAPH_OUTPUT_PATH = BASE_DIR / "entity_graph.json"



def _normalize(name: str) -> str:
    """Entity adlarını müqayisə üçün normallaşdırır (resolution üçün açar).
    Sadə case-insensitive + boşluq təmizləmə — 'Prezident' və 'prezident'
    eyni node-a düşür, amma 'Prezident' və 'dövlət başçısı' fərqli qalır
    (bunun üçün LLM/embedding-əsaslı fuzzy matching lazım olardı)."""
    return " ".join(name.strip().lower().split())


def load_extractions(path: Path = GRAPH_EXTRACTION_PATH) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_graph(extractions: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()

    # Node-ları toplayırıq: normalize açar -> {name, type, descriptions[], source_chunks[]}
    node_data: dict[str, dict] = defaultdict(lambda: {
        "name": None, "type": None, "descriptions": [], "source_chunks": [],
    })

    for row in extractions:
        chunk_id = row["chunk_id"]
        for ent in row.get("entities", []):
            key = _normalize(ent["name"])
            data = node_data[key]
            if data["name"] is None:
                data["name"] = ent["name"]  # ilk görünən yazılışı canonical kimi saxla
                data["type"] = ent["type"]
            data["descriptions"].append(ent["description"])
            data["source_chunks"].append(chunk_id)

    for key, data in node_data.items():
        graph.add_node(
            key,
            name=data["name"],
            type=data["type"],
            description=" | ".join(dict.fromkeys(data["descriptions"])),  # təkrarları at
            source_chunks=list(dict.fromkeys(data["source_chunks"])),
        )

    # Edge-ləri əlavə edirik: eyni cüt arasında bir neçə əlaqə varsa, birləşdiririk.
    edge_data: dict[tuple, dict] = defaultdict(lambda: {
        "descriptions": [], "strength": 0, "source_chunks": [],
    })

    for row in extractions:
        chunk_id = row["chunk_id"]
        for rel in row.get("relations", []):
            src_key = _normalize(rel["source"])
            tgt_key = _normalize(rel["target"])
            # Yalnız hər iki tərəf entity siyahısında mövcuddursa əlavə et
            # (LLM bəzən relation-da entities-də olmayan ad yaza bilər).
            if src_key not in node_data or tgt_key not in node_data:
                continue
            edge_key = (src_key, tgt_key)
            data = edge_data[edge_key]
            data["descriptions"].append(rel["relation"])
            data["strength"] = max(data["strength"], rel["strength"])
            data["source_chunks"].append(chunk_id)

    for (src_key, tgt_key), data in edge_data.items():
        graph.add_edge(
            src_key, tgt_key,
            description=" | ".join(dict.fromkeys(data["descriptions"])),
            strength=data["strength"],
            source_chunks=list(dict.fromkeys(data["source_chunks"])),
        )

    return graph


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_OUTPUT_PATH) -> None:
    data = nx.node_link_data(graph, edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(path: Path = GRAPH_OUTPUT_PATH) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


if __name__ == "__main__":
    extractions = load_extractions()
    print(f"Yükləndi: {len(extractions)} chunk-ın entity/əlaqə nəticəsi")

    graph = build_graph(extractions)
    save_graph(graph)

    print("=" * 60)
    print("QRAF QURULDU")
    print("=" * 60)
    print(f"Node sayı (unikal entity): {graph.number_of_nodes()}")
    print(f"Edge sayı (unikal əlaqə):  {graph.number_of_edges()}")

    # Ən çox əlaqəsi olan (ən "mərkəzi") entity-ləri göstər — bunlar
    # çox güman ki, Prezident, Milli Məclis kimi əsas institutlardır.
    top_nodes = sorted(graph.degree, key=lambda x: -x[1])[:10]
    print("\nƏn çox əlaqəli entity-lər:")
    for key, degree in top_nodes:
        name = graph.nodes[key]["name"]
        node_type = graph.nodes[key]["type"]
        print(f"  - {name} ({node_type}) — {degree} əlaqə")

    print(f"\nNəticə: {GRAPH_OUTPUT_PATH}")
