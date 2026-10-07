"""
GraphRAG-ın 2-ci mərhələsi: graph_extract.py-nin çıxardığı entity/əlaqələri
oxuyub networkx qrafı qurur və hierarchical Leiden ilə çoxsəviyyəli
community-lər təyin edir.

Giriş: graph_extraction_checkpoint.jsonl (graph_extract.py-nin çıxışı)
Çıxış: entity_graph.json   — node_link_data; hər node-da "community" atributu
                             ən aşağı (final) səviyyədəki community id-dir.
       communities.json    — bütün səviyyələr: [{"id", "level", "parent", "members"}]
                             level 0 = ən ümumi (yuxarı) səviyyə.

İstifadə:
    python graph_build.py
"""

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities

BASE_DIR = Path(__file__).resolve().parent
GRAPH_EXTRACTION_PATH = "graph_extraction_checkpoint.jsonl"
GRAPH_OUTPUT_PATH = BASE_DIR / "entity_graph.json"

COMMUNITIES_PATH = BASE_DIR / "communities.json"

# Bir community bundan böyükdürsə Leiden onu alt-community-lərə bölür.
MAX_CLUSTER_SIZE = 15
LEIDEN_SEED = 42


def _normalize(name: str) -> str:
    """Entity adlarını müqayisə üçün normallaşdırır. Azərbaycan hərfləri üçün
    xüsusi xəritələmə: 'İ' -> 'i', 'I' -> 'ı' (standart .lower() 'İ'-ni
    birləşdirici nöqtəli 'i̇'-yə, 'I'-ni isə 'i'-yə çevirir)."""
    s = unicodedata.normalize("NFC", name.strip())
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return " ".join(s.split())


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

    node_data: dict[str, dict] = defaultdict(lambda: {
        "names": Counter(), "types": Counter(), "descriptions": [], "source_chunks": [],
    })

    for row in extractions:
        chunk_id = row["chunk_id"]
        for ent in row.get("entities", []):
            data = node_data[_normalize(ent["name"])]
            data["names"][ent["name"]] += 1
            data["types"][ent["type"]] += 1
            data["descriptions"].append(ent["description"])
            data["source_chunks"].append(chunk_id)

    for key, data in node_data.items():
        graph.add_node(
            key,
            name=data["names"].most_common(1)[0][0],   # ən çox işlənən yazılış
            type=data["types"].most_common(1)[0][0],   # ən çox təkrarlanan tip
            description=" | ".join(dict.fromkeys(data["descriptions"])),
            source_chunks=list(dict.fromkeys(data["source_chunks"])),
        )

    edge_data: dict[tuple, dict] = defaultdict(lambda: {
        "descriptions": [], "strength": 0, "weight": 0, "source_chunks": [],
    })

    for row in extractions:
        chunk_id = row["chunk_id"]
        for rel in row.get("relations", []):
            src_key = _normalize(rel["source"])
            tgt_key = _normalize(rel["target"])
            if src_key not in node_data or tgt_key not in node_data or src_key == tgt_key:
                continue
            data = edge_data[(src_key, tgt_key)]
            data["descriptions"].append(rel["relation"])
            data["strength"] = max(data["strength"], rel["strength"])
            data["weight"] += rel["strength"]  # GraphRAG kimi: təkrarlanan əlaqələrin çəkisi toplanır
            data["source_chunks"].append(chunk_id)

    for (src_key, tgt_key), data in edge_data.items():
        graph.add_edge(
            src_key, tgt_key,
            description=" | ".join(dict.fromkeys(data["descriptions"])),
            strength=data["strength"],
            weight=data["weight"],
            source_chunks=list(dict.fromkeys(data["source_chunks"])),
        )

    return graph


def _undirected_weighted(graph: nx.DiGraph) -> nx.Graph:
    """A→B və B→A çəkiləri toplanaraq tək undirected edge olur."""
    und = nx.Graph()
    und.add_nodes_from(graph.nodes)
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1)
        if und.has_edge(u, v):
            und[u][v]["weight"] += w
        else:
            und.add_edge(u, v, weight=w)
    return und


def assign_communities(graph: nx.DiGraph, seed: int = LEIDEN_SEED) -> dict[int, dict]:
    """Rekursiv Louvain ilə iyerarxik community-lər. level 0 = ən ümumi.
    Hər node-a graph.nodes[key]['community'] (yarpaq/final community id) yazır.
    Qaytarır: {community_id: {"id", "level", "parent", "members"}}."""
    und = _undirected_weighted(graph)
    communities: dict[int, dict] = {}
    counter = iter(range(10**9))

    def add(members, level, parent):
        cid = next(counter)
        communities[cid] = {"id": cid, "level": level, "parent": parent, "members": sorted(members)}
        return cid

    def split(members, level, parent):
        parts = louvain_communities(und.subgraph(members), weight="weight", seed=seed)
        if len(parts) == 1 and parent is not None:
            return  # bölünmür, bu community yarpaq qalır
        for part in parts:
            cid = add(part, level, parent)
            if len(part) > MAX_CLUSTER_SIZE:
                split(part, level + 1, cid)

    split(list(und.nodes), 0, None)

    has_child = {c["parent"] for c in communities.values() if c["parent"] is not None}
    for cid, c in communities.items():
        if cid not in has_child:
            for key in c["members"]:
                graph.nodes[key]["community"] = cid

    return communities


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_OUTPUT_PATH) -> None:
    data = nx.node_link_data(graph, edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(path: Path = GRAPH_OUTPUT_PATH) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def save_communities(communities: dict[int, dict], path: Path = COMMUNITIES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(communities.values()), f, ensure_ascii=False, indent=2)


def load_communities(path: Path = COMMUNITIES_PATH) -> dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)}


if __name__ == "__main__":
    extractions = load_extractions()
    print(f"Yükləndi: {len(extractions)} chunk-ın entity/əlaqə nəticəsi")

    graph = build_graph(extractions)
    communities = assign_communities(graph)
    save_graph(graph)
    save_communities(communities)

    print("=" * 60)
    print("QRAF QURULDU")
    print("=" * 60)
    print(f"Node sayı (unikal entity): {graph.number_of_nodes()}")
    print(f"Edge sayı (unikal əlaqə):  {graph.number_of_edges()}")
    print(f"Community sayı (bütün səviyyələr): {len(communities)}")

    by_level = Counter(c["level"] for c in communities.values())
    print("Səviyyələr üzrə community sayı:", dict(sorted(by_level.items())))

    top_nodes = sorted(graph.degree, key=lambda x: -x[1])[:10]
    print("\nƏn çox əlaqəli entity-lər:")
    for key, degree in top_nodes:
        print(f"  - {graph.nodes[key]['name']} ({graph.nodes[key]['type']}) — {degree} əlaqə")

    final_sizes = Counter(d["community"] for _, d in graph.nodes(data=True))
    size_list = sorted(final_sizes.values(), reverse=True)
    print(f"\nFinal community ölçüləri (ilk 15): {size_list[:15]}")
    print(f"1 node-luq final community sayı: {sum(1 for s in size_list if s == 1)}")

    print(f"\nNəticə: {GRAPH_OUTPUT_PATH}, {COMMUNITIES_PATH}")