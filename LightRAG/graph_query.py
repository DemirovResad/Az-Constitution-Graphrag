"""
GraphRAG-ın 3-cü mərhələsi: LightRAG-tərzi qraf axtarışı.

Giriş:  entity_graph.json (graph_build.py) + embeddings_checkpoint.jsonl
        (Retriever vasitəsilə — chunk mətnləri və embedding-ləri oradan gəlir).
Cache:  graph_embed_cache.npz — entity/relation mətnlərinin embedding-ləri
        (mətnin hash-i ilə). Yalnız cache-də olmayan mətnlər embed olunur,
        hər batch-dən sonra diskə yazılır (kəsilsə davam edir).

Axtarış (Retriever.search ilə eyni formatda chunk qaytarır):
  1) Keyword Extraction (LightRAG-tərzi): sual LLM-ə göndərilir, ondan iki
     ayrı açar söz siyahısı çıxarılır — local (konkret entity adları) və
     global (mövzu/əlaqə xarakterli ifadələr). use_keyword_extraction=False
     ilə bu addım keçilir, local/global axtarış birbaşa sualın tam
     embedding-i ilə aparılır (əvvəlki sadə davranış).
  2) Keyword Matching:
       local  — local keyword-lərin embedding-i ən yaxın ENTITY-lərə
                uyğunlaşdırılır; onların və çıxan/daxil olan
                əlaqələrinin source_chunks-u toplanır.
       global — global keyword-lərin embedding-i ən yaxın RELATION-lara
                uyğunlaşdırılır; onların source_chunks-u toplanır.
       hybrid — ikisi birlikdə (defolt).
  3) Namizəd chunk-lar iki siyahı üzrə RRF ilə sıralanır: (a) qraf skoru,
     (b) sualın TAM mətni ilə chunk-ın öz embedding-i arasında cosine
     oxşarlığı. Cosine hissəsi "Prezident" kimi yüzlərlə chunk-a bağlı hub
     entity-lərin nəticəni boğmasının qarşısını alır.

İstifadə:
    python graph_query.py "Prezident kimləri təyin edir?"
    python graph_query.py "..." --mode local --top-k 5
    python graph_query.py "..." --no-keywords     # keyword extraction-sız
    python graph_query.py --rebuild                # yalnız indeksi yenidən qur
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
from google.genai import types
from pydantic import BaseModel, Field

from call_model import get_model, invoke_with_retry
from config import (
    EMBED_MODEL,
    OUTPUT_DIM,
    RPD_LIMIT,
    RPM_LIMIT,
    TPM_LIMIT,
    _DAILY_QUOTA_PATTERN,
    _RETRY_DELAY_PATTERN,
    DailyQuotaExceeded,
    RateLimiter,
    estimate_tokens,
)
from LightRAG.graph_build import GRAPH_OUTPUT_PATH, load_graph
from logging_config import get_logger
from retriever import Retriever

logger = get_logger(__name__)

# Entity/relation embedding cache. Açar = mətnin hash-i (model + ölçü + mətn),
# yəni entity_graph.json-un mtime-ı yox, MƏZMUNU vacibdir: qraf yenidən
# qurulsa belə mətni dəyişməyən entity-lər təkrar embed olunmur. Yol
# işlədiyin qovluqdan asılı deyil (bu faylın yanında saxlanır).
EMBED_CACHE_PATH = Path(__file__).resolve().parent / "graph_embed_cache.npz"

EMBED_BATCH = 50
MAX_TEXT_CHARS = 600  # uzun birləşmiş təsvirlər embedding keyfiyyətini aşağı salır
CONTEXT_DESC_CHARS = 400  # LLM kontekstinə düşən entity/əlaqə təsvirinin maksimum uzunluğu


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


# ============================================================
# Query Keyword Extraction (LightRAG-tərzi)
#
# Orijinal LightRAG sualı birbaşa embed etmir — əvvəlcə LLM-dən iki ayrı
# açar söz siyahısı çıxarır (local: konkret entity-lər, global: mövzu/əlaqə
# xarakterli ifadələr) və YALNIZ bunları entity/relation indeksi ilə
# müqayisə edir. Uzun, çoxhissəli suallarda bu, tam sualın "ortalaşmış"
# embedding-indən daha dəqiq seed verir.
# ============================================================
class QueryKeywords(BaseModel):
    local_keywords: list[str] = Field(
        description="Sualda keçən və ya nəzərdə tutulan KONKRET entity adları: "
                    "dövlət orqanı, vəzifə, hüquq/azadlıq adı."
    )
    global_keywords: list[str] = Field(
        description="Sualın MÖVZUSUNU və ya axtardığı ƏLAQƏ NÖVÜNÜ təsvir edən "
                    "qısa ifadələr (məs. 'təyinat səlahiyyəti', 'hüquqi müdafiə')."
    )


_KEYWORD_PROMPT = """\
Aşağıdakı sual Azərbaycan Respublikasının Konstitusiyası üzrə verilib.
Sualdan iki növ açar söz çıxar:

- local_keywords: sualda keçən və ya nəzərdə tutulan KONKRET entity adları
  (dövlət orqanı, vəzifə, hüquq/azadlıq adı). Məs. "Prezident",
  "Mərkəzi Seçki Komissiyası", "ölkəyə qayıtmaq hüququ".
- global_keywords: sualın MÖVZUSUNU və ya axtardığı ƏLAQƏ NÖVÜNÜ təsvir
  edən qısa ifadələr. Məs. "təyinat səlahiyyəti", "vətəndaşın hüquqi
  müdafiəsi", "dövlət orqanları arasında səlahiyyət bölgüsü".

Hər siyahıda 2-5 element, Azərbaycan dilində, sualda açıq yazılmayıbsa belə
nəzərdə tutulanı çıxar (uydurma, yalnız sualdan çıxan).

Sual: {query}
"""

_keyword_model = None


def _get_keyword_model():
    global _keyword_model
    if _keyword_model is None:
        _keyword_model = get_model().with_structured_output(QueryKeywords)
    return _keyword_model


def extract_keywords(query: str) -> QueryKeywords:
    try:
        return invoke_with_retry(_get_keyword_model(), _KEYWORD_PROMPT.format(query=query))
    except Exception as exc:
        logger.warning("keyword extraction uğursuz oldu (%s) — tam sual istifadə olunur", str(exc)[:100])
        return QueryKeywords(local_keywords=[], global_keywords=[])


class GraphRetriever:
    def __init__(
        self,
        retriever: Retriever | None = None,
        graph_path: Path = GRAPH_OUTPUT_PATH,
        rebuild_index: bool = False,
        trust_legacy: bool = False,
    ):
        # Retriever paylaşılır: chunk mətnləri/embedding-ləri və sual embedding-i oradan.
        self.retriever = retriever or Retriever()
        self.graph_path = Path(graph_path)
        self.graph = load_graph(self.graph_path)
        self._id_to_idx = {cid: i for i, cid in enumerate(self.retriever.ids)}

        self.entity_keys: list[str] = []
        self.edge_keys: list[tuple[str, str]] = []
        self.entity_emb: np.ndarray
        self.edge_emb: np.ndarray
        self._entity_idx: dict[str, int] = {}
        self._edge_idx: dict[tuple[str, str], int] = {}

        # Köhnə graph_index.npz-i mtime uyğun gəlməsə də qəbul et (qraf dəyişməyib
        # deyə əminsənsə): TRUST_LEGACY_INDEX=1 və ya CLI-da --trust-legacy.
        trust_legacy = trust_legacy or os.getenv("TRUST_LEGACY_INDEX") == "1"
        self._load_or_build_index(rebuild_index, trust_legacy)

    # ------------------------------------------------------------
    # İndeks
    # ------------------------------------------------------------
    def _entity_text(self, key: str) -> str:
        d = self.graph.nodes[key]
        return _clip(f"{d['name']} ({d['type']}): {d['description']}")

    def _edge_text(self, src: str, tgt: str) -> str:
        e = self.graph.edges[src, tgt]
        src_name = self.graph.nodes[src]["name"]
        tgt_name = self.graph.nodes[tgt]["name"]
        return _clip(f"{src_name} → {tgt_name}: {e['description']}")

    @staticmethod
    def _text_key(text: str) -> str:
        sig = f"{EMBED_MODEL}|{OUTPUT_DIM}|RETRIEVAL_DOCUMENT|{text}"
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_cache() -> dict[str, np.ndarray]:
        if not EMBED_CACHE_PATH.exists():
            return {}
        try:
            data = np.load(EMBED_CACHE_PATH, allow_pickle=False)
            return dict(zip(data["keys"].tolist(), data["vectors"]))
        except Exception as exc:
            logger.warning("embed cache oxunmadı (%s) — sıfırdan qurulacaq", str(exc)[:100])
            return {}

    @staticmethod
    def _save_cache(cache: dict[str, np.ndarray]) -> None:
        if not cache:
            return
        keys = list(cache)
        tmp = EMBED_CACHE_PATH.with_name(EMBED_CACHE_PATH.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
        os.replace(tmp, EMBED_CACHE_PATH)  # yarımçıq fayl qalmasın

    def _embed_missing(self, missing: list[tuple[str, str]], cache: dict[str, np.ndarray]) -> None:
        """Cache-də olmayan mətnləri batch-batch embed edir. Hər batch-dən sonra
        cache diskə yazılır — kəsilsə (429/günlük limit) qalan yerdən davam edir."""
        pool = self.retriever.pool  # KeyPool: günlük limit bitəndə növbəti key
        limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)
        start = 0
        while start < len(missing):
            batch = missing[start:start + EMBED_BATCH]
            texts = [t for _, t in batch]
            est = sum(estimate_tokens(t) for t in texts)

            try:
                limiter.wait_if_needed(est)
            except DailyQuotaExceeded:
                if pool.rotate():
                    limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)
                    continue
                self._save_cache(cache)
                raise RuntimeError("Bütün API key-lərin günlük limiti bitdi; cache saxlanıldı, sonra davam et.")

            attempt = 0
            while True:
                try:
                    res = pool.client.models.embed_content(
                        model=EMBED_MODEL,
                        contents=texts,
                        config=types.EmbedContentConfig(
                            output_dimensionality=OUTPUT_DIM,
                            task_type="RETRIEVAL_DOCUMENT",
                        ),
                    )
                    break
                except Exception as exc:
                    err = str(exc)
                    is_rate = "RESOURCE_EXHAUSTED" in err or "429" in err
                    if is_rate and _DAILY_QUOTA_PATTERN.search(err):
                        logger.warning("günlük limit bitdi (%s)", pool.current_key_masked)
                        if pool.rotate():
                            limiter = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)
                            continue
                        self._save_cache(cache)
                        raise RuntimeError("Bütün API key-lərin günlük limiti bitdi; cache saxlanıldı, sonra davam et.")
                    if is_rate and attempt < 5:
                        attempt += 1
                        m = _RETRY_DELAY_PATTERN.search(err)
                        wait = float(m.group(1)) + 3 if m else 10.0 * attempt
                        logger.warning("429 — %.0fs gözlənilir (cəhd %d/5)", wait, attempt)
                        time.sleep(wait)
                        continue
                    self._save_cache(cache)
                    raise

            for (key, _), item in zip(batch, res.embeddings):
                vec = np.array(item.values, dtype=np.float32)
                cache[key] = vec / np.linalg.norm(vec)
            limiter.record(est)
            self._save_cache(cache)
            start += EMBED_BATCH
            logger.info("embed: %d/%d", min(start, len(missing)), len(missing))

    def _import_legacy(
        self,
        cache: dict[str, np.ndarray],
        entity_texts: list[str],
        edge_texts: list[str],
        trust: bool,
    ) -> bool:
        """Köhnə formatdakı graph_index.npz + graph_index_meta.json-dan vektorları
        API çağırışı OLMADAN yeni cache-ə köçürür. Yalnız qraf həqiqətən eynidirsə:
        entity/əlaqə açarları tam üst-üstə düşməli və entity_graph.json-un mtime-ı
        meta ilə eyni olmalıdır (trust=True olsa mtime yoxlanmır)."""
        for base in (EMBED_CACHE_PATH.parent, Path.cwd()):
            index_p, meta_p = base / "graph_index.npz", base / "graph_index_meta.json"
            if index_p.exists() and meta_p.exists():
                break
        else:
            return False

        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            data = np.load(index_p)
            ent_emb, edge_emb = data["entity_emb"], data["edge_emb"]
        except Exception as exc:
            logger.warning("köhnə indeks oxunmadı (%s)", str(exc)[:100])
            return False

        e_idx = {k: i for i, k in enumerate(meta["entity_keys"])}
        r_idx = {tuple(k): i for i, k in enumerate(meta["edge_keys"])}
        same_keys = (
            set(e_idx) == set(self.entity_keys)
            and set(r_idx) == {tuple(k) for k in self.edge_keys}
            and len(ent_emb) == len(e_idx) and len(edge_emb) == len(r_idx)
            and (not len(ent_emb) or ent_emb.shape[1] == OUTPUT_DIM)
        )
        same_mtime = meta.get("graph_mtime") == self.graph_path.stat().st_mtime_ns
        if not same_keys:
            logger.info("köhnə indeks istifadə olunmur: qrafın entity/əlaqə siyahısı dəyişib")
            return False
        if not (same_mtime or trust):
            logger.info(
                "köhnə indeks istifadə olunmur: %s dəyişdirilib (mtime fərqli). Qraf "
                "dəyişməyibsə --trust-legacy (və ya TRUST_LEGACY_INDEX=1) ilə qəbul et.",
                self.graph_path,
            )
            return False

        for key, text in zip(self.entity_keys, entity_texts):
            cache[self._text_key(text)] = ent_emb[e_idx[key]].astype(np.float32)
        for key, text in zip(self.edge_keys, edge_texts):
            cache[self._text_key(text)] = edge_emb[r_idx[tuple(key)]].astype(np.float32)
        logger.info(
            "köhnə indeksdən köçürüldü (API çağırışı olmadan): %d entity, %d əlaqə",
            len(self.entity_keys), len(self.edge_keys),
        )
        return True

    def _load_or_build_index(self, rebuild: bool, trust_legacy: bool = False) -> None:
        self.entity_keys = list(self.graph.nodes)
        self.edge_keys = list(self.graph.edges)
        entity_texts = [self._entity_text(k) for k in self.entity_keys]
        edge_texts = [self._edge_text(u, v) for u, v in self.edge_keys]

        cache = {} if rebuild else self._load_cache()
        wanted = {self._text_key(t): t for t in entity_texts + edge_texts}
        missing = [(k, t) for k, t in wanted.items() if k not in cache]

        if missing and not rebuild and len(missing) == len(wanted):
            if self._import_legacy(cache, entity_texts, edge_texts, trust_legacy):
                self._save_cache({k: cache[k] for k in wanted})
                missing = []
                logger.info("qraf indeksi yeni cache formatına keçirildi: %s", EMBED_CACHE_PATH)

        if missing:
            logger.info(
                "qraf indeksi: %d/%d mətn cache-də yoxdur, embed olunur "
                "(%d entity, %d əlaqə; cache: %s)",
                len(missing), len(wanted), len(self.entity_keys), len(self.edge_keys),
                EMBED_CACHE_PATH,
            )
            self._embed_missing(missing, cache)
            self._save_cache({k: cache[k] for k in wanted})  # köhnə/artıq mətnləri təmizlə
        else:
            logger.info(
                "qraf indeksi cache-dən yükləndi (%d entity, %d əlaqə)",
                len(self.entity_keys), len(self.edge_keys),
            )

        def stack(texts: list[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, OUTPUT_DIM), dtype=np.float32)
            return np.stack([cache[self._text_key(t)] for t in texts])

        self.entity_emb = stack(entity_texts)
        self.edge_emb = stack(edge_texts)
        self._entity_idx = {k: i for i, k in enumerate(self.entity_keys)}
        self._edge_idx = {tuple(k): i for i, k in enumerate(self.edge_keys)}

    # ------------------------------------------------------------
    # Axtarış
    # ------------------------------------------------------------
    def seed_entities(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        if not self.entity_keys:
            return []
        sims = self.entity_emb @ qvec
        return [(self.entity_keys[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    def seed_edges(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, str, float]]:
        if not self.edge_keys:
            return []
        sims = self.edge_emb @ qvec
        return [(*self.edge_keys[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    def _add(self, scores: dict, chunk_ids: list[str], weight: float) -> None:
        valid = [c for c in chunk_ids if c in self._id_to_idx]
        if not valid:
            return
        # Çox chunk-a bağlı (hub) entity-nin çəkisi bölünür — IDF-ə bənzər effekt.
        w = weight / math.sqrt(len(valid))
        for c in valid:
            scores[c] += w

    def search_context(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        entity_k: int = 8,
        edge_k: int = 8,
        rrf_k: int = 60,
        max_entities: int = 10,
        max_relations: int = 10,
        use_keyword_extraction: bool = True,
    ) -> dict:
        """LightRAG konteksti: {"chunks": [...], "entities": [...], "relations": [...]}

        use_keyword_extraction=True (defolt) — LightRAG-tərzi: sualdan əvvəlcə
        LLM ilə local/global açar sözlər çıxarılır, seed axtarışı bunlarla
        aparılır. False olsa, köhnə davranış: seed axtarışı birbaşa sualın
        tam embedding-i ilə aparılır (daha ucuz, bir API çağırışı az).

        chunks    — qraf skoru + sualın TAM mətni ilə cosine (RRF).
        entities  — local: seed entity-lər; global: seçilən əlaqələrin uc
                    entity-ləri.
        relations — local: həmin entity-lərin əlaqələri; global: seed əlaqələr.
        Hub entity-nin yüzlərlə əlaqəsi ola bilər — ona görə namizəd
        entity/əlaqələr sualın TAM mətni ilə cosine oxşarlığına görə
        sıralanıb max_entities / max_relations qədəri götürülür."""
        qvec = self.retriever._embed_query(query)  # chunk-level cosine və entity/relation sıralaması üçün

        if use_keyword_extraction:
            kw = extract_keywords(query)
            local_text = " ".join(kw.local_keywords) or query
            global_text = " ".join(kw.global_keywords) or query
            local_vec = self.retriever._embed_query(local_text) if kw.local_keywords else qvec
            global_vec = self.retriever._embed_query(global_text) if kw.global_keywords else qvec
            logger.debug("keywords: local=%s global=%s", kw.local_keywords, kw.global_keywords)
        else:
            local_vec = global_vec = qvec

        scores: dict[str, float] = defaultdict(float)
        ent_cands: set[str] = set()
        rel_cands: set[tuple[str, str]] = set()

        if mode in ("local", "hybrid"):
            for key, sim in self.seed_entities(local_vec, entity_k):
                self._add(scores, self.graph.nodes[key]["source_chunks"], sim)
                ent_cands.add(key)
                incident = chain(
                    self.graph.out_edges(key, data=True),
                    self.graph.in_edges(key, data=True),
                )
                for u, v, ed in incident:
                    strength = ed.get("strength", 5) / 10
                    self._add(scores, ed["source_chunks"], sim * strength)
                    rel_cands.add((u, v))

        if mode in ("global", "hybrid"):
            for u, v, sim in self.seed_edges(global_vec, edge_k):
                self._add(scores, self.graph.edges[u, v]["source_chunks"], sim)
                rel_cands.add((u, v))
                ent_cands.update((u, v))

        if not scores:
            logger.warning("graph search: namizəd chunk tapılmadı (query=%r)", query[:80])
            return {"chunks": [], "entities": [], "relations": []}

        cand = list(scores)
        idxs = np.array([self._id_to_idx[c] for c in cand])
        cos = self.retriever.embeddings[idxs] @ qvec
        cos_by_id = {c: float(cos[j]) for j, c in enumerate(cand)}

        graph_sorted = sorted(cand, key=lambda c: (-scores[c], -cos_by_id[c]))
        cos_sorted = sorted(cand, key=lambda c: -cos_by_id[c])
        graph_rank = {c: r + 1 for r, c in enumerate(graph_sorted)}
        cosine_rank = {c: r + 1 for r, c in enumerate(cos_sorted)}

        fused = {c: 1.0 / (rrf_k + graph_rank[c]) + 1.0 / (rrf_k + cosine_rank[c]) for c in cand}
        top = sorted(cand, key=lambda c: -fused[c])[:top_k]

        logger.debug("graph search: %s", [(c, round(fused[c], 5)) for c in top])
        chunks = [
            {
                "id": c,
                "text": self.retriever.texts[self._id_to_idx[c]],
                "score": float(fused[c]),
                "graph_rank": graph_rank[c],
                "cosine_rank": cosine_rank[c],
            }
            for c in top
        ]

        # --- entity / əlaqələr: sualın TAM mətni ilə birbaşa oxşarlığa görə seçilir
        ent_sims = self.entity_emb @ qvec if len(self.entity_keys) else np.zeros(0)
        edge_sims = self.edge_emb @ qvec if len(self.edge_keys) else np.zeros(0)

        ent_top = sorted(ent_cands, key=lambda k: -ent_sims[self._entity_idx[k]])[:max_entities]
        rel_top = sorted(rel_cands, key=lambda e: -edge_sims[self._edge_idx[e]])[:max_relations]

        entities = []
        for key in ent_top:
            n = self.graph.nodes[key]
            entities.append({
                "name": n["name"],
                "type": n["type"],
                "description": _clip(n["description"], CONTEXT_DESC_CHARS),
                "source_chunks": [c for c in n["source_chunks"] if c in self._id_to_idx],
                "score": float(ent_sims[self._entity_idx[key]]),
            })
        relations = []
        for u, v in rel_top:
            ed = self.graph.edges[u, v]
            relations.append({
                "source": self.graph.nodes[u]["name"],
                "target": self.graph.nodes[v]["name"],
                "description": _clip(ed["description"], CONTEXT_DESC_CHARS),
                "strength": ed.get("strength"),
                "source_chunks": [c for c in ed["source_chunks"] if c in self._id_to_idx],
                "score": float(edge_sims[self._edge_idx[(u, v)]]),
            })

        return {"chunks": chunks, "entities": entities, "relations": relations}

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        """Yalnız chunk-lar (əvvəlki interfeys): [{"id","text","score","graph_rank","cosine_rank"}]"""
        return self.search_context(query, top_k=top_k, **kwargs)["chunks"]


# ============================================================
# CLI — sürətli yoxlama
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-keywords", action="store_true",
                        help="LLM keyword extraction addımını keç, birbaşa tam sualla axtar")
    parser.add_argument("--trust-legacy", action="store_true",
                        help="köhnə graph_index.npz-i mtime uyğun gəlməsə də qəbul et (qraf dəyişməyib)")
    args = parser.parse_args()

    gr = GraphRetriever(rebuild_index=args.rebuild, trust_legacy=args.trust_legacy)
    if not args.query:
        print(f"İndeks hazırdır: {len(gr.entity_keys)} entity, {len(gr.edge_keys)} əlaqə")
        raise SystemExit

    use_kw = not args.no_keywords
    if use_kw:
        kw = extract_keywords(args.query)
        print(f"Local keywords:  {kw.local_keywords}")
        print(f"Global keywords: {kw.global_keywords}")

    qvec = gr.retriever._embed_query(args.query)
    print("\nSeed entity-lər (tam sual üzrə, məlumat üçün):")
    for key, sim in gr.seed_entities(qvec, 5):
        print(f"  {sim:.3f}  {gr.graph.nodes[key]['name']}")
    print("Seed əlaqələr (tam sual üzrə, məlumat üçün):")
    for u, v, sim in gr.seed_edges(qvec, 5):
        print(f"  {sim:.3f}  {gr.graph.nodes[u]['name']} → {gr.graph.nodes[v]['name']}")

    res = gr.search_context(args.query, top_k=args.top_k, mode=args.mode, use_keyword_extraction=use_kw)
    print(f"\nEntity-lər ({args.mode}):")
    for e in res["entities"]:
        print(f"  {e['score']:.3f}  {e['name']} ({e['type']})")
    print("Əlaqələr:")
    for r in res["relations"]:
        print(f"  {r['score']:.3f}  {r['source']} → {r['target']}: {r['description'][:80]}")
    print("Chunk-lar:")
    for r in res["chunks"]:
        print(f"  {r['id']}  graph#{r['graph_rank']} cos#{r['cosine_rank']}  {r['text'][:120]!r}")