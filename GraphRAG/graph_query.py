"""
GraphRAG axtarışı — LightRAG-ın əlaqə-embedding-əsaslı "global"ı ÇIXARILIB,
yerinə əsl GraphRAG-ın community report üzərində map-reduce-u qoyulub.

Giriş:  entity_graph.json (graph_build.py — community atributu ilə)
        + embeddings_checkpoint.jsonl (Retriever vasitəsilə)
        + community_reports.jsonl (community_report.py)
Cache:  graph_embed_cache.npz — entity VƏ community report mətnlərinin
        embedding-i (mətnin hash-i ilə, kəsilsə davam edir).

İki rejim:
  local  — sualın embedding-i ən yaxın ENTITY-lərə uyğunlaşdırılır; onların
           source_chunks-u, əlaqələri və aid olduqları community-lərin
           report-u toplanır. Sürətli, ucuz (yalnız embedding).
  global — sual bütün community report-larla (embedding ilə) müqayisə
           olunur, ən uyğun top-N seçilir, batch-lərə bölünüb LLM-ə
           göndərilir (map), qismən cavablar bir LLM çağırışı ilə
           birləşdirilir (reduce). Yavaş və bahalı (bir neçə chat-model
           çağırışı), amma "kitabın ümumi mövzusu" tipli suallarda
           local-ın tapa bilmədiyi sintezi verir.

İstifadə:
    python graph_query.py "Prezident kimləri təyin edir?"                # local
    python graph_query.py "Konstitusiyada hakimiyyət necə bölünüb?" --mode global
    python graph_query.py --rebuild        # yalnız indeksi yenidən qur
"""

import argparse
import hashlib
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
from GraphRAG.community_report import REPORTS_PATH, load_community_reports
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
from GraphRAG.graph_build import GRAPH_OUTPUT_PATH, load_graph
from logging_config import get_logger
from retriever import Retriever

logger = get_logger(__name__)

EMBED_CACHE_PATH = Path(__file__).resolve().parent / "graph_embed_cache.npz"
EMBED_BATCH = 50
MAX_TEXT_CHARS = 600
CONTEXT_DESC_CHARS = 400

MAP_BATCH_SIZE = 3       # bir map çağırışında neçə community report birlikdə gedir
DEFAULT_TOP_COMMUNITIES = 6


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


# ============================================================
# Global search (map-reduce) üçün struktur
# ============================================================
class MapAnswer(BaseModel):
    relevant: bool = Field(description="Bu community xülasələri sualla əlaqəlidirmi")
    partial_answer: str = Field(description="Əlaqəlidirsə bu community-lərə əsaslanan qismən cavab, deyilsə boş sətir")


class ReduceAnswer(BaseModel):
    answer: str = Field(description="Bütün qismən cavabları birləşdirən yekun, sintez edilmiş cavab")


MAP_PROMPT = """\
Sual: {query}

Aşağıda bir bilik qrafının bəzi community (mövzu qrupu) xülasələri verilib.
Bunlar Azərbaycan Respublikası Konstitusiyasından avtomatik çıxarılıb.

Bu xülasələr sualla əlaqəlidirsə, YALNIZ bu xülasələrə əsaslanaraq qismən
cavab yaz (relevant=true). Əlaqəli deyilsə, relevant=false yaz və
partial_answer-i boş burax. Uydurma, yalnız verilən mətnə əsaslan.

Community xülasələri:
{communities}
"""

REDUCE_PROMPT = """\
Sual: {query}

Aşağıda bu suala aid, fərqli mənbələrdən gələn qismən cavablar verilib.
Bunları BİRLƏŞDİRİB tək, tutarlı bir cavab yaz. Təkrarları at, ziddiyyət
varsa hər ikisini qeyd et. Yalnız aşağıdakı mətnlərə əsaslan, uydurma.

Qismən cavablar:
{partials}
"""


class GraphRetriever:
    def __init__(
        self,
        retriever: Retriever | None = None,
        graph_path: Path = GRAPH_OUTPUT_PATH,
        reports_path: Path = REPORTS_PATH,
        rebuild_index: bool = False,
    ):
        self.retriever = retriever or Retriever()
        self.graph_path = Path(graph_path)
        self.graph = load_graph(self.graph_path)
        self.community_reports = load_community_reports(reports_path)
        self._id_to_idx = {cid: i for i, cid in enumerate(self.retriever.ids)}

        self.entity_keys: list[str] = []
        self.community_ids: list[int] = []
        self.entity_emb: np.ndarray
        self.community_emb: np.ndarray

        self._load_or_build_index(rebuild_index)

    # ------------------------------------------------------------
    # İndeks (entity + community report embedding-ləri)
    # ------------------------------------------------------------
    def _entity_text(self, key: str) -> str:
        d = self.graph.nodes[key]
        return _clip(f"{d['name']} ({d['type']}): {d['description']}")

    def _community_text(self, report: dict) -> str:
        return _clip(f"{report['title']}: {report['summary']}")

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
        os.replace(tmp, EMBED_CACHE_PATH)

    def _embed_missing(self, missing: list[tuple[str, str]], cache: dict[str, np.ndarray]) -> None:
        pool = self.retriever.pool
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
                    res = self.retriever.pool.client.models.embed_content(
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

    def _load_or_build_index(self, rebuild: bool) -> None:
        self.entity_keys = list(self.graph.nodes)
        self.community_ids = list(self.community_reports)
        entity_texts = [self._entity_text(k) for k in self.entity_keys]
        community_texts = [self._community_text(self.community_reports[cid]) for cid in self.community_ids]

        cache = {} if rebuild else self._load_cache()
        wanted = {self._text_key(t): t for t in entity_texts + community_texts}
        missing = [(k, t) for k, t in wanted.items() if k not in cache]

        if missing:
            logger.info(
                "qraf indeksi: %d/%d mətn cache-də yoxdur, embed olunur "
                "(%d entity, %d community; cache: %s)",
                len(missing), len(wanted), len(self.entity_keys), len(self.community_ids),
                EMBED_CACHE_PATH,
            )
            self._embed_missing(missing, cache)
            self._save_cache({k: cache[k] for k in wanted})
        else:
            logger.info(
                "qraf indeksi cache-dən yükləndi (%d entity, %d community)",
                len(self.entity_keys), len(self.community_ids),
            )

        def stack(texts: list[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, OUTPUT_DIM), dtype=np.float32)
            return np.stack([cache[self._text_key(t)] for t in texts])

        self.entity_emb = stack(entity_texts)
        self.community_emb = stack(community_texts)
        self._entity_idx = {k: i for i, k in enumerate(self.entity_keys)}

    # ------------------------------------------------------------
    # Local search — entity-lər + əlaqələr + community report + chunk-lar
    # ------------------------------------------------------------
    def seed_entities(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        if not self.entity_keys:
            return []
        sims = self.entity_emb @ qvec
        return [(self.entity_keys[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    def _add(self, scores: dict, chunk_ids: list[str], weight: float) -> None:
        valid = [c for c in chunk_ids if c in self._id_to_idx]
        if not valid:
            return
        w = weight / math.sqrt(len(valid))
        for c in valid:
            scores[c] += w

    def local_search(
        self,
        query: str,
        top_k: int = 5,
        entity_k: int = 8,
        rrf_k: int = 60,
        max_entities: int = 10,
        max_relations: int = 10,
    ) -> dict:
        """{"chunks": [...], "entities": [...], "relations": [...], "community_reports": [...]}"""
        qvec = self.retriever._embed_query(query)
        scores: dict[str, float] = defaultdict(float)
        rel_cands: set[tuple[str, str]] = set()
        community_ids_hit: set[int] = set()

        seeds = self.seed_entities(qvec, entity_k)
        for key, sim in seeds:
            self._add(scores, self.graph.nodes[key]["source_chunks"], sim)
            cid = self.graph.nodes[key].get("community")
            if cid is not None:
                community_ids_hit.add(cid)
            incident = chain(
                self.graph.out_edges(key, data=True),
                self.graph.in_edges(key, data=True),
            )
            for u, v, ed in incident:
                strength = ed.get("strength", 5) / 10
                self._add(scores, ed["source_chunks"], sim * strength)
                rel_cands.add((u, v))

        if not scores:
            logger.warning("local search: namizəd chunk tapılmadı (query=%r)", query[:80])
            return {"chunks": [], "entities": [], "relations": [], "community_reports": []}

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

        seed_keys = [k for k, _ in seeds][:max_entities]
        entities = []
        for key in seed_keys:
            n = self.graph.nodes[key]
            entities.append({
                "name": n["name"],
                "type": n["type"],
                "description": _clip(n["description"], CONTEXT_DESC_CHARS),
                "source_chunks": [c for c in n["source_chunks"] if c in self._id_to_idx],
            })

        rel_top = list(rel_cands)[:max_relations]
        relations = []
        for u, v in rel_top:
            ed = self.graph.edges[u, v]
            relations.append({
                "source": self.graph.nodes[u]["name"],
                "target": self.graph.nodes[v]["name"],
                "description": _clip(ed["description"], CONTEXT_DESC_CHARS),
                "strength": ed.get("strength"),
                "source_chunks": [c for c in ed["source_chunks"] if c in self._id_to_idx],
            })

        community_reports = [
            {"title": self.community_reports[cid]["title"], "summary": self.community_reports[cid]["summary"]}
            for cid in community_ids_hit if cid in self.community_reports
        ]

        return {"chunks": chunks, "entities": entities, "relations": relations, "community_reports": community_reports}

    # ------------------------------------------------------------
    # Global search — community report-lar üzərində map-reduce
    # ------------------------------------------------------------
    def global_search(
        self,
        query: str,
        top_communities: int = DEFAULT_TOP_COMMUNITIES,
        batch_size: int = MAP_BATCH_SIZE,
    ) -> dict:
        """{"answer": str, "communities_used": [titles]}"""
        if not self.community_ids:
            return {"answer": "", "communities_used": [], "articles_used": []}

        qvec = self.retriever._embed_query(query)
        sims = self.community_emb @ qvec
        order = np.argsort(-sims)[:top_communities]
        selected_ids = [self.community_ids[i] for i in order]
        selected = [self.community_reports[cid] for cid in selected_ids]

        model = get_model()
        map_model = model.with_structured_output(MapAnswer)

        partials: list[str] = []
        articles_used: set[str] = set()
        for i in range(0, len(selected), batch_size):
            batch = selected[i:i + batch_size]
            batch_text = "\n\n".join(
                f"[{r['title']}] (maddələr: {', '.join(r.get('articles', [])) or 'yoxdur'})\n{r['summary']}"
                for r in batch
            )
            prompt = MAP_PROMPT.format(query=query, communities=batch_text)
            try:
                res = invoke_with_retry(map_model, prompt)
            except Exception as e:
                logger.error("global search map xəta (batch %d): %s", i // batch_size, str(e)[:150])
                continue
            if res.relevant and res.partial_answer.strip():
                partials.append(res.partial_answer.strip())
                # Maddələr LLM-dən DEYİL, bu batch-ın community report-larında
                # artıq saxlanılan siyahıdan götürülür — halüsinasiya riski yoxdur.
                # Approksimasiya: batch relevant sayılırsa, batch-dakı bütün
                # report-ların maddələri "istifadə olunub" sayılır (map LLM-i
                # hər report üçün ayrıca çağırmırıq, xərci artırmamaq üçün).
                for r in batch:
                    articles_used.update(r.get("articles", []))

        logger.info(
            "global search: %d community seçildi, %d batch-dən %d relevant qismən cavab, %d maddə",
            len(selected), math.ceil(len(selected) / batch_size), len(partials), len(articles_used),
        )

        if not partials:
            return {"answer": "", "communities_used": [r["title"] for r in selected], "articles_used": []}

        reduce_model = model.with_structured_output(ReduceAnswer)
        reduce_prompt = REDUCE_PROMPT.format(
            query=query, partials="\n\n".join(f"- {p}" for p in partials)
        )
        try:
            final = invoke_with_retry(reduce_model, reduce_prompt)
            answer = final.answer
        except Exception as e:
            logger.error("global search reduce xəta: %s", str(e)[:150])
            answer = "\n\n".join(partials)  # fallback: xam birləşmə

        sorted_articles = sorted(
            articles_used,
            key=lambda label: 0 if label == "Preambula" else int(label.split()[1]),
        )
        return {
            "answer": answer,
            "communities_used": [r["title"] for r in selected],
            "articles_used": sorted_articles,
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        """Geriyə uyğunluq: yalnız chunk-lar."""
        return self.local_search(query, top_k=top_k, **kwargs)["chunks"]


# ============================================================
# CLI — sürətli yoxlama
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", default="local", choices=["local", "global"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    gr = GraphRetriever(rebuild_index=args.rebuild)
    if not args.query:
        print(f"İndeks hazırdır: {len(gr.entity_keys)} entity, {len(gr.community_ids)} community")
        raise SystemExit

    if args.mode == "local":
        res = gr.local_search(args.query, top_k=args.top_k)
        print(f"\nEntity-lər ({len(res['entities'])}):")
        for e in res["entities"]:
            print(f"  {e['name']} ({e['type']})")
        print(f"\nCommunity report-lar ({len(res['community_reports'])}):")
        for c in res["community_reports"]:
            print(f"  [{c['title']}] {c['summary'][:100]}")
        print(f"\nChunk-lar ({len(res['chunks'])}):")
        for r in res["chunks"]:
            print(f"  {r['id']}  graph#{r['graph_rank']} cos#{r['cosine_rank']}  {r['text'][:120]!r}")
    else:
        res = gr.global_search(args.query)
        print(f"\nİstifadə olunan community-lər: {res['communities_used']}")
        print(f"İstifadə olunan maddələr: {res['articles_used']}")
        print(f"\nCavab:\n{res['answer']}")