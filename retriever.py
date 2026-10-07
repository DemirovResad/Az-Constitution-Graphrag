import argparse
import json
import re

import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi
 

from config import API_KEYS, CHECKPOINT_PATH, EMBED_MODEL, OUTPUT_DIM, KeyPool
from logging_config import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """BM25 üçün sadə tokenizer. Azərbaycan hərflərini (ə, ı, ö, ü, ş, ç)
    \\w unicode dəstəyi sayəsində düzgün tanıyır."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class Retriever:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.pool = KeyPool(API_KEYS)

        self.ids: list[str] = []
        self.texts: list[str] = []
        self.embeddings: np.ndarray | None = None

        self._load_index()

    def _load_index(self):
        """Checkpoint .jsonl faylını yaddaşa yükləyir və embedding-ləri
        tək bir numpy matrisinə çevirir (sürətli cosine similarity üçün)."""
        ids, texts, vectors = [], [], []

        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ids.append(row["id"])
                texts.append(row["text"])
                vectors.append(row["embedding"])

        if not vectors:
            raise ValueError(
                f"{self.checkpoint_path} boşdur — əvvəlcə embed_gemini.py ilə "
                "embedding-ləri yaratmalısan."
            )

        self.ids = ids
        self.texts = texts
        matrix = np.array(vectors, dtype=np.float32)
        # Norması əvvəlcədən hesablanır ki, hər axtarışda təkrar hesablanmasın.
        self.embeddings = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        # BM25 indeksi — cosine similarity ilə paralel işləyəcək.
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("%d chunk yükləndi (%s).", len(self.ids), self.checkpoint_path)

    def _embed_query(self, query: str) -> np.ndarray:
        
        result = self.pool.client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIM,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def _cosine_scores(self, query: str) -> np.ndarray:
        query_vec = self._embed_query(query)
        # Vektorlar artıq normalize olunduğu üçün dot product = cosine similarity.
        return self.embeddings @ query_vec

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokenized_query = _tokenize(query)
        return np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 3,
        per_method_k: int = 5,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Sualı embed edib nəticə qaytarır.

        mode="hybrid" (defolt) — Reciprocal Rank Fusion (RRF):
          1. Cosine (semantik) öz TOP `per_method_k` (defolt 5) nəticəsini verir.
          2. BM25 (leksik) da öz TOP `per_method_k` nəticəsini verir.
          3. İki siyahının BİRLƏŞMƏSİ (təkrarlar bir dəfə sayılır) mövqeyə
             (rank-a) görə yenidən xallandırılır:
                 rrf_score = sum( 1 / (rrf_k + rank) )  hər siyahıda göründüyü yerə görə
             Xam skorları (cosine -1..1, BM25 0..sonsuz) qarışdırmadığı üçün
             miqyas fərqindən asılı deyil — yalnız "hansı metodda neçənci yerdədir" önəmlidir.
          4. Birləşmiş sıralamadan yekun TOP `top_k` (defolt 4) qaytarılır.

        mode="cosine" / "bm25" — yalnız həmin metodun öz TOP `top_k`-i.

        Nəticə: [{"id", "text", "score", "cosine_rank", "bm25_rank"}, ...] (score azalan sırada).
        """
        logger.debug("search: query=%r mode=%s top_k=%d", query, mode, top_k)

        if mode == "cosine":
            cosine_raw = self._cosine_scores(query)
            top_indices = np.argsort(-cosine_raw)[:top_k]
            return [
                {"id": self.ids[i], "text": self.texts[i], "score": float(cosine_raw[i]),
                 "cosine_rank": rank + 1, "bm25_rank": None}
                for rank, i in enumerate(top_indices)
            ]

        if mode == "bm25":
            bm25_raw = self._bm25_scores(query)
            top_indices = np.argsort(-bm25_raw)[:top_k]
            return [
                {"id": self.ids[i], "text": self.texts[i], "score": float(bm25_raw[i]),
                 "cosine_rank": None, "bm25_rank": rank + 1}
                for rank, i in enumerate(top_indices)
            ]

        # mode == "hybrid" — RRF
        cosine_raw = self._cosine_scores(query)
        bm25_raw = self._bm25_scores(query)

        cosine_top = np.argsort(-cosine_raw)[:per_method_k]
        bm25_top = np.argsort(-bm25_raw)[:per_method_k]

        cosine_ranks = {int(idx): rank + 1 for rank, idx in enumerate(cosine_top)}
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_top)}

        candidate_indices = set(cosine_ranks) | set(bm25_ranks)

        rrf_scores = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in cosine_ranks:
                score += 1.0 / (rrf_k + cosine_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[idx])
            rrf_scores[idx] = score

        ranked = sorted(candidate_indices, key=lambda idx: -rrf_scores[idx])[:top_k]

        logger.debug(
            "search: hybrid nəticə — %s",
            [(self.ids[idx], round(rrf_scores[idx], 5)) for idx in ranked],
        )

        return [
            {
                "id": self.ids[idx],
                "text": self.texts[idx],
                "score": float(rrf_scores[idx]),
                "cosine_rank": cosine_ranks.get(idx),
                "bm25_rank": bm25_ranks.get(idx),
            }
            for idx in ranked
        ]


