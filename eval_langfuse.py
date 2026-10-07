"""
Langfuse Dataset + Experiment + LLM-as-a-judge ilə RAG agentinin qiymətləndirilməsi.

Axın:
  1) upload — gruth_dataset.json -> Langfuse dataset (hər sual bir dataset item;
     item id-si sabitdir, təkrar upload dublikat yaratmır, yalnız yeniləyir).
  2) run    — dataset.run_experiment(): hər sual üçün agent (deep_agent.ask_detailed)
     işləyir, nəticə item-səviyyəli evaluator-lardan keçir, skorlar həmin
     trace-ə yazılır. Nəticələr Langfuse-da Datasets -> <dataset> -> Runs
     altında run-lar arası müqayisə olunur.

Evaluator-lar (hamısı SDK-da, prosesin içində işləyir):
  Deterministik (LLM-siz):
    tool_call_correct   — agent axtarış alətini lazım olanda çağırıb / lazım
                          olmayanda (imtina halları) çağırmayıb
    retrieval_recall    — gözlənilən maddələrin neçəsi agentə verilən
                          "Mətn parçaları"nda var
    citation_recall     — gözlənilən maddələrin neçəsi cavabın "İstinad olunan
                          maddələr" sətrində var
    citation_precision  — həmin sətirdəki maddələrin neçəsi gözlənilənlərdəndir
    no_false_citation   — (gözlənilən maddə yoxdursa) heç bir maddə göstərilməyib
  LLM hakim (judge) — TƏK LLM çağırışı, TƏK prompt; agenti bütün meyarlar
  üzrə birlikdə qiymətləndirir:
    judge_chunks_used_correctly — cavab, axtarışla əldə olunan kontekstə
                                  (maddələrə) sadiqdirmi, uydurma əlavə edibmi
    judge_answers_question      — cavab istifadəçinin sualına faktiki cavab
                                  verir, yoxsa yan keçir/əlaqəsizdir
    judge_element_coverage      — (YALNIZ expected_response_elements boş
                                  olmayan item-lərdə) etalon elementlərin neçə
                                  faizi cavabda məzmunca var
    judge_overall               — yuxarıdakı bütün tətbiq olunan meyarlara
                                  (element əhatəsi də daxil) əsaslanan ümumi
                                  keyfiyyət, 1-5
  Run-səviyyəli: yalnız judge_pass_rate (judge_overall >= 4 nisbəti). Hər
  skorun orta qiyməti Langfuse-un öz "Average Scores" bölməsində onsuz da
  göstərilir — bunu ikinci dəfə hesablamırıq.

.env-də olmalıdır: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL.
Hakim modeli defolt olaraq agentlə eyni modeldir (temperature=0). Fərqli
(tercihən daha güclü) hakim üçün: JUDGE_MODEL_NAME, JUDGE_MODEL_PROVIDER.

İstifadə:
    python eval_langfuse.py upload
    python eval_langfuse.py run --run-name Light_Rag-v1 --save-json eval_result_lg.json
    python eval_langfuse.py run --limit 5 --no-judge     # sürətli sınaq
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from langfuse import Evaluation
from pydantic import BaseModel, Field

from call_model import (
    DEFAULT_MODEL,
    _RETRY_DELAY_PATTERN,
    get_langfuse_client,
    get_model,
    invoke_with_retry,
)
from config import _DAILY_QUOTA_PATTERN
from logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "constitution-rag-gt_v3"
DEFAULT_FILE = Path(__file__).resolve().parent / "gruth_dataset.json"
EXPERIMENT_NAME = "constitution-rag_v3"

JUDGE_MAX_CONTEXT_CHARS = 14000
JUDGE_PASS_THRESHOLD = 4  # judge_overall >= 4 "keçdi" sayılır

_ARTICLE_RE = re.compile(r"(Maddə\s+\d+|Preambula)")
# format_context() chunk başlığı: "[Maddə 109, hissə 4]" (öz sətrində)
_CHUNK_HEADER_RE = re.compile(r"^\[(Maddə \d+|Preambula)(?:, hissə \d+)?\]\s*$", re.MULTILINE)
_CHUNK_SECTION = "## Mətn parçaları (Konstitusiya)"
_CITATION_LINE_RE = re.compile(r"İstinad olunan maddələr\s*:\s*(.+)", re.IGNORECASE)


# ============================================================
# 1) Dataset upload
# ============================================================
def upload(dataset_name: str, path: Path) -> None:
    lf = get_langfuse_client()
    items = json.loads(path.read_text(encoding="utf-8"))

    try:
        lf.create_dataset(
            name=dataset_name,
            description="Azərbaycan Konstitusiyası RAG agenti üçün etalon (ground truth) suallar",
            metadata={"source": path.name, "n_items": str(len(items))},
        )
        logger.info("dataset yaradıldı: %s", dataset_name)
    except Exception as exc:  # artıq mövcud ola bilər
        logger.info("create_dataset: %s — mövcud dataset ilə davam edilir", str(exc)[:120])

    failed: list[tuple[int, str]] = []
    for it in items:
        eb = it["expected_behavior"]
        try:
            lf.create_dataset_item(
                dataset_name=dataset_name,
                id=f"{dataset_name}-{it['id']}",  # sabit id: təkrar upload = yeniləmə
                input=it["question"],
                expected_output={
                    "expected_articles": eb.get("expected_articles") or [],
                    "expected_response_elements": eb.get("expected_response_elements") or [],
                },
                metadata={
                    "should_call_tool": bool(eb.get("should_call_tool")),
                    "target_tool_query": eb.get("target_tool_query"),
                },
            )
        except Exception as exc:
            failed.append((it["id"], str(exc)[:200]))
            logger.error("item id=%s yüklənmədi: %s", it["id"], str(exc)[:200])
            continue  # bir item-in xətası qalanların yüklənməsini dayandırmasın

    lf.flush()
    ok = len(items) - len(failed)
    if failed:
        logger.warning(
            "%d/%d item yükləndi, %d BAŞARISIZ (id-lər: %s). Səbəbləri yuxarıdakı "
            "ERROR sətirlərində — düzəldib təkrar 'upload' işlətsən, yalnız uğursuz "
            "olanlar deyil, hamısı yenidən göndərilir (upsert, zərəri yoxdur).",
            ok, len(items), len(failed), [i for i, _ in failed],
        )
    else:
        logger.info("%d/%d item uğurla yükləndi: %s", ok, len(items), dataset_name)


# ============================================================
# 2) Task — agenti işlədir
# ============================================================
def _with_retry(fn, max_retries: int = 4):
    """429 (dəqiqəlik limit) və keçici şəbəkə xətalarında (timeout, bağlantı)
    təkrar cəhd edir. Günlük limit bitibsə gözləməyin mənası yoxdur — dərhal
    xəta atır."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            err = str(exc)
            rate = "RESOURCE_EXHAUSTED" in err or "429" in err
            transient = rate or any(
                s in err for s in ("timed out", "Timeout", "ConnectionError", "ConnectionReset")
            )
            if not transient or attempt == max_retries or _DAILY_QUOTA_PATTERN.search(err):
                raise
            if rate:
                m = _RETRY_DELAY_PATTERN.search(err)
                wait = float(m.group(1)) + 3 if m else 20.0 * (attempt + 1)
            else:
                wait = 10.0 * (attempt + 1)  # timeout/bağlantı xətası — retryDelay yoxdur
            logger.warning("task: keçici xəta (%s) — %.0fs gözlənilir (cəhd %d/%d)", err[:80], wait, attempt + 1, max_retries)
            time.sleep(wait)


def make_task():
    from LightRAG.deep_agent import ask_detailed  # yalnız run üçün lazımdır (agent qurulur)

    def task(*, item, **kwargs):
        question = item["input"] if isinstance(item, dict) else item.input
        return _with_retry(lambda: ask_detailed(question))

    return task


# ============================================================
# 3) Deterministik evaluator-lar
# ============================================================
def _norm_article(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _expected_articles(expected_output) -> list[str]:
    eo = expected_output if isinstance(expected_output, dict) else {}
    return [_norm_article(a) for a in (eo.get("expected_articles") or [])]


def _expected_elements(expected_output) -> list[str]:
    eo = expected_output if isinstance(expected_output, dict) else {}
    return list(eo.get("expected_response_elements") or [])


def _cited_articles(answer: str) -> set[str]:
    """Cavabın SONUNDAKI 'İstinad olunan maddələr: ...' sətrindəki maddələr."""
    lines = _CITATION_LINE_RE.findall(answer or "")
    if not lines:
        return set()
    return {_norm_article(a) for a in _ARTICLE_RE.findall(lines[-1])}


def _retrieved_articles(contexts: list[str]) -> set[str]:
    """Agentə verilən 'Mətn parçaları' bölməsindəki chunk başlıqlarından maddələr
    (entity/əlaqə sətirlərindəki maddələr sayılmır — onlar mətn deyil)."""
    found: set[str] = set()
    for ctx in contexts:
        section = ctx.split(_CHUNK_SECTION)[-1] if _CHUNK_SECTION in ctx else ctx
        found.update(_norm_article(a) for a in _CHUNK_HEADER_RE.findall(section))
    return found


def retrieval_evaluator(*, input, output, expected_output, metadata, **kwargs):
    out = output if isinstance(output, dict) else {}
    expected_articles = _expected_articles(expected_output)
    should_call = bool((metadata or {}).get("should_call_tool", bool(expected_articles)))
    queries = out.get("tool_queries") or []
    called = bool(queries)

    evals = [Evaluation(
        name="tool_call_correct",
        value=1.0 if called == should_call else 0.0,
        comment=f"gözlənilən={should_call}, faktiki={called}; sorğular={queries}",
    )]

    cited = _cited_articles(out.get("answer", ""))
    if expected_articles:
        exp = set(expected_articles)
        retrieved = _retrieved_articles(out.get("contexts") or [])
        hit_cited = exp & cited
        evals += [
            Evaluation(
                name="retrieval_recall",
                value=len(exp & retrieved) / len(exp),
                comment=f"gözlənilən={sorted(exp)}, agentə verilən={sorted(retrieved)}",
            ),
            Evaluation(
                name="citation_recall",
                value=len(hit_cited) / len(exp),
                comment=f"gözlənilən={sorted(exp)}, istinad={sorted(cited)}",
            ),
            Evaluation(
                name="citation_precision",
                value=len(hit_cited) / len(cited) if cited else 0.0,
                comment=f"gözlənilən={sorted(exp)}, istinad={sorted(cited)}",
            ),
        ]
    else:
        evals.append(Evaluation(
            name="no_false_citation",
            value=1.0 if not cited else 0.0,
            comment=f"istinad olunan maddələr={sorted(cited)}",
        ))
    return evals


# ============================================================
# 4) LLM hakim (LLM-as-a-judge) — TƏK çağırış, TƏK prompt.
# Etalon elementlər (expected_response_elements) varsa, onlar da eyni promptda
# verilir və hakim agenti bütün meyarlar üzrə birlikdə qiymətləndirir; judge_overall
# da element əhatəsini nəzərə alır.
# ============================================================
class ElementVerdict(BaseModel):
    element: str = Field(description="Etalon element (verilən mətnlə eyni)")
    covered: bool = Field(description="Cavab bu elementi məzmunca əhatə edirmi")


class JudgeResult(BaseModel):
    chunks_used_correctly: bool = Field(
        description="Cavab, axtarışla əldə olunan kontekstdəki (maddələrdəki) "
                     "məlumatdan düzgün istifadə edib — uydurma əlavə etməyib, "
                     "kontekstə zidd getməyib, kontekstdə olmayan konkret hüquqi "
                     "iddia irəli sürməyib"
    )
    chunks_issue: str = Field(description="chunks_used_correctly=false olduqda qısa (1 cümlə) izah, əks halda boş sətir")
    answers_question: bool = Field(
        description="Cavab istifadəçinin SUALINA birbaşa, aydın cavab verir "
                     "(mövzudan yayınmır, yarımçıq qalmır, əlaqəsiz məlumat verməyib)"
    )
    answer_issue: str = Field(description="answers_question=false olduqda qısa (1 cümlə) izah, əks halda boş sətir")
    element_verdicts: list[ElementVerdict] = Field(
        default_factory=list,
        description="YALNIZ etalon elementlər verilibsə: hər element üçün bir qiymətləndirmə, "
                     "etalon sırası ilə. Etalon element verilməyibsə boş siyahı.",
    )
    overall: int = Field(
        ge=1, le=5,
        description="Ümumi keyfiyyət 1-5 — yuxarıdakı bütün tətbiq olunan meyarlara əsasən",
    )
    overall_reason: str = Field(description="1-2 cümlə izah")


JUDGE_PROMPT = """\
Sən Azərbaycan Respublikasının Konstitusiyası üzrə RAG köməkçisinin cavabını
qiymətləndirən hakimsən. YALNIZ aşağıda verilən sual/kontekst/cavaba əsaslan,
öz biliyinə görə hökm vermə.

İstifadəçinin sualı:
{question}

Köməkçinin axtarışla əldə etdiyi kontekst:
{context}

Köməkçinin cavabı:
{answer}
{elements_block}
Tapşırıq — aşağıdakı meyarları ayrıca qiymətləndir, sonra hamısını birlikdə
nəzərə alıb ümumi qiymət ver:

- chunks_used_correctly: cavabdakı konkret hüquqi iddialar (maddənin
  məzmunu, hüquq, səlahiyyət) yuxarıdakı kontekstlə dəstəklənirmi? Kontekstdə
  olmayan konkret iddia varsa false ver, chunks_issue-da qısaca yaz. Kontekst
  yoxdursa ("axtarış aləti çağırılmayıb") və cavab da konkret Konstitusiya
  iddiası irəli sürmürsə (məs. imtina və ya ümumi söhbət), true ver.

- answers_question: cavab istifadəçinin SUALINA faktiki cavab verirmi?
  Doğru məlumat versə belə, sualdan yayınırsa, yarımçıqdırsa və ya başqa bir
  şeydən bəhs edirsə, false ver, answer_issue-da qısaca yaz.
{elements_task}
- overall (1-5): yuxarıdakı bütün tətbiq olunan meyarlara birlikdə əsaslan.
  5 = hamısı tam doğru; 3 = bir meyar qismən problemlidir;
  1 = bir neçəsi, ya da biri ciddi şəkildə yanlışdır (uydurma, sualı
  cavablamır{overall_elements_hint}).
"""

_ELEMENTS_BLOCK = """
Cavabda olmalı olan elementlər (etalon):
{elements}
"""

_ELEMENTS_TASK = """
- element_verdicts: hər etalon element üçün, cavabın onu MƏZMUNCA əhatə
  edib-etmədiyini müəyyən et (söz-söz eyni olmaq lazım deyil, məna
  kifayətdir). Etalon sırası ilə, hər element üçün bir qiymətləndirmə ver.
"""


def make_judge_evaluator():
    model = get_model(
        model_name=os.getenv("JUDGE_MODEL_NAME") or None,
        provider=os.getenv("JUDGE_MODEL_PROVIDER") or None,
        temperature=0.0,
    )
    structured = model.with_structured_output(JudgeResult)

    def judge(*, input, output, expected_output, metadata, **kwargs):
        out = output if isinstance(output, dict) else {}
        contexts = out.get("contexts") or []
        context = "\n\n=====\n\n".join(contexts)[:JUDGE_MAX_CONTEXT_CHARS] or "(axtarış aləti çağırılmayıb)"
        elements = _expected_elements(expected_output)

        prompt = JUDGE_PROMPT.format(
            question=input if isinstance(input, str) else json.dumps(input, ensure_ascii=False),
            context=context,
            answer=out.get("answer", ""),
            elements_block=_ELEMENTS_BLOCK.format(
                elements="\n".join(f"{i}. {e}" for i, e in enumerate(elements, 1))
            ) if elements else "",
            elements_task=_ELEMENTS_TASK if elements else "",
            overall_elements_hint=", ya da etalon elementlərin çoxunu buraxır" if elements else "",
        )
        res = invoke_with_retry(structured, prompt)
        if res is None:
            raise ValueError("hakim model struktur qaytarmadı")

        evals = [
            Evaluation(
                name="judge_chunks_used_correctly",
                value=1.0 if res.chunks_used_correctly else 0.0,
                comment=res.chunks_issue or "problem tapılmadı",
            ),
            Evaluation(
                name="judge_answers_question",
                value=1.0 if res.answers_question else 0.0,
                comment=res.answer_issue or "sualı cavablandırır",
            ),
            Evaluation(name="judge_overall", value=float(res.overall), comment=res.overall_reason),
        ]

        if elements and res.element_verdicts:
            n = len(elements)
            covered = sum(1 for e in res.element_verdicts if e.covered)
            missing = [e.element for e in res.element_verdicts if not e.covered]
            evals.append(Evaluation(
                name="judge_element_coverage",
                value=min(covered, n) / n,
                comment=("əhatə olunmayanlar: " + "; ".join(missing)) if missing else "hamısı əhatə olunub",
            ))
        return evals

    return judge


# ============================================================
# 5) Run-səviyyəli evaluator — YALNIZ pass_rate (avg_* Langfuse-un öz
# "Average Scores" bölməsində onsuz da var, ikinci dəfə hesablamırıq).
# ============================================================
def pass_rate_evaluator(*, item_results, **kwargs):
    overall_scores = [
        float(ev.value)
        for r in item_results
        for ev in r.evaluations
        if ev.name == "judge_overall" and isinstance(ev.value, (int, float)) and not isinstance(ev.value, bool)
    ]
    if not overall_scores:
        return []
    passed = sum(1 for x in overall_scores if x >= JUDGE_PASS_THRESHOLD)
    return [Evaluation(
        name="judge_pass_rate",
        value=passed / len(overall_scores),
        comment=f"judge_overall >= {JUDGE_PASS_THRESHOLD}: {passed}/{len(overall_scores)}",
    )]


# ============================================================
# 6) Run
# ============================================================
def _export_items(result) -> list[dict]:
    """item_results-u sənin əvvəl gördüyün formata (trace_id, dataset_item_id,
    input, answer, tool_queries, contexts, scores) çevirir — notebook/skriptdə
    gruth_dataset.json ilə müqayisə üçün.

    Bəzi langfuse SDK versiyalarında item-in task-ı xəta versə (məs. timeout),
    o item item_results-da tam obyekt yox, yalnız {"trace_id","error"} kimi
    minimal formada qala bilir — bu halda "error" sahəsi ilə saxlanır, "input"/
    "answer" boş qalır (dataset_item_id da naməlum ola bilər)."""
    rows = []
    for r in result.item_results:
        if isinstance(r, dict):  # minimal uğursuz-item forması
            rows.append({
                "trace_id": r.get("trace_id"),
                "dataset_item_id": None,
                "input": None,
                "answer": None,
                "tool_queries": None,
                "contexts": None,
                "scores": {},
                "error": r.get("error"),
            })
            logger.warning("export: item uğursuz oldu (dataset_item_id naməlum): %s", r.get("error"))
            continue

        item = r.item
        out = r.output if isinstance(r.output, dict) else {}
        rows.append({
            "trace_id": r.trace_id,
            "dataset_item_id": getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None),
            "input": getattr(item, "input", None) or (item.get("input") if isinstance(item, dict) else None),
            "answer": out.get("answer"),
            "tool_queries": out.get("tool_queries"),
            "contexts": out.get("contexts"),
            "scores": {e.name: e.value for e in r.evaluations},
            "error": out.get("error"),
        })
    return rows


def run(
    dataset_name: str,
    run_name: str,
    limit: int | None,
    concurrency: int,
    use_judge: bool,
    save_json: Path | None,
) -> None:
    from LightRAG.deep_agent import MAX_ENTITIES, MAX_RELATIONS, SEARCH_TOP_K, _get_retriever

    lf = get_langfuse_client()
    dataset = lf.get_dataset(dataset_name)
    if limit:
        dataset.items = dataset.items[:limit]
    logger.info("dataset=%s, %d item, run=%s", dataset_name, len(dataset.items), run_name)

    # Retriever/qraf indeksini eksperimentdən ƏVVƏL yüklə: paralel task-lar eyni
    # anda ilk dəfə yükləməsin, indeks/kvota xətası da başlanğıcda görünsün.
    _get_retriever()

    evaluators = [retrieval_evaluator]
    if use_judge:
        evaluators.append(make_judge_evaluator())

    result = dataset.run_experiment(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        description="Konstitusiya RAG agenti — etalon suallar üzrə eval",
        task=make_task(),
        evaluators=evaluators,
        run_evaluators=[pass_rate_evaluator] if use_judge else [],
        max_concurrency=concurrency,
        metadata={
            "model": os.getenv("MODEL_NAME", DEFAULT_MODEL),
            "search_top_k": str(SEARCH_TOP_K),
            "max_entities": str(MAX_ENTITIES),
            "max_relations": str(MAX_RELATIONS),
            "judge": (os.getenv("JUDGE_MODEL_NAME") or os.getenv("MODEL_NAME", DEFAULT_MODEL)) if use_judge else "off",
        },
    )
    print(result.format(include_item_results=True))
    lf.flush()

    if save_json:
        rows = _export_items(result)
        save_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("item nəticələri yazıldı: %s (%d item)", save_json, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="gruth_dataset.json -> Langfuse dataset")
    up.add_argument("--dataset", default=DEFAULT_DATASET)
    up.add_argument("--file", default=str(DEFAULT_FILE))

    rn = sub.add_parser("run", help="agenti dataset üzərində işlət və qiymətləndir")
    rn.add_argument("--dataset", default=DEFAULT_DATASET)
    rn.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M"))
    rn.add_argument("--limit", type=int, default=None, help="yalnız ilk N item (sınaq üçün)")
    rn.add_argument("--concurrency", type=int, default=2, help="paralel sual sayı (429-dan qorunmaq üçün kiçik)")
    rn.add_argument("--no-judge", action="store_true", help="LLM hakimi söndür (yalnız deterministik metrikalar)")
    rn.add_argument(
        "--save-json", default=None,
        help="item nəticələrini (sual/cavab/kontekst/skorlar) bu fayla yaz (notebook üçün)",
    )

    args = parser.parse_args()
    if args.cmd == "upload":
        upload(args.dataset, Path(args.file))
    else:
        run(
            args.dataset, args.run_name, args.limit, args.concurrency, not args.no_judge,
            Path(args.save_json) if args.save_json else None,
        )


if __name__ == "__main__":
    main()