import json
import re
import tempfile
import uuid
from pathlib import Path

import requests
import streamlit as st

from config import CHECKPOINT_PATH
from embed_gemini import chunk_by_article, embed_chunks, extract_pdf_text

API_URL = "http://127.0.0.1:8001/v1/chat/completions"
RELOAD_URL = "http://127.0.0.1:8001/reload"
MODEL_ID = "konstitusiya-agent"
REQUEST_TIMEOUT = 120  # saniyə — agent tool çağırışları vaxt apara bilər

st.set_page_config(
    page_title="Konstitusiya Agent",
    page_icon="⚖️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    section[data-testid="stSidebar"] { width: 300px !important; }
    .conv-btn button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def _new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": None, "messages": []}
    st.session_state.active_id = conv_id
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_id" not in st.session_state or (
    st.session_state.active_id not in st.session_state.conversations
):
    _new_conversation()


def _active_conv() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def _short_title(text: str, max_len: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def call_api(messages: list[dict]) -> str:
    """api.py-dəki /v1/chat/completions endpoint-inə OpenAI formatında
    sorğu göndərir və assistant cavabını qaytarır. Server ayrıca prosesdə
    (uvicorn) işə salınmalıdır — bax modulun yuxarısındakı qeydə."""
    try:
        resp = requests.post(
            API_URL,
            json={"model": MODEL_ID, "messages": messages, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ API serverə qoşula bilmədim. Əvvəlcə ayrı terminalda "
            "`uvicorn api:app --host 127.0.0.1 --port 8001` ilə işə sal."
        )
    except Exception as e:
        return f"⚠️ Xəta baş verdi: {e}"


# ============================================================
# Sənəd embedding — PDF / JSON yükləyib checkpoint-ə əlavə etmək
# ============================================================
def _slugify(name: str) -> str:
    """Fayl adından checkpoint id-lərini namespace etmək üçün qısa slug —
    fərqli sənədlərin eyni id-li (məs. 'madde_1') chunk-ları toqquşmasın."""
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return slug or "doc"


def _load_chunks_from_upload(uploaded_file) -> list[dict]:
    """Yüklənmiş PDF və ya JSON-u [{"id","text"}, ...] chunk siyahısına
    çevirir. JSON həm sadə mətn siyahısı, həm {"id","text"} siyahısı, həm
    də {"chunks": [...]} formasında ola bilər. PDF olduqda
    embed_gemini.py-dəki eyni chunk_by_article() (Maddə N. başlıqları)
    istifadə olunur."""
    slug = _slugify(uploaded_file.name)
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".json":
        data = json.load(uploaded_file)
        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]
        chunks = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                chunks.append({"id": f"{slug}__{i}", "text": item})
            elif isinstance(item, dict) and item.get("text"):
                raw_id = item.get("id", i)
                chunks.append({"id": f"{slug}__{raw_id}", "text": item["text"]})
        return chunks

    # Defolt: PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    text = extract_pdf_text(tmp_path)
    raw_chunks = chunk_by_article(text)
    return [{"id": f"{slug}__{c['id']}", "text": c["text"]} for c in raw_chunks]


def _trigger_index_reload() -> str:
    """embed_chunks() checkpoint.jsonl-ə yazır, amma API server (uvicorn)
    Retriever-i yalnız ilk çağırışda yükləyib yaddaşda saxlayır (bax
    deep_agent.py-dəki _get_retriever) — ona görə yeni chunk-ların dərhal
    axtarışda görünməsi üçün serverə /reload göndəririk."""
    try:
        resp = requests.post(RELOAD_URL, timeout=30)
        if resp.ok:
            n = resp.json().get("chunks", "?")
            return f"API serverdəki index yeniləndi (indi {n} chunk)."
        return "API server index-i yeniləyə bilmədi (server xəta qaytardı)."
    except requests.exceptions.ConnectionError:
        return "API server işləmir — server işə düşəndə yeni chunk-lar avtomatik yüklənəcək."
    except Exception as e:
        return f"Index yenilənərkən xəta: {e}"


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### ⚖️ Konstitusiya Agent")

    if st.button("➕ Yeni Söhbət", use_container_width=True):
        _new_conversation()
        st.rerun()

    st.markdown("---")
    st.caption("Söhbətlər")

    # Ən son yaradılan söhbət yuxarıda görünsün
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        label = conv["title"] or "Yeni söhbət"
        is_active = conv_id == st.session_state.active_id

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                ("🟢 " if is_active else "") + label,
                key=f"conv_{conv_id}",
                use_container_width=True,
            ):
                st.session_state.active_id = conv_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                del st.session_state.conversations[conv_id]
                if st.session_state.active_id == conv_id:
                    if st.session_state.conversations:
                        st.session_state.active_id = next(
                            iter(st.session_state.conversations)
                        )
                    else:
                        _new_conversation()
                st.rerun()

    st.markdown("---")
    st.caption("📄 Sənəd əlavə et (embedding)")
    uploaded_doc = st.file_uploader(
        "PDF və ya JSON (chunk siyahısı)",
        type=["pdf", "json"],
        key="doc_uploader",
        label_visibility="collapsed",
    )
    if st.button(
        "📥 Embed et",
        use_container_width=True,
        disabled=uploaded_doc is None,
    ):
        with st.spinner("Sənəd chunk-lanır və embed olunur — bu bir neçə dəqiqə çəkə bilər..."):
            try:
                chunks = _load_chunks_from_upload(uploaded_doc)
                if not chunks:
                    st.warning("Sənəddən heç bir chunk çıxarıla bilmədi.")
                else:
                    embed_chunks(chunks, checkpoint_path=CHECKPOINT_PATH)
                    reload_msg = _trigger_index_reload()
                    st.success(f"✅ {len(chunks)} chunk embed olunub checkpoint-ə əlavə edildi. {reload_msg}")
            except Exception as e:
                st.error(f"Xəta: {e}")


# ============================================================
# Əsas chat sahəsi
# ============================================================
conv = _active_conv()

st.title("Azərbaycan Konstitusiyası üzrə Agent")
st.caption("Suallarınızı verin — agent lazım gəldikdə Konstitusiya mətninə istinad edəcək.")

chat_col = st.container()

with chat_col:
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

question = st.chat_input("Sualınızı yazın...")

if question:
    conv["messages"].append({"role": "user", "content": question})
    if conv["title"] is None:
        conv["title"] = _short_title(question)

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Cavab hazırlanır..."):
                api_messages = [
                    {"role": m["role"], "content": m["content"]} for m in conv["messages"]
                ]
                answer = call_api(api_messages)
            st.markdown(answer)

    conv["messages"].append({"role": "assistant", "content": answer})
    st.rerun()