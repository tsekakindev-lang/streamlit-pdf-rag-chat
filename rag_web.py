# pip install --upgrade pip
# pip install streamlit pymupdf langchain langchain-community langchain-text-splitters langchain-huggingface sentence-transformers

import os
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ======================
# Configuration
# ======================
st.set_page_config(page_title="本地中文 RAG 聊天机器人", layout="wide")
st.title("📚 本地 RAG 聊天机器人")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5

# 建议：normalize_embeddings=True 时，score_threshold 常见要 0.6~0.75
USE_SCORE_THRESHOLD = False  # 先关掉最稳，想开再改 True
SCORE_THRESHOLD = 0.65

OLLAMA_MODEL = "deepseek-r1:14b"      # 改成你本机 ollama 已拉取的模型名
CHROMA_DIR = "./chroma_db_1"
PDF_DIR = "./uploaded_pdfs"
EMBED_MODEL_PATH = "./bge-large-zh-v1.5"  # 本地 bge 模型路径

COLLECTION_NAME = "rag_collection"

# ======================
# Helpers (cache)
# ======================

@st.cache_resource
def load_embeddings():
    model_path = os.path.abspath(EMBED_MODEL_PATH)
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def load_llm():
    return OllamaLLM(model=OLLAMA_MODEL)

def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

def build_or_load_vectorstore(embeddings: HuggingFaceEmbeddings):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    return None


def make_retriever(vectorstore: Chroma):
    if USE_SCORE_THRESHOLD:
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
        )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def history_to_text(messages):
    """
    把 st.session_state.messages 转为较干净的文本 history
    """
    lines = []
    for m in messages:
        role = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines[-12:])  # 只取最近 12 轮，防止 prompt 过长

def clear_chat():
    st.session_state.messages = []
    # Optional: also clear any UI/state flags you use
    # st.session_state.vectorstore_ready = False
    st.rerun()

# ======================
# Sidebar: PDF Upload & Indexing
# ======================
ensure_dirs()

with st.sidebar:
    st.header("文档管理")

    uploaded_files = st.file_uploader(
        "上传 PDF 文件（支持多个）",
        type=["pdf"],
        accept_multiple_files=True,
    )

    reindex = st.button("🗂️ 重新索引文档")

    st.divider()
    if st.button("🧹 清空聊天记录", use_container_width=True):
        clear_chat()

    if reindex:
        if not uploaded_files and not os.listdir(PDF_DIR):
            st.warning("请先上传 PDF 文件")
            st.stop()

        with st.spinner("正在处理文档并构建索引..."):
            # 保存上传的文件（追加/覆盖同名）
            for f in uploaded_files or []:
                save_path = os.path.join(PDF_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())

            # 读取目录中所有 pdf
            docs = []
            for filename in os.listdir(PDF_DIR):
                if filename.lower().endswith(".pdf"):
                    loader = PyMuPDFLoader(os.path.join(PDF_DIR, filename))
                    docs.extend(loader.load())

            if not docs:
                st.error("未加载到任何文档内容（PDF 为空或读取失败）")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = splitter.split_documents(docs)
            st.info(f"分割为 {len(chunks)} 个文本块")

            embeddings = load_embeddings()

            import chromadb

            # 只刪同一個 collection（不刪資料夾，避免 WinError 32）
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

            # 建新索引（用同一個 collection_name）
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
            )

            st.session_state.vectorstore_ready = True
            st.success("✅ 索引完成！可以开始聊天了")


# ======================
# Build Chain (LCEL)
# ======================
embeddings = load_embeddings()
llm = load_llm()

vectorstore = build_or_load_vectorstore(embeddings)

if vectorstore is not None:
    retriever = make_retriever(vectorstore)

#- 允许中英文混合，优先
    PROMPT_TEMPLATE = """
你是一个严谨但表达自然的助手。请严格根据【上下文】回答问题。
- 如果上下文中没有相关信息，请直接回答：“文档中未找到相关内容。”
- 用中文回答。

【上下文】
{context}

【对话历史】
{chat_history}

【用户问题】
{input}

【回答】
""".strip()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "input"],
    )

    # stuff chain：把检索到的 docs 塞进 prompt 的 {context}
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # retrieval chain：输入 {"input": "..."}，内部先检索，再把 docs 交给 combine_chain
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    st.session_state.rag_chain = rag_chain
else:
    st.session_state.rag_chain = None


# ======================
# Chat Interface
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.rag_chain is None:
    st.info("👆 请先在左侧上传 PDF 并点击“重新索引文档”，或确认本地已有索引目录。")
    st.stop()

# 输入框
user_q = st.chat_input("请输入您的问题...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            chat_history_text = history_to_text(st.session_state.messages[:-1])

            # LCEL chain invoke
            result = st.session_state.rag_chain.invoke(
                {
                    "input": user_q,
                    "chat_history": chat_history_text,
                }
            )

            # create_retrieval_chain 的典型输出：
            # result["answer"] -> 模型回答
            # result["context"] -> 检索到的 Document 列表
            answer = result.get("answer", "").strip()
            st.markdown(answer)

            # 引用来源
            with st.expander("📑 查看引用来源"):
                ctx_docs = result.get("context", []) or []
                if not ctx_docs:
                    st.write("（本次未检索到匹配片段）")
                else:
                    for i, doc in enumerate(ctx_docs, start=1):
                        src = doc.metadata.get("source", "未知文件")
                        page = doc.metadata.get("page", "?")
                        st.write(f"**来源 {i}**：{src}（第 {page} 页）")
                        st.write(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
