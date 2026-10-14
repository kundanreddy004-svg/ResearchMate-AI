import streamlit as st
import ollama
from ddgs import DDGS
from pypdf import PdfReader
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from xml.sax.saxutils import escape


st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #172554 100%);
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.feature-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 130px;
}
.feature-icon { font-size: 2rem; }
.feature-title { font-weight: 700; font-size: 1rem; margin-top: 8px; }
.feature-text { color: #cbd5e1; font-size: 0.85rem; }
.research-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}
.answer-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 22px;
    margin-top: 15px;
}
.source-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}
.paper-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}
section[data-testid="stSidebar"] { background: #0b1120; }
.stButton > button { border-radius: 10px; font-weight: 700; }
.stTextInput input { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

if "research_history" not in st.session_state:
    st.session_state.research_history = []

if "paper_data" not in st.session_state:
    st.session_state.paper_data = {}


def web_search(query, max_results=5):
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
    except Exception as e:
        st.warning(f"Web search unavailable: {e}")
    return results


def create_chunks(text, chunk_size=1200, overlap=200):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_embedding(text):
    try:
        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )
        embeddings = response.get("embeddings")
        if embeddings:
            return embeddings[0]
    except Exception:
        pass

    try:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Could not create embedding: {e}")


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a, dtype=np.float32)
    b = np.array(vector_b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = page_text.strip()

        page_chunks = create_chunks(
            page_text,
            chunk_size=1200,
            overlap=200
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk
            })

    if not chunks:
        return []

    embedded_chunks = []
    progress = st.progress(0)
    total = len(chunks)

    for index, item in enumerate(chunks, start=1):
        embedding = create_embedding(item["text"])

        embedded_chunks.append({
            "page": item["page"],
            "chunk": item["chunk"],
            "text": item["text"],
            "embedding": embedding
        })

        progress.progress(index / total)

    progress.empty()
    return embedded_chunks


def retrieve_relevant_chunks(question, chunks, top_k=5):
    if not chunks:
        return []

    question_embedding = create_embedding(question)
    scored_chunks = []

    for item in chunks:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        scored_chunks.append({
            "page": item["page"],
            "chunk": item["chunk"],
            "text": item["text"],
            "score": score
        })

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


def create_pdf(text, topic):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = [
        Paragraph(
            "ResearchMate AI - Research Report",
            title_style
        ),
        Spacer(1, 15),
        Paragraph(
            f"<b>Research Topic:</b> {escape(topic)}",
            body_style
        ),
        Spacer(1, 20)
    ]

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        safe_line = escape(line)

        if safe_line.startswith("**") and safe_line.endswith("**"):
            safe_line = safe_line[2:-2]
            story.append(
                Paragraph(
                    f"<b>{safe_line}</b>",
                    heading_style
                )
            )
        elif safe_line.startswith("#"):
            heading = safe_line.lstrip("#").strip()
            story.append(
                Paragraph(
                    f"<b>{heading}</b>",
                    heading_style
                )
            )
        else:
            story.append(
                Paragraph(
                    safe_line,
                    body_style
                )
            )

        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 ResearchMate AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your local AI-powered research assistant</div>',
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Local AI</div>
        <div class="feature-text">Powered by Llama 3.2</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌐</div>
        <div class="feature-title">Web Research</div>
        <div class="feature-text">Uses live web sources</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">RAG Analysis</div>
        <div class="feature-text">Semantic paper search</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">PDF Export</div>
        <div class="feature-text">Download research reports</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔬 ResearchMate AI")
    st.markdown("---")
    st.markdown("### ⚙️ System")

    st.success("🟢 Llama 3.2 Online")
    st.info("🌐 Web Search Enabled")
    st.info("🔗 Citation Support Enabled")
    st.info("🧠 RAG Enabled")
    st.info("📄 PDF Export Enabled")

    if st.session_state.paper_data:
        st.success(
            f"📚 {len(st.session_state.paper_data)} paper(s) loaded"
        )
    else:
        st.info("📚 No research papers loaded")

    st.markdown("---")
    st.markdown("### 📊 Session")

    st.metric(
        "Researches Generated",
        len(st.session_state.research_history)
    )

    if st.session_state.research_history:
        if st.button(
            "🗑️ Clear Research History",
            use_container_width=True
        ):
            st.session_state.research_history = []
            st.rerun()


# ============================================================
# RESEARCH PAPER UPLOAD
# ============================================================

st.markdown("---")
st.markdown("## 📄 Research Paper Analyzer")
st.write(
    "Upload research papers to build a local semantic search library."
)

uploaded_papers = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS UPLOADED PAPERS
# ============================================================

if uploaded_papers:
    for paper in uploaded_papers:
        paper_key = f"{paper.name}_{paper.size}"

        if paper_key not in st.session_state.paper_data:
            try:
                with st.spinner(f"🔍 Indexing {paper.name}..."):
                    chunks = process_pdf(paper)

                if chunks:
                    st.session_state.paper_data[paper_key] = {
                        "filename": paper.name,
                        "size": paper.size,
                        "chunks": chunks
                    }

                    st.success(
                        f"✅ {paper.name} indexed successfully."
                    )
                else:
                    st.warning(
                        f"⚠️ No readable text found in {paper.name}."
                    )

            except Exception as e:
                st.error(
                    f"❌ Could not index {paper.name}: {e}"
                )


# ============================================================
# DISPLAY LOADED PAPERS
# ============================================================

if st.session_state.paper_data:
    st.markdown("### 📚 Loaded Research Papers")

    for paper_key, paper_info in st.session_state.paper_data.items():
        chunks = paper_info["chunks"]

        pages = sorted(
            set(chunk["page"] for chunk in chunks)
        )

        st.markdown(
            f"""
            <div class="paper-card">
                <b>📄 {paper_info["filename"]}</b><br>
                Pages indexed: {len(pages)}<br>
                Semantic chunks: {len(chunks):,}<br>
                Embedding model: nomic-embed-text
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

if st.session_state.paper_data:
    st.markdown("---")
    st.markdown("## 💬 Ask Questions About Your Paper")

    paper_keys = list(st.session_state.paper_data.keys())

    paper_labels = [
        st.session_state.paper_data[key]["filename"]
        for key in paper_keys
    ]

    selected_label = st.selectbox(
        "Select a research paper",
        paper_labels
    )

    selected_index = paper_labels.index(selected_label)
    selected_key = paper_keys[selected_index]

    selected_paper = st.session_state.paper_data[selected_key]

    question = st.text_input(
        "Your question",
        placeholder="Example: What dataset was used in this study?"
    )

    ask_question = st.button(
        "🧠 Ask ResearchMate",
        use_container_width=True
    )

    if ask_question:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            try:
                with st.spinner(
                    "🔍 Finding the most relevant sections of the paper..."
                ):
                    relevant_chunks = retrieve_relevant_chunks(
                        question,
                        selected_paper["chunks"],
                        top_k=5
                    )

                if not relevant_chunks:
                    st.warning("No relevant information was found.")
                else:
                    retrieved_context = ""

                    for index, chunk in enumerate(
                        relevant_chunks,
                        start=1
                    ):
                        retrieved_context += f"""
[Retrieved Section {index}]
Page: {chunk["page"]}
Relevance Score: {chunk["score"]:.3f}

{chunk["text"]}

"""

                    with st.spinner(
                        "🧠 Llama 3.2 is analyzing the retrieved sections..."
                    ):
                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""
You are an academic research-paper assistant.

Research Paper:
{selected_paper["filename"]}

User Question:
{question}

Retrieved sections from the paper:

{retrieved_context}

Answer the user's question using ONLY the retrieved sections.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the retrieved sections do not contain enough information,
   say so clearly.
4. Cite important claims using the page number.
5. Use this exact format:
   [Page 3]
6. Only cite page numbers actually present in the retrieved sections.
7. Keep the answer clear and academically useful.
8. Use bullet points when appropriate.
"""
                                }
                            ]
                        )

                        answer = response["message"]["content"]

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown("### 🧠 ResearchMate Answer")
                    st.markdown(answer)

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("### 📌 Retrieved Paper Sections")

                    for index, chunk in enumerate(
                        relevant_chunks,
                        start=1
                    ):
                        with st.expander(
                            f"📄 Page {chunk['page']} — "
                            f"Relevance {chunk['score']:.3f}"
                        ):
                            st.write(chunk["text"])

            except Exception as e:
                st.error(f"❌ RAG question failed: {e}")


# ============================================================
# WEB RESEARCH
# ============================================================

st.markdown("---")
st.markdown("## 🌐 Web Research")

topic = st.text_input(
    "Research Topic",
    placeholder="Example: Artificial Intelligence in Healthcare",
    label_visibility="collapsed"
)

search_clicked = st.button(
    "🚀 Generate Research",
    use_container_width=True
)

if search_clicked:
    if not topic:
        st.warning("⚠️ Please enter a research topic first.")
    else:
        with st.spinner(
            "🔬 ResearchMate is searching the web and analyzing the information..."
        ):
            try:
                sources = web_search(
                    topic,
                    max_results=5
                )

                source_text = ""

                for i, source in enumerate(sources, 1):
                    source_text += f"""
[Source {i}]
Title: {source["title"]}
URL: {source["url"]}
Information: {source["snippet"]}

"""

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
You are an expert academic research assistant.

Research Topic:
{topic}

Web sources:

{source_text}

Create a clear academic research summary.

Include:
1. Short Introduction
2. Key Concepts
3. Major Applications
4. Advantages
5. Challenges
6. Future Scope
7. Five Useful Research Keywords

Citation rules:
- Use only the provided sources.
- Add citations after supported claims.
- Use exactly:
  [Source 1]
  [Source 2]
  [Source 3]
- Do not invent source numbers.
- Do not invent URLs.
- Do not invent references.
- Do not attach citations to unsupported claims.

Keep the answer academically structured.
"""
                        }
                    ]
                )

                result = response["message"]["content"]

                st.session_state.research_history.append({
                    "topic": topic,
                    "result": result
                })

                st.success(
                    "✅ Research generated successfully!"
                )

                st.markdown(
                    '<div class="research-box">',
                    unsafe_allow_html=True
                )

                st.markdown("## 📚 Research Summary")
                st.markdown(result)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### 📄 Export")

                pdf_file = create_pdf(
                    result,
                    topic
                )

                st.download_button(
                    label="📥 Download Research as PDF",
                    data=pdf_file,
                    file_name="ResearchMate_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                st.markdown("---")
                st.markdown("## 🔗 Sources & Citations")

                if sources:
                    for i, source in enumerate(sources, 1):
                        st.markdown(
                            f"""
                            <div class="source-card">
                            <b>[Source {i}]</b><br>
                            <a href="{source["url"]}" target="_blank">
                            {source["title"]}
                            </a>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if source["snippet"]:
                            st.caption(source["snippet"])
                else:
                    st.info("No web sources were found.")

            except Exception as e:
                st.error(
                    f"❌ Something went wrong: {e}"
                )


# ============================================================
# RESEARCH HISTORY
# ============================================================

if st.session_state.research_history:
    st.markdown("---")
    st.markdown("## 📚 Research History")

    for item in reversed(
        st.session_state.research_history
    ):
        with st.expander(
            f"🔎 {item['topic']}"
        ):
            st.markdown(item["result"])
