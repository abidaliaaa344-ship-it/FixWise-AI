import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="FixWise AI",
    page_icon="🔧",
    layout="wide"
)


# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------

def get_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ---------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------

def extract_pdf(file):
    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# ---------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------

def create_chunks(pages, chunk_size=1200):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        text = re.sub(r"\s+", " ", text).strip()

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            if len(chunk.strip()) > 100:

                chunks.append({
                    "page": page_number,
                    "text": chunk
                })

    return chunks


# ---------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------

def retrieve_chunks(query, chunks, top_k=4):

    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        documents + [query]
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indexes:

        if similarities[index] > 0:

            results.append({
                "page": chunks[index]["page"],
                "text": chunks[index]["text"],
                "score": similarities[index]
            })

    return results


# ---------------------------------------------------
# GROQ TROUBLESHOOTING
# ---------------------------------------------------

def troubleshoot(problem, retrieved_sections):

    client = get_groq_client()

    context = ""

    for section in retrieved_sections:

        context += f"""
PAGE {section['page']}
{section['text']}

"""

    system_prompt = """
You are FixWise AI, an equipment manual troubleshooting assistant.

Your job is to help users understand and safely follow
manufacturer documentation.

STRICT RULES:

1. Use only the supplied manual context for technical claims.
2. Do not invent specifications, procedures, limits or warnings.
3. If information is not available in the manual, clearly say so.
4. Clearly separate:
   - What the manual states
   - Possible checks
   - Safety warnings
   - Information not found
5. Never recommend bypassing safety devices.
6. Never recommend opening, repairing or modifying hazardous
   electrical, gas, pressure, refrigerant or high-energy systems
   unless the manual explicitly states that the user can do so.
7. Recommend qualified service personnel where appropriate.
8. Include the manual page number supporting every major recommendation.

Structure the response using:

### Problem Understanding
### Recommended Checks
### Safety Notes
### When to Contact a Technician
### Sources
"""

    user_prompt = f"""
USER PROBLEM:

{problem}

RELEVANT MANUAL CONTENT:

{context}

Provide a concise, evidence-grounded troubleshooting response.
"""

    response = client.chat.completions.create(
        model=st.secrets.get(
            "MODEL_NAME",
            "openai/gpt-oss-20b"
        ),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1200
    )

    return response.choices[0].message.content


# ---------------------------------------------------
# APP HEADER
# ---------------------------------------------------

st.title("🔧 FixWise AI")

st.subheader(
    "AI-powered troubleshooting grounded in product manuals"
)

st.caption(
    "Upload a manual → describe the issue → get source-grounded guidance"
)


st.info(
    "FixWise AI is a prototype decision-support tool. "
    "Always follow the manufacturer's safety instructions and "
    "use qualified technicians for hazardous repairs."
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        """
        FixWise AI uses Generative AI and Retrieval-Augmented
        Generation (RAG) to help users understand equipment
        manuals and troubleshoot problems.
        """
    )

    st.markdown("---")

    st.write("**Hackathon MVP**")

    st.write(
        """
        • PDF manual retrieval  
        • Groq Generative AI  
        • Source-based answers  
        • Safety-aware troubleshooting
        """
    )


# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------

st.header("1. Upload Equipment Manual")

uploaded_file = st.file_uploader(
    "Upload a PDF manual",
    type=["pdf"]
)


if uploaded_file:

    with st.spinner("Reading manual..."):

        pages = extract_pdf(uploaded_file)

        chunks = create_chunks(pages)

        st.session_state["manual_chunks"] = chunks

        st.session_state["manual_name"] = uploaded_file.name

    st.success(
        f"Manual loaded: {uploaded_file.name}"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Pages",
        len(pages)
    )

    col2.metric(
        "Knowledge Chunks",
        len(chunks)
    )


# ---------------------------------------------------
# PROBLEM
# ---------------------------------------------------

st.header("2. Describe the Problem")

problem = st.text_area(
    "What problem are you experiencing?",
    placeholder=(
        "Example: The air conditioner is running "
        "but the room is not cooling."
    ),
    height=120
)


# ---------------------------------------------------
# INVESTIGATE
# ---------------------------------------------------

investigate = st.button(
    "🔍 Troubleshoot with FixWise AI",
    type="primary",
    use_container_width=True
)


if investigate:

    if "manual_chunks" not in st.session_state:

        st.error(
            "Please upload an equipment manual first."
        )

    elif not problem.strip():

        st.error(
            "Please describe the problem."
        )

    else:

        with st.spinner(
            "Searching the manual..."
        ):

            retrieved = retrieve_chunks(
                problem,
                st.session_state["manual_chunks"]
            )

        if not retrieved:

            st.warning(
                "No sufficiently relevant information "
                "was found in the manual."
            )

        else:

            st.header("3. Relevant Manual Evidence")

            for i, result in enumerate(
                retrieved,
                start=1
            ):

                with st.expander(
                    f"Evidence {i} — Page {result['page']}"
                ):

                    st.write(
                        result["text"]
                    )

                    st.caption(
                        f"Retrieval similarity: "
                        f"{result['score']:.2f}"
                    )

            with st.spinner(
                "FixWise AI is preparing guidance..."
            ):

                answer = troubleshoot(
                    problem,
                    retrieved
                )

            st.header(
                "4. FixWise Recommendation"
            )

            st.markdown(answer)

            st.warning(
                "Always validate these recommendations "
                "against the complete manufacturer manual."
            )
