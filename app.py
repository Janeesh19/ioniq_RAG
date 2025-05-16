import os
import pathlib
import re
import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from wordllama import WordLlama

# ——— Page config ———
st.set_page_config(page_title="IONIQ 5 Sales RAG Assistant", page_icon="🚗", layout="centered")

# ——— Configuration ———
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GENAI_MODEL      = os.environ.get("GENAI_MODEL", "models/gemini-2.0-flash-001")
DATA_FILE_PATH   = os.environ.get("DATA_FILE_PATH", "ioniq.csv")
QDRANT_URL       = "https://2bb626d0-8e3b-4aa5-87ba-03803e523506.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZYF3sjRoh_oJE49OhXZji6f1yQqBQAccxpsz83TFPt4"
COLLECTION_NAME  = "rag_qa"

# ——— System Prompt ———
base_prompt = """
You are a professional automotive sales consultant.

VERY IMPORTANT INSTRUCTION:-
**DO NOT REPLY TO ANY OF THE QUESTION ANYTIME OTHER THAN IONIQ5. YOU ARE JUST SALES AGENT FOR IONIQ 5. THATS IT.DO NOT GO OUT OF THIS.JUST TALK ABOUT THE CAR**

*IMPORTANT INSTRUCTION:-*
**Use bullet points in giving answer about the question where ever necessary.keep it short and concise**
**after your answer to a question, in the next line suggest 1–2 questions that can help the customer based on the current question.**
**DON’T ADD SUGGESTED QUESTIONS IN BULLET POINTS.**

**Always greet the customer warmly before starting any conversation. Do not use structured response formats while greeting.**

Engage naturally in a multi-turn dialogue and always refer to previous conversation details to maintain continuity. Your communication must always be in ENGLISH. If the user asks a question in another language, politely ask them to continue in English.

Your primary role is to guide the customer towards making a confident and informed decision by:
1. Understanding their needs,
2. Providing relevant, clear answers,
3. Keeping the conversation engaging and friendly.

Your tone should be warm, helpful, and professional. Never rush to the end—build rapport as you go.

**Session Management:**
- If the user says goodbye (e.g., "bye", "goodbye", "see you", "talk later"), you must respond with a friendly closing and END the session.
- If the user is inactive for 2 minutes, politely end the session with a goodbye message.

**PRODUCT-SPECIFIC INSTRUCTION (Hyundai IONIQ 5 ONLY):**
You are representing the Hyundai IONIQ 5.
Do not answer any questions about other vehicles or unrelated topics. Focus solely on this model—its features, benefits, pricing, performance, interior/exterior, EV technology, financing, warranty, or test-drive process.

Your key objectives:
1. Close the sale by addressing the customer's concerns and creating a sense of urgency.
2. Be the customer's trusted expert on the Hyundai IONIQ 5.

If the customer's query is not related to the Hyundai IONIQ 5, politely refuse to answer.
"""

# ——— Load CSV at startup ———
file_path = pathlib.Path(DATA_FILE_PATH)
if not file_path.is_file():
    st.error(f"Data file not found at: {DATA_FILE_PATH}")
    st.stop()
with open(file_path, 'r', encoding='utf-8') as f:
    data_content = f.read()

# ——— Initialise Qdrant and WordLlama ———
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
wl = WordLlama.load(trunc_dim=64)
_dim = wl.embed(["_"])[0].shape[0]

# Ensure collection exists and ingest if needed
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=_dim, distance=Distance.COSINE),
    )
    lines = [ln for ln in data_content.splitlines() if ln.strip()]
    vectors = wl.embed(lines)
    points = [{"id": i, "vector": vec.tolist(), "payload": {"text": lines[i]}} for i, vec in enumerate(vectors)]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

# ——— Initialise Gemini client ———
client = genai.Client(api_key=GOOGLE_API_KEY)
model_name = GENAI_MODEL

# ——— Conversation state ———
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

# ——— Utility to record chat ———
def update_conversation(question: str, answer: str):
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))
    if len(st.session_state.history) > 40:
        st.session_state.history = st.session_state.history[-40:]

# ——— Generate a single response with RAG ———
def generate_response(question: str) -> str:
    # Retrieve context from Qdrant
    qvec = wl.embed([question])[0]
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec.tolist(),
        limit=5,
        with_payload=True
    )
    retrieved = "\n".join(hit.payload["text"] for hit in hits)

    # Build prompt
    recent = st.session_state.history[-4:] if len(st.session_state.history) >= 4 else st.session_state.history
    context_hist = "\n".join(f"{r.capitalize()}: {t}" for r, t in recent)
    prompt_parts = [
        base_prompt.strip(),
        "",  # separator
        "RETRIEVED INFORMATION:\n" + retrieved,
        "",  # separator
        context_hist,
        f"Customer: {question}" 
    ]
    full_prompt = "\n".join(part for part in prompt_parts if part)

    # Call Gemini
    resp = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(temperature=0.2, top_p=0.1),
    )
    text = resp.text or ""
    clean = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean = re.sub(r'(?i)^json\s*', '', clean)
    answer = clean.strip()

    update_conversation(question, answer)
    return answer

# ——— Streamlit UI ———
st.title("🚗 Hyundai IONIQ 5 Sales RAG Assistant")

# Display chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# User input and response
if user_input := st.chat_input("Ask something about the IONIQ 5…"):
    st.chat_message("user").write(user_input)
    answer = generate_response(user_input)
    st.chat_message("assistant").write(answer)
