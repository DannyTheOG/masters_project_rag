#####################  IMPORTING MODULES AND INITIALIZING VARIABLES   ######################
import os
import json
import re
import pandas as pd
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever  # RAG OPTIMISATION 2: Multi-Query Retrieval

# load environment variables
load_dotenv()  

###############################   INITIALIZE EMBEDDINGS MODEL  ##########################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   ######################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

###############################   RETRIEVER HELPERS   ###################################################

# ── RAG OPTIMISATION 1: Metadata Filtering ──────────────────────────────────────────────────────────
# Maps keywords from the patient's diagnosis / primary_concern to the therapy_focus
# tags stored in ChromaDB (e.g. "phonology|articulation").
# When a match is found, a $in filter is built from _KNOWN_FOCUS_VALUES so only chunks
# from clinically relevant documents are considered — preventing irrelevant therapy areas
# (e.g. stammering docs when treating phonology) from occupying retrieval slots.
#
# NOTE: ChromaDB does not support $contains on metadata fields. Instead we pre-enumerate
# every pipe-joined therapy_focus string that exists in the vector store and use $in,
# selecting only those values whose string contains the hint as a substring.

_FOCUS_KEYWORDS = {
    "phonology":      ["phonol", "cluster", "minimal pair", "fronting", "backing", "gliding"],
    "articulation":   ["articul", "speech sound", "lisp"],
    "stammering":     ["stammer", "stutter", "fluency", "dysfluenc"],
    "early_language": ["language", "vocab", "early interven", "intensive interaction"],
}

# All unique pipe-joined therapy_focus strings written to ChromaDB during ingestion.
# Update this set whenever new documents with new focus combinations are added.
_KNOWN_FOCUS_VALUES = {
    "phonology",
    "articulation",
    "stammering",
    "early_language",
    "phonology|articulation",
    "articulation|early_language",
    "early_language|stammering",
    "phonology|articulation|stammering|early_language",
    "phonology|articulation|early_language|stammering",
}


def get_therapy_focus(diagnosis: str, primary_concern: str) -> str:
    """Return the best-matching therapy_focus tag for the current patient.
    Returns '' (no filter) if nothing matches, so retrieval stays broad."""
    combined = (diagnosis + " " + primary_concern).lower()
    for focus, keywords in _FOCUS_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return focus
    return ""


def build_base_retriever(therapy_focus_hint: str = ""):
    """Build an MMR retriever, optionally scoped to a specific therapy_focus.

    Uses a $in filter against _KNOWN_FOCUS_VALUES — selecting every stored
    pipe-joined string that contains the hint as a substring (e.g. hint='phonology'
    matches 'phonology', 'phonology|articulation', etc.).
    Falls back to searching all documents when no hint is provided.
    """
    search_kwargs: dict = {"k": 6, "fetch_k": 20}

    if therapy_focus_hint:
        # Build the $in list: every known stored value that contains the hint.
        matching_values = [
            v for v in _KNOWN_FOCUS_VALUES if therapy_focus_hint in v
        ]
        if matching_values:
            search_kwargs["filter"] = {"therapy_focus": {"$in": matching_values}}
        # If no match found (shouldn't happen), fall through with no filter.

    return vector_store.as_retriever(
        search_type="mmr",   # MMR ensures diverse results rather than near-duplicate chunks
        search_kwargs=search_kwargs,
    )


# ── RAG OPTIMISATION 3: Evidence-question detection ──────────────────────────────────────────────
# When the user asks a WHY / rationale question (e.g. "why 10-minute sessions?",
# "what does the research say?"), retrieval should search the FULL medbase —
# including research papers — not be narrowed to a single therapy_focus tag.
# Without this, academic papers (Hegarty, McFaul, Sugden, etc.) are outcompeted by
# practice handouts whose language more closely matches therapy-specific surface terms.

_EVIDENCE_KEYWORDS = [
    "why", "reason", "because", "evidence", "research", "study", "studies", "based on", "what does", "how do we know",
    "is it important", "does it help", "what is the", "explain",
]

def is_evidence_question(question: str) -> bool:
    """Return True when the question is asking for rationale or research backing.
    These questions should search the full medbase (no therapy_focus filter)
    so research papers are included in the retrieval candidate pool."""
    q = question.lower()
    return any(kw in q for kw in _EVIDENCE_KEYWORDS)


# ── RAG OPTIMISATION 4: Minimal-pairs CSV lookup ─────────────────────────────────────────────────
# When the user asks for minimal pairs, the LLM cannot reliably generate correct
# phoneme contrasts from the error_patterns field — it hallucinates rhyming words
# instead. This lookup queries phoneme_minimal_pairs.csv deterministically and
# injects the real pairs into the context so the LLM just has to format them.

_MP_CSV_PATH = os.path.join(os.path.dirname(__file__), "phoneme_minimal_pairs.csv")
_mp_df: pd.DataFrame | None = None   # loaded once on first use

# Maps "X→Y substitution" patterns to the phoneme_contrast codes in the CSV.
# Both orderings are included because the CSV uses e.g. 'w-r' (not 'r-w').
_SUBSTITUTION_TO_CONTRAST = {
    ("r", "w"): "w-r",   ("w", "r"): "w-r",
    ("r", "l"): "l-r",   ("l", "r"): "l-r",
    ("s", "th"): "th-f", ("th", "s"): "th-f",
    ("f", "th"): "th-f", ("th", "f"): "th-f",
    ("s", "sh"): "s-sh", ("sh", "s"): "s-sh",
    ("s", "z"): "s-z",   ("z", "s"): "s-z",
    ("t", "d"): "t-d",   ("d", "t"): "t-d",
    ("k", "g"): "k-g",   ("g", "k"): "k-g",
    ("p", "b"): "p-b",   ("b", "p"): "p-b",
    ("m", "n"): "m-n",   ("n", "m"): "m-n",
    ("f", "v"): "f-v",   ("v", "f"): "f-v",
    ("ch", "j"): "ch-j", ("j", "ch"): "ch-j",
}


def _load_mp_df() -> pd.DataFrame:
    global _mp_df
    if _mp_df is None:
        _mp_df = pd.read_csv(_MP_CSV_PATH)
    return _mp_df


def _parse_contrasts_from_error_patterns(error_patterns: str) -> list[str]:
    """Extract phoneme_contrast codes from a free-text error_patterns string.
    Handles formats like 'r→w substitution', 'r->w', 'r/w', 's-z' etc."""
    contrasts: list[str] = []
    # Normalise arrows and separators
    normalised = error_patterns.lower().replace("→", "->").replace("/", "->").replace(" ", "")
    # Find all X->Y or X→Y patterns
    for match in re.finditer(r'([a-z]+)->([a-z]+)', normalised):
        src, tgt = match.group(1), match.group(2)
        code = _SUBSTITUTION_TO_CONTRAST.get((src, tgt))
        if code and code not in contrasts:
            contrasts.append(code)
    return contrasts


def is_minimal_pairs_question(question: str, prev_was_minimal_pairs: bool = False) -> bool:
    """Return True when the user is asking for minimal pairs,
    OR when the question describes a substitution error in natural language
    (e.g. 'says wabbit for rabbit', 'X instead of Y') — these questions
    always benefit from minimal pair context being injected.
    """
    q = question.lower()
    # Direct request
    if "minimal pair" in q or "minimal-pair" in q:
        return True
    # Follow-up within an ongoing minimal-pairs exchange
    follow_up_keywords = ["more", "another", "additional", "give me more",
                          "more examples", "more pairs", "show me more"]
    if prev_was_minimal_pairs and any(kw in q for kw in follow_up_keywords):
        return True
    # Natural-language substitution description
    # e.g. "says 'wabbit' instead of 'rabbit'", "wed for red", "error pattern"
    substitution_phrases = ["instead of", "substitutes", "replaces"]
    if any(kw in q for kw in substitution_phrases):
        return True
    return False


def get_minimal_pairs_context(error_patterns: str, n: int = 8) -> str:
    """Return a formatted string of n real minimal pairs sourced from the CSV,
    matched to the patient's error pattern. Returns '' if no match found."""
    if not error_patterns or error_patterns == "Not specified":
        return ""
    contrasts = _parse_contrasts_from_error_patterns(error_patterns)
    if not contrasts:
        return ""
    df = _load_mp_df()
    results: list[str] = []
    for contrast in contrasts:
        subset = df[df["phoneme_contrast"] == contrast].drop_duplicates(
            subset=["word_A", "word_B"]
        )
        sample = subset.sample(min(n, len(subset)), random_state=42)
        pairs = [f"{row.word_A} / {row.word_B}" for _, row in sample.iterrows()]
        results.append(
            f"[Minimal Pairs — {contrast} contrast] (Source: phoneme_minimal_pairs.csv)\n"
            + "\n".join(pairs)
        )
    return "\n\n".join(results)

###############################   INITIALIZE CHAT MODEL   ###############################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0.7,   # 0 = fully deterministic → same answer every time
)



prompt = PromptTemplate.from_template("""
You are a clinically-aligned Speech and Language Therapy (SLT) support assistant.

You must ONLY use the therapist-approved information provided in the retrieved context.
Do NOT invent exercises, targets, diagnoses, or advice that is not explicitly included.
If information is missing, say: 'I do not have enough therapist-approved information to answer that safely.'



You are currently supporting this child:
- Age: {age} | Diagnosis: {diagnosis}
- Primary concern: {primary_concern}
- Therapy stage: {therapy_stage}
- Target sounds: {target_sounds}
- Error patterns: {error_patterns}
- Last session accuracy: {accuracy_last_session}%

Family context:
- Parent engagement: {parent_engagement} | Confidence: {parent_confidence}
- Home practice: {home_practice_compliance} | Feedback style: {reinforcement_style}

The following clinical notes are provided as BACKGROUND EVIDENCE ONLY. \
Do NOT copy them as activity templates, follow their format/structure, or \
adopt their clinical writing style. Translate any clinical language into \
plain, warm, parent-friendly English. Write your response from scratch based \
on the child's specific profile.

{retrieved_documents}

A few ground rules:
- Match the response to what is actually being asked. A quick question gets a short, \
direct answer. A request for a session plan or activity gets a clear structured response.
- LANGUAGE REGISTER: Responses are read by parents, not clinicians. Use plain, \
conversational English. Do NOT use IPA phonetic notation (e.g. /rɛd/, /ɹ/). \
Do NOT use unexplained acronyms or clinical passive voice ('the child is expected \
to show...'). Write as if talking to a capable, caring parent with no SLT training.
- ACCURACY-BASED PRACTICE RULES — apply strictly, do not hedge or blend levels: \
  • Below 60% → Parent/therapist models EVERY production. Child imitates only. \
    NO independent practice. NO generalisation tasks. NO spontaneous speech tasks. \
  • Above 80% → Child practises more independently. Begin generalisation to phrases/sentences.
- Do not introduce new therapy techniques or change existing goals.
- Do not give medical diagnoses or advice outside SLT scope.
- If you genuinely don't have enough evidence to answer safely, say so briefly.
- Don't repeat advice already covered in this conversation unless asked.
- NEVER start your response by describing, summarising, or narrating the user's \
request (e.g. do not say "The user is asking for..."). Begin your answer directly.
- NEVER invent previous sessions, shared history, or an ongoing relationship that \
is not in the conversation history above. If the chat history is empty, do not \
reference "our previous sessions", "as we discussed", or "last time". Only \
reference what is actually present in the conversation history provided.
- TRUSTWORTHINESS: If asked "can I trust you?", "how do you know what's right?", \
or any meta-question about your reliability, answer honestly: \
(1) You are an AI decision-support tool — NOT a clinician or a clinical team. \
(2) Your advice is grounded in two things: the child's profile entered by the \
therapist, and clinical documents retrieved from the therapy toolkit. \
(3) You do NOT run clinical trials, follow unnamed institutional guidelines, or \
make final clinical decisions. The treating SLT clinician should be consulted \
for final clinical decisions. \
Do NOT over-reassure. Do NOT claim institutional authority you don't have.
- When asked how to tell if home practice is working, give SPECIFIC, COUNTABLE \
behavioural indicators the parent can observe without clinical training. \
 Do not give vague answers like 'you will notice improvements'.
- When minimal pairs appear in the retrieved notes above, present ONLY those exact \
pairs — do not generate, substitute, or invent your own. If no pairs are in the \
notes, say so rather than guessing.
- When providing minimal pairs, format each pair as: word / word (one pair per line).
- SUBSTITUTION ERRORS: When a parent describes a child saying WORD_A instead of \
WORD_B (e.g. 'wabbit' for 'rabbit', 'wed' for 'red'), this signals a phonological \
substitution pattern. Respond by: (1) naming the substitution type (e.g. r→w), \
(2) recommending minimal pair contrasts from the retrieved notes above at word level \
with heavy parent modelling (since accuracy is below 60%), (3) NOT recommending \
generalisation across word positions — stay at the position where the error occurs.
- NON-SPEECH ORAL MOTOR EXERCISES (NSOMEs): If asked about tongue exercises, \
blow exercises, or oral motor drills for improving speech sounds, Direct sound practice is \
more effective. Do not endorse NSOMEs as a primary strategy.
- THERAPY HIERARCHY — use this when explaining why we stay at a certain stage: \
  The evidence-based progression is: isolation → syllable → word → phrase → \
  sentence → conversation. Advancing before a stage is consolidated (typically \
  70–80% accuracy) causes the new sound to break down under the extra demands of \
  longer speech, and the error gets practised instead of corrected. When explaining \
  this to a caregiver, name the stages, state the accuracy benchmark (70–80%), \
  and explain the error-reinforcement risk of advancing too early.
- DEVELOPMENTAL NORMS — use this when a parent asks if a child will "grow out of it" \
  or compares to another child's experience: \
  /r/ is one of the LAST sounds children typically acquire — most children develop it \
  naturally between ages 6 and 8. So a neighbour's experience of their child growing \
  out of it is plausible and should NOT be dismissed. \
  However, when answering for a specific child in active therapy, give a CALIBRATED \
  response: (1) acknowledge the neighbour's experience is genuinely possible, \
  (2) explain that the child's diagnosed delay and current therapy stage mean structured \
  support gives the best outcome rather than watchful waiting, \
  (3) reference the child's specific age and diagnosis from their profile. \
  Do NOT over-reassure ('he'll definitely be fine') or over-worry ('this is serious'). \
  The tone should be: "yes, that can happen — here's why we're not leaving it to chance."

Read the user's request carefully and decide the best response type:

• If the request asks for a session plan, activity, exercises, or practice, \
use this structured coaching format:

  **Session Focus:** (one sentence goal)
  **Activity:** (step-by-step instructions)
  **Practice Examples:** (at least 5 age-appropriate word examples)
  **Parent Coaching Script:** (actual words the parent can say)
  **Encouragement:** (short warm message for the child and family)

• If the request is conversational (e.g. a question, follow-up, clarification, or \
short answer) → respond naturally and concisely, like a knowledgeable colleague. \
Do NOT force the coaching template into every reply.

Conversation so far:
{chat_history}

Question: {input}
Response:"""
)
# For /r/, start with INITIAL position words (red, run, ring) — not final /r/ (car, far). \


################################### STREAMLIT APP ########################################################

st.set_page_config(page_title="SLT SUPPORT CHATBOT", page_icon="🤖", layout="wide")
st.title("🤖 SLT SUPPORT CHATBOT")

# -------------------------
# Load patients from JSON
# -------------------------
_patients_path = os.path.join(os.path.dirname(__file__), "patients.json")
with open(_patients_path, "r") as _f:
    _patients_data = json.load(_f)["patients"]

# Build lookup: name → patient dict
_patient_lookup = {p["patient_id"]: p for p in _patients_data}
_patient_names  = ["— Select a patient —"] + list(_patient_lookup.keys())

# -------------------------
# SIDEBAR — Patient & Parent Profile
# -------------------------
with st.sidebar:
    st.header("🧒 Patient Profile")

    # ── Patient dropdown ──────────────────────────────────────────
    selected_name = st.selectbox("Patient", _patient_names)

    # Resolve selected patient (None if placeholder chosen)
    _pt = _patient_lookup.get(selected_name, {})
    _pr = _pt.get("parent", {})

    # Helper: return pre-filled value if patient loaded, else ""
    def _pv(key, default=""):
        return _pt.get(key, default)

    # ── Therapy stage options ─────────────────────────────────────
    _stage_options = [
        "Sound in Isolation",
        "Structured Word Level",
        "Word Level",
        "Phrase Level",
        "Early Intervention",
        "Mid Therapy",
        "Consolidation",
        "Discharge Planning",
    ]
    _stage_default = _pv("therapy_stage", _stage_options[0])
    _stage_idx     = _stage_options.index(_stage_default) if _stage_default in _stage_options else 0

    # ── Patient fields (pre-populated from JSON) ──────────────────
    age                   = st.text_input("Age",             value=_pv("age"),             placeholder="e.g. 4 years")
    diagnosis             = st.text_input("Diagnosis",       value=_pv("diagnosis"),       placeholder="e.g. Speech Sound Disorder")
    primary_concern       = st.text_input("Primary Concern", value=_pv("primary_concern"), placeholder="e.g. Cluster reduction – /s/, /r/")
    therapy_stage         = st.selectbox("Therapy Stage",    _stage_options,               index=_stage_idx)
    target_sounds         = st.text_input("Target Sounds",   value=_pv("target_sounds"),   placeholder="e.g. /s/, /r/")
    error_patterns        = st.text_input("Error Patterns",  value=_pv("error_patterns"),  placeholder="e.g. r→w substitution, cluster reduction")
    accuracy_last_session = st.slider("Last Session Accuracy (%)", 0, 100, int(_pv("accuracy_last_session", 0)))

    st.divider()
    st.header("👨‍👩‍👧 Parent Context")

    # ── Engagement ───────────────────────────────────────────────
    _eng_options  = ["High", "Moderate", "Low"]
    _eng_default  = _pr.get("engagement_level", "Moderate")
    _eng_idx      = _eng_options.index(_eng_default) if _eng_default in _eng_options else 1
    parent_engagement = st.selectbox("Parent Engagement Level", _eng_options, index=_eng_idx)

    # ── Confidence (not in JSON; keep as manual selectbox) ───────
    parent_confidence = st.selectbox("Parent Confidence", ["High", "Moderate", "Low"], index=1)

    # ── Home practice compliance ──────────────────────────────────
    _comp_options = ["Consistent", "Occasional", "Rarely"]
    _comp_raw     = _pr.get("home_practice_compliance", "")
    # Map the free-text JSON value to the nearest option
    if "Consistent" in _comp_raw or "5" in _comp_raw or "6" in _comp_raw:
        _comp_idx = 0
    elif "3" in _comp_raw or "4" in _comp_raw:
        _comp_idx = 1
    else:
        _comp_idx = 2
    home_practice_compliance = st.selectbox("Home Practice Compliance", _comp_options, index=_comp_idx)

    # ── Reinforcement / contact preference → feedback style ───────
    _rf_options  = ["Verbal praise", "Visual charts", "Sticker rewards", "Written notes"]
    _contact_raw = _pr.get("contact_preference", "")
    if "App" in _contact_raw:
        _rf_idx = 2      # Sticker rewards / app reminders
    elif "Email" in _contact_raw:
        _rf_idx = 3      # Written notes
    elif "Weekly" in _contact_raw or "Report" in _contact_raw:
        _rf_idx = 1      # Visual charts
    else:
        _rf_idx = 0
    reinforcement_style = st.selectbox("Preferred Feedback Style", _rf_options, index=_rf_idx)

# -------------------------
# Chat history display
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# -------------------------
# Chat input
# -------------------------
user_question = st.chat_input("Ask me anything...")

if user_question:

    # show user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # -------------------------
    # 🔍 RAG PROCESS
    # -------------------------

    # ── RAG OPTIMISATION 1: Metadata Filtering ─────────────────────────────────────────────────────
    # Derive the patient's therapy focus from their diagnosis / primary concern.
    # For WHY / evidence questions, bypass the therapy_focus filter entirely so
    # research papers are always included in the retrieval candidate pool.
    therapy_focus_hint = get_therapy_focus(diagnosis, primary_concern)
    if is_evidence_question(user_question):
        therapy_focus_hint = ""   # no filter → full medbase searched
    base_retriever = build_base_retriever(therapy_focus_hint)

    # ── RAG OPTIMISATION 2: Multi-Query Retrieval ────────────────────────────────────────────────────
    # Wrap the base retriever with MultiQueryRetriever.
    # The LLM generates several rephrasings of the therapist's question, each is
    # run through the base retriever, and results are deduplicated.
    # This improves recall when natural therapist language doesn't match clinical PDF vocabulary.
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    # Retrieve relevant documents using the combined optimised pipeline
    docs = mq_retriever.invoke(user_question)

    # Format retrieved context for the prompt
    retrieved_documents = ""

    # ── RAG OPTIMISATION 4: Prepend real minimal pairs when requested ──────────────────
    # Deterministic CSV lookup — bypasses the LLM's unreliable pair generation.
    # Also detects follow-up requests ("more", "another") by checking whether the
    # previous assistant message already contained minimal pairs.
    prev_msgs_content = " ".join(
        m.content.lower()
        for m in st.session_state.messages[:-1]
        if isinstance(m, AIMessage)
    )
    prev_was_minimal_pairs = (
        "minimal pair" in prev_msgs_content
        or " / " in prev_msgs_content   # formatted pairs from a previous response
    )
    if is_minimal_pairs_question(user_question, prev_was_minimal_pairs):
        mp_context = get_minimal_pairs_context(error_patterns)
        if mp_context:
            retrieved_documents += mp_context + "\n\n"

    for doc in docs:
        title   = doc.metadata.get("title",   doc.metadata.get("source", "unknown"))
        source  = doc.metadata.get("source",  "CPFT Toolkit")
        link    = doc.metadata.get("link",    "")
        retrieved_documents += (
            f"[{title}] (Source: {source}{', Link: ' + link if link else ''})\n"
            f"{doc.page_content}\n\n"
        )

    # Format chat history as a readable string
    chat_history_text = ""
    for msg in st.session_state.messages[:-1]:   # exclude the current message
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        chat_history_text += f"{role}: {msg.content}\n"

    # Build the final prompt with all required variables
    final_prompt = prompt.format(
        age=age,
        diagnosis=diagnosis,
        primary_concern=primary_concern,
        therapy_stage=therapy_stage,
        target_sounds=target_sounds,
        error_patterns=error_patterns or "Not specified",
        accuracy_last_session=accuracy_last_session,
        parent_engagement=parent_engagement,
        parent_confidence=parent_confidence,
        home_practice_compliance=home_practice_compliance,
        reinforcement_style=reinforcement_style,
        retrieved_documents=retrieved_documents or "No relevant documents retrieved.",
        input=user_question,
        chat_history=chat_history_text or "No prior chat history.",
    )

    # with st.spinner("✍️ Generating coaching session..."):
    with st.spinner("🧠 Thinking..."):

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(final_prompt):
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

    # Save message to history
    st.session_state.messages.append(AIMessage(content=full_response))


    # # Generate answer
    # with st.spinner("🧠 Thinking..."):
        
    #     # result = llm.invoke(final_prompt)
    #     # answer = result.content

    #     result = llm.stream(final_prompt)


    #     with st.chat_message("assistant"):
    #         st.write_stream(result)

    #     print(result)


    # # print("\n\nuser query:", user_question)
    # # print("\n\nretrieved context:", retrieved_documents)
    # # print("\n\nfinal answer:", answer)




    # with st.chat_message("assistant"):
    #     placeholder = st.empty()
    #     full_response = ""

    #     for chunk in llm.stream(final_prompt):
    #         full_response += chunk.content
    #         placeholder.markdown(full_response + "▌")

    #     placeholder.markdown(full_response)



    # # Debug expander: shows which documents were actually retrieved.
    # # Useful for verifying that retrieval changes per query.
    # # with st.expander("📄 Retrieved Sources", expanded=False):
    # #     if docs:
    # #         for i, doc in enumerate(docs, 1):
    # #             title  = doc.metadata.get("title",   doc.metadata.get("source", "unknown"))
    # #             page   = doc.metadata.get("page",    "?")
    # #             focus  = doc.metadata.get("therapy_focus", "")
    # #             st.markdown(f"**{i}. {title}** — page {page} | focus: `{focus}`")
    # #             st.caption(doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""))
    # #     else:
    # #         st.warning("No documents were retrieved for this query.")

    # st.session_state.messages.append(AIMessage(result.content))
