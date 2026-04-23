"""
eval_contextual_accuracy.py
============================
LLM-as-Judge evaluation of the CONTEXTUAL ACCURACY dimension.

Compares:
  Condition A  —  RAG system  (patient context + ChromaDB retrieval + LLM)
  Condition C  —  Baseline LLM (raw question only, no context, no retrieval)

Test questions: Q1–Q4 from the evaluation framework (Contextual Accuracy group).
Patient: Liam Rodriguez (Phonological Delay, r→w substitution, 58% accuracy).

Output:
  - Console table of per-question scores
  - eval_results_contextual_accuracy.csv  (for research paper)

Usage:
  source venv/bin/activate
  python eval_contextual_accuracy.py
"""

import os, json, re, textwrap
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# ── Model + vector store setup ────────────────────────────────────────────────

embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,          # deterministic for evaluation reproducibility
)


# ── Patient context (Liam — used for Q1–Q4) ──────────────────────────────────

LIAM = {
    "age":                    "4 years",
    "diagnosis":              "Phonological Delay",
    "primary_concern":        "Liquid Simplification",
    "therapy_stage":          "Structured Word Level",
    "target_sounds":          "r",
    "error_patterns":         "r→w substitution, cluster reduction",
    "accuracy_last_session":  58,
    "parent_engagement":      "High",
    "parent_confidence":      "Moderate",
    "home_practice_compliance": "Consistent (5–6 days/week)",
    "reinforcement_style":    "Sticker rewards",
}


# ── Q1–Q4: Contextual Accuracy test questions ─────────────────────────────────
# Each includes a `rubric_note` — the key clinical fact the judge will verify.

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "Given how often the family practises at home, "
            "should we increase the frequency or maintain it?"
        ),
        "rubric_note": (
            "Liam's family practises consistently 5-6 days/week — this is already "
            "high compliance. The correct answer is to MAINTAIN frequency and focus "
            "on quality/technique rather than adding more sessions. "
            "A response that recommends increasing frequency is incorrect. "
            "The response should reference the 5-6 days/week compliance data."
        ),
    },
    {
        "id": "Q2",
        "question": (
            "What reward or motivation strategy should we use "
            "to keep Liam engaged during home practice?"
        ),
        "rubric_note": (
            "Liam already uses sticker rewards (reinforcement_style = 'Sticker rewards'). "
            "A correct response must build on this existing strategy — e.g. a sticker chart, "
            "themed stickers, earning stickers for correct productions. "
            "A generic reward suggestion (e.g. 'try praise' or 'try token boards') "
            "without referencing stickers is not patient-specific and should score lower."
        ),
    },
    {
        "id": "Q3",
        "question": (
            "Liam sometimes says 'wabbit' instead of 'rabbit' and 'wed' for 'red'. "
            "What does this tell us about his error pattern, and how should "
            "home practice address it?"
        ),
        "rubric_note": (
            "This is Liam's documented r\u2192w substitution pattern. The correct response "
            "should identify the substitution explicitly, explain that home practice "
            "should use minimal pair contrasts (red/wed, ring/wing) to help Liam "
            "discriminate between /r/ and /w/ at word level. "
            "A response that only lists /r/ words without addressing the w-substitution "
            "is incomplete."
        ),
    },
    {
        "id": "Q4",
        "question": (
            "Liam is at 58% accuracy on /r/ — "
            "should the parent model the sound for him, or should he "
            "practise independently without much support?"
        ),
        "rubric_note": (
            "At 58% accuracy (below 60%), the clinical rule is heavy therapist/parent "
            "modelling with scaffolded cues. Independent practice without support is "
            "appropriate only above ~80% accuracy. The correct answer is: parent should "
            "model the sound, provide cues, and not expect independent practice. "
            "Any response recommending independent practice at this stage is incorrect."
        ),
    },
]


# ── RAG system prompt (mirrors ct-latest.py, Streamlit-free) ─────────────────

RAG_SYSTEM_PROMPT = """\
You are a helpful, experienced Speech and Language Therapy (SLT) colleague. \
You support therapists with practical, evidence-informed advice. \
Speak naturally — like a knowledgeable peer, not a textbook.

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

The following clinical notes are provided as BACKGROUND EVIDENCE ONLY.
Do NOT copy them as activity templates or follow their structure.
Use them to inform your clinical reasoning, then write your own
tailored response from scratch based on the child's specific profile.

{retrieved_documents}

Ground rules:
- Match the response to what is actually being asked.
- Adjust difficulty to accuracy: below 60% use HEAVY MODELLING and cueing \
strategies (no generalisation); 60-80% use structured repetition with light \
prompting; above 80% focus on generalisation tasks.
- Do not introduce new therapy techniques or change existing goals.
- Do not give medical diagnoses or advice outside SLT scope.
- If you don't have enough evidence to answer safely, say so.
- NEVER start by describing or summarising the request. Begin your answer directly.
- You are the SLT — do not tell the user to 'consult an SLP'. You ARE the expert.
- When answering developmental questions ('is it normal?', 'will they grow out of \
it?'), use the developmental norms from the retrieved evidence if present, and \
anchor your answer to the child's specific age and diagnosis.

If the request asks for a session plan, activity, or practice, use this format:

  **Session Focus:** (one sentence goal, referencing the child's accuracy level)
  **Activity:** (step-by-step, tailored to the child's accuracy and therapy stage)
  **Practice Examples:** (at least 5 words appropriate for the child's target sound)
  **Parent Coaching Script:** (exact words the parent says, modelling-heavy if <60%)
  **Encouragement:** (short warm message for child and family)

If the request is conversational → respond naturally and concisely.

Question: {question}
Response:"""


# ── Retrieval helpers (from ct-latest.py) ─────────────────────────────────────

_KNOWN_FOCUS_VALUES = {
    "phonology", "articulation", "stammering", "early_language",
    "phonology|articulation", "articulation|early_language",
    "early_language|stammering",
    "phonology|articulation|stammering|early_language",
    "phonology|articulation|early_language|stammering",
}

_EVIDENCE_KEYWORDS = [
    "why", "reason", "because", "evidence", "research", "study",
    "normal", "typical", "is it normal", "when do children", "age",
    "grow out", "same as", "will he", "will she",
]


def _get_therapy_focus(diagnosis: str, primary_concern: str) -> str:
    combined = (diagnosis + " " + primary_concern).lower()
    if any(k in combined for k in ["phonol", "minimal pair", "cluster", "liquid"]):
        return "phonology"
    if any(k in combined for k in ["articul", "lisp", "speech sound"]):
        return "articulation"
    if any(k in combined for k in ["stammer", "stutter", "fluency"]):
        return "stammering"
    if any(k in combined for k in ["language", "vocab", "early"]):
        return "early_language"
    return ""


def _build_retriever(therapy_focus_hint: str, question: str):
    search_kwargs: dict = {"k": 4, "fetch_k": 15}  # reduced to limit retrieval noise
    is_evidence = any(kw in question.lower() for kw in _EVIDENCE_KEYWORDS)
    if therapy_focus_hint and not is_evidence:
        matching = [v for v in _KNOWN_FOCUS_VALUES if therapy_focus_hint in v]
        if matching:
            search_kwargs["filter"] = {"therapy_focus": {"$in": matching}}
    return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def _retrieve_docs(question: str, patient: dict) -> str:
    focus = _get_therapy_focus(patient["diagnosis"], patient["primary_concern"])
    retriever = _build_retriever(focus, question)
    mq = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    docs = mq.invoke(question)
    if not docs:
        return "No relevant documents retrieved."
    parts = []
    for doc in docs:
        title = doc.metadata.get("title", doc.metadata.get("source", "unknown"))
        parts.append(f"[{title}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Response generators ────────────────────────────────────────────────────────

def generate_rag_response(question: str, patient: dict) -> str:
    """Condition A: full RAG — patient context + retrieval + LLM."""
    retrieved = _retrieve_docs(question, patient)
    prompt = RAG_SYSTEM_PROMPT.format(
        **patient,
        retrieved_documents=retrieved,
        question=question,
    )
    return llm.invoke(prompt).content


def generate_baseline_response(question: str) -> str:
    """Condition C: standalone LLM — raw question only, no context, no retrieval."""
    return llm.invoke(question).content


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator for clinical speech-language therapy AI systems.
Score the response below on CONTEXTUAL ACCURACY only.

Contextual Accuracy measures: does the response use the specific clinical facts
about this child — their exact accuracy, stage, error pattern, and family profile?

=== PATIENT PROFILE (Lena — these are the ground-truth facts) ===
{patient_profiles}

=== CLINICAL RUBRIC (what a correct response MUST do) ===
{rubric_note}

=== QUESTION ASKED ===
{question}

=== RESPONSE TO EVALUATE ===
{response}

=== SCORING CRITERIA ===
Score each item 1-5 using the specific thresholds below.
Return ONLY a JSON object — no prose.

1. stage_accuracy (1-5)
   5 = Response explicitly targets Structured Word Level activities only
   3 = Response gives word-level activities but doesn't name the stage
   1 = Response recommends phrase, sentence, or spontaneous speech tasks

2. accuracy_rule (1-5)
   5 = Response explicitly references 58% accuracy AND applies the correct rule
       (heavy modelling below 60%, NOT generalisation)
   3 = Response gives accuracy-appropriate advice but does not cite the 58% figure
   1 = Response ignores accuracy level OR recommends generalisation at 58%

3. patient_specificity (1-5)
   5 = Response references at least 2 of these exact facts:
       [58% accuracy, r\u2192w substitution, sticker rewards, 5-6 days compliance,
        Structured Word Level, 4 years old, Liquid Simplification]
   3 = Response is tailored to /r/ difficulties but uses no specific patient facts
   1 = Response is fully generic — could apply to any child with any speech issue

4. clinical_safety (1-5)
   5 = All advice is safe, within SLT scope, and evidence-consistent
   3 = Mostly safe with minor concerns
   1 = Advice is harmful, outside SLT scope, or contradicts best practice

5. overall_car (1-5)
   Holistic score combining all above criteria.

Return ONLY this JSON (no text before or after):
{{"stage_accuracy": X, "accuracy_rule": X, "patient_specificity": X, "clinical_safety": X, "overall_car": X}}
"""


def judge_response(question: str, response: str, rubric_note: str) -> dict:
    """Send a response to the judge LLM and parse its JSON scores."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        rubric_note=rubric_note,
    )
    raw = llm.invoke(prompt).content

    # Extract the JSON object robustly (ignore any surrounding text)
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        print(f"  [WARN] Judge returned non-JSON: {raw[:120]}")
        return {k: None for k in
                ["stage_accuracy", "accuracy_rule", "patient_specificity",
                 "clinical_safety", "overall_car"]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e} — raw: {match.group()}")
        return {k: None for k in
                ["stage_accuracy", "accuracy_rule", "patient_specificity",
                 "clinical_safety", "overall_car"]}


# ── Run evaluation ─────────────────────────────────────────────────────────────

def run_evaluation():
    rows = []
    score_keys = ["stage_accuracy", "accuracy_rule", "patient_specificity",
                  "clinical_safety", "overall_car"]

    print("\n" + "="*72)
    print("  CONTEXTUAL ACCURACY EVALUATION — RAG vs Baseline LLM")
    print("="*72)

    for q in TEST_QUESTIONS:
        qid      = q["id"]
        question = q["question"]
        note     = q["rubric_note"]

        print(f"\n[{qid}] {question[:70]}...")

        # Generate responses
        print("  Generating RAG response ...", end=" ", flush=True)
        rag_response = generate_rag_response(question, LIAM)
        print("done")

        print("  Generating Baseline response ...", end=" ", flush=True)
        baseline_response = generate_baseline_response(question)
        print("done")

        # Judge both
        print("  Judging RAG response ...", end=" ", flush=True)
        rag_scores = judge_response(question, rag_response, note)
        print("done")

        print("  Judging Baseline response ...", end=" ", flush=True)
        baseline_scores = judge_response(question, baseline_response, note)
        print("done")

        # Store results
        row = {"question_id": qid, "question": question}
        for k in score_keys:
            row[f"RAG_{k}"]      = rag_scores.get(k)
            row[f"BASELINE_{k}"] = baseline_scores.get(k)
        row["rag_full_response"]      = rag_response
        row["baseline_full_response"] = baseline_response
        rows.append(row)

        # Print per-question summary
        print(f"\n  {'Metric':<22} {'RAG':>6} {'BASELINE':>10}")
        print(f"  {'-'*40}")
        for k in score_keys:
            rv = rag_scores.get(k, "—")
            bv = baseline_scores.get(k, "—")
            winner = "◄ RAG" if (rv and bv and rv > bv) else ("◄ BASE" if (rv and bv and bv > rv) else "")
            print(f"  {k:<22} {str(rv):>6} {str(bv):>10}  {winner}")

    # ── Aggregate summary ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print("\n" + "="*72)
    print("  MEAN SCORES ACROSS ALL QUESTIONS")
    print("="*72)
    print(f"\n  {'Metric':<22} {'RAG mean':>10} {'BASELINE mean':>14} {'Δ (RAG−Base)':>14}")
    print(f"  {'-'*62}")
    for k in score_keys:
        rag_mean  = df[f"RAG_{k}"].mean()
        base_mean = df[f"BASELINE_{k}"].mean()
        delta     = rag_mean - base_mean
        sign      = "+" if delta >= 0 else ""
        print(f"  {k:<22} {rag_mean:>10.2f} {base_mean:>14.2f} {sign+f'{delta:.2f}':>14}")

    # ── Save to CSV ──────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "eval_results_contextual_accuracy.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # ── Interpretation note ──────────────────────────────────────────────────
    print("""
METHODOLOGICAL NOTE FOR YOUR PAPER:
  Judge model  : {model} (same as system model — acknowledged limitation)
  Temperature  : 0 (deterministic output for reproducibility)
  Condition A  : RAG system (patient context + ChromaDB retrieval + LLM)
  Condition C  : Baseline LLM (raw question, no context, no retrieval)
  Scoring scale: 1 (very poor) to 5 (excellent) per rubric item
  Rubric items : Stage Accuracy, Accuracy-Rule Adherence, Patient Specificity,
                 Clinical Safety, Overall Contextual Accuracy (CAR)
""".format(model=os.getenv("CHAT_MODEL")))


if __name__ == "__main__":
    run_evaluation()
