"""
eval_usability.py
==================
LLM-as-Judge evaluation — USABILITY dimension.

Usability measures: Is the response clear, practical, and accessible to a
non-clinician caregiver? Can a parent actually act on this advice without
clinical training?

Compares:
  Condition A — RAG  (patient context + retrieval + LLM)
  Condition C — Baseline LLM (raw question only)

Test questions: U1–U4 (Usability group, caregiver perspective)
Patient: Liam Rodriguez (Phonological Delay, r→w, 58% accuracy)

Output:
  eval_results_usability.csv
"""

import os, json, re
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

embeddings   = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
)

# ── Patient context ───────────────────────────────────────────────────────────
LIAM = {
    "age": "4 years",
    "diagnosis": "Phonological Delay",
    "primary_concern": "Liquid Simplification",
    "therapy_stage": "Structured Word Level",
    "target_sounds": "r",
    "error_patterns": "r→w substitution",
    "accuracy_last_session": 58,
    "parent_engagement": "High",
    "parent_confidence": "Moderate",
    "home_practice_compliance": "Consistent (5–6 days/week)",
    "reinforcement_style": "Sticker rewards",
}

# ── U1–U4: Usability test questions ───────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "id": "U1",
        "question": (
            "What should I say to Liam when he gets the /r/ "
            "sound wrong during practice?"
        ),
        "rubric_note": (
            "Response must provide EXACT words the parent can say — not abstract "
            "concepts like 'give corrective feedback'. Should use warm, non-technical "
            "language (no jargon like 'phoneme', 'articulation', 'minimal pair'). "
            "Should recommend modelling the correct form, not direct correction."
        ),
    },
    {
        "id": "U2",
        "question": (
            "Can you give me a simple 5-minute routine I can do "
            "with Liam every evening?"
        ),
        "rubric_note": (
            "Response must be structured as a numbered step-by-step routine that a "
            "parent with no SLT background can follow immediately. Steps should be "
            "concrete and time-bounded. Must not exceed 5 minutes in total. "
            "Must not use unexplained clinical terminology."
        ),
    },
    {
        "id": "U3",
        "question": (
            "Liam refuses to practise — he says it's boring. What can I do?"
        ),
        "rubric_note": (
            "Response must give specific, named engagement strategies (not vague 'try "
            "games'). Should reference Liam's existing reinforcement style (sticker "
            "rewards) and suggest how to build on it. Tone must be warm and empathetic "
            "— parent is frustrated. Should not lecture about the importance of practice."
        ),
    },
    {
        "id": "U4",
        "question": (
            "I'm not sure if I'm doing the home practice correctly. "
            "How will I know if it's working?"
        ),
        "rubric_note": (
            "Response must give objective, observable success indicators a non-clinician "
            "can check (e.g. 'Liam says the /r/ word correctly more than half the time "
            "without you modelling first'). Must not use ambiguous clinical terms like "
            "'accuracy' without explaining what to count. Should feel encouraging, not "
            "like a test the parent might fail."
        ),
    },
]

# ── RAG system prompt ─────────────────────────────────────────────────────────
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

The following clinical notes are BACKGROUND EVIDENCE ONLY. Do NOT copy them \
as activity templates. Use them to inform your clinical reasoning, then write \
your own tailored response from scratch.

{retrieved_documents}

Ground rules:
- Use plain, accessible language — the caregiver is not a clinician.
- Avoid unexplained jargon (phoneme, articulation, minimal pair, etc.).
- Give concrete, specific advice the parent can act on immediately.
- Warm, encouraging tone throughout.
- ACCURACY RULES — apply strictly: \
  Below 60% → parent models EVERY production, child imitates only. \
  NO independent practice.

If the request asks for a routine or activity, use numbered steps.
If conversational — respond naturally and concisely.

Question: {question}
Response:"""

# ── Retrieval ─────────────────────────────────────────────────────────────────
_KNOWN_FOCUS_VALUES = {
    "phonology", "articulation", "stammering", "early_language",
    "phonology|articulation", "articulation|early_language",
    "early_language|stammering",
    "phonology|articulation|stammering|early_language",
    "phonology|articulation|early_language|stammering",
}

def _get_focus(diagnosis, concern):
    combined = (diagnosis + " " + concern).lower()
    if any(k in combined for k in ["phonol", "cluster", "liquid"]): return "phonology"
    if any(k in combined for k in ["articul", "lisp"]):              return "articulation"
    if any(k in combined for k in ["stammer", "fluency"]):           return "stammering"
    return ""

def _retrieve(question, patient):
    focus = _get_focus(patient["diagnosis"], patient["primary_concern"])
    kwargs = {"k": 4, "fetch_k": 15}
    if focus:
        matching = [v for v in _KNOWN_FOCUS_VALUES if focus in v]
        if matching:
            kwargs["filter"] = {"therapy_focus": {"$in": matching}}
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=kwargs)
    mq = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    docs = mq.invoke(question)
    if not docs:
        return "No relevant documents retrieved."
    return "\n\n".join(
        f"[{d.metadata.get('title', 'unknown')}]\n{d.page_content}" for d in docs
    )

# ── Response generators ───────────────────────────────────────────────────────
def generate_rag_response(question, patient):
    retrieved = _retrieve(question, patient)
    prompt = RAG_SYSTEM_PROMPT.format(**patient, retrieved_documents=retrieved,
                                      question=question)
    return llm.invoke(prompt).content

def generate_baseline_response(question):
    return llm.invoke(question).content

# ── Judge ─────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are evaluating an AI speech therapy assistant on USABILITY only.

Usability measures: Is the response clear, jargon-free, and actionable for a
non-clinician caregiver (parent) with no SLT training?

=== CAREGIVER CONTEXT ===
Parent confidence: Moderate. Home practice compliance: High (5-6 days/week).
Child: Liam, 4 years, r→w substitution, 58% accuracy last session.

=== USABILITY RUBRIC ===
{rubric_note}

=== QUESTION ===
{question}

=== RESPONSE TO EVALUATE ===
{response}

=== SCORING CRITERIA ===
Score each item 1-5. Return ONLY a JSON object.

1. clarity (1-5)
   5 = All advice is in plain English; no unexplained clinical jargon
   3 = Mostly clear, but one or two unexplained terms used
   1 = Heavy clinical jargon; a non-clinician caregiver would be confused

2. actionability (1-5)
   5 = Response gives specific, concrete steps the parent can do RIGHT NOW
   3 = Response gives general strategies but lacks specific how-to detail
   1 = Response is abstract or conceptual — parent cannot act on it directly

3. appropriate_length (1-5)
   5 = Response is appropriately concise — no padding, all content is useful
   3 = Slightly too long or too short for the question asked
   1 = Excessive padding OR so brief it is unhelpful

4. caregiver_tone (1-5)
   5 = Warm, encouraging, peer-to-peer tone; parent feels supported not lectured
   3 = Mostly appropriate but slightly clinical or impersonal in places
   1 = Cold, clinical, or condescending tone that would discourage a parent

5. overall_usability (1-5)
   Holistic score — would a non-clinician caregiver find this response genuinely helpful?

Return ONLY this JSON:
{{"clarity": X, "actionability": X, "appropriate_length": X, "caregiver_tone": X, "overall_usability": X}}
"""

def judge_response(question, response, rubric_note):
    raw = llm.invoke(JUDGE_PROMPT.format(
        question=question, response=response, rubric_note=rubric_note)).content
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        print(f"  [WARN] non-JSON: {raw[:80]}")
        return {k: None for k in ["clarity","actionability","appropriate_length",
                                   "caregiver_tone","overall_usability"]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {k: None for k in ["clarity","actionability","appropriate_length",
                                   "caregiver_tone","overall_usability"]}

# ── Runner ────────────────────────────────────────────────────────────────────
def run_evaluation():
    score_keys = ["clarity","actionability","appropriate_length",
                  "caregiver_tone","overall_usability"]
    rows = []

    print("\n" + "="*70)
    print("  USABILITY EVALUATION — RAG vs Baseline LLM")
    print("="*70)

    for q in TEST_QUESTIONS:
        print(f"\n[{q['id']}] {q['question'][:65]}...")
        print("  RAG response ...",      end=" ", flush=True)
        rag_ans  = generate_rag_response(q["question"], LIAM);  print("done")
        print("  Baseline response ...", end=" ", flush=True)
        base_ans = generate_baseline_response(q["question"]);   print("done")
        print("  Judging RAG ...",       end=" ", flush=True)
        rag_s    = judge_response(q["question"], rag_ans,  q["rubric_note"]); print("done")
        print("  Judging Baseline ...",  end=" ", flush=True)
        base_s   = judge_response(q["question"], base_ans, q["rubric_note"]); print("done")

        row = {"question_id": q["id"], "question": q["question"],
               "rag_full_response": rag_ans, "baseline_full_response": base_ans}
        for k in score_keys:
            row[f"RAG_{k}"]      = rag_s.get(k)
            row[f"BASELINE_{k}"] = base_s.get(k)
        rows.append(row)

        print(f"\n  {'Metric':<22} {'RAG':>6} {'BASELINE':>10}")
        print(f"  {'-'*40}")
        for k in score_keys:
            rv, bv = rag_s.get(k,"—"), base_s.get(k,"—")
            win = "◄ RAG" if (rv and bv and rv>bv) else ("◄ BASE" if (rv and bv and bv>rv) else "")
            print(f"  {k:<22} {str(rv):>6} {str(bv):>10}  {win}")

    df = pd.DataFrame(rows)
    print("\n" + "="*70 + "\n  MEAN SCORES\n" + "="*70)
    print(f"\n  {'Metric':<22} {'RAG':>8} {'BASELINE':>12} {'Δ':>8}")
    print(f"  {'-'*54}")
    for k in score_keys:
        rm, bm = df[f"RAG_{k}"].mean(), df[f"BASELINE_{k}"].mean()
        d = rm - bm
        print(f"  {k:<22} {rm:>8.2f} {bm:>12.2f} {'+' if d>=0 else ''}{d:>7.2f}")

    out = os.path.join(os.path.dirname(__file__), "eval_results_usability.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

if __name__ == "__main__":
    run_evaluation()
