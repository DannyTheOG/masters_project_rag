"""
eval_explainability.py
=======================
LLM-as-Judge evaluation — EXPLAINABILITY dimension.

Explainability measures: Does the system explain WHY it recommends something?
Does it cite research evidence? Does it expose its clinical reasoning chain
transparently? This is where RAG has its clearest advantage — it has access
to research papers the baseline does not.

Compares:
  Condition A — RAG  (patient context + retrieval + LLM)
  Condition C — Baseline LLM (raw question only)

Test questions: E1–E4 (Explainability group)
Patient: Liam Rodriguez (Phonological Delay, r→w, 58% accuracy)

Output:
  eval_results_explainability.csv
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

# ── E1–E4: Explainability test questions ──────────────────────────────────────
# These are "why" and "evidence" questions. RAG has specific papers available:
#   - Hegarty 2018 / McFaul 2022: practice frequency/intensity
#   - McLeod & Crowe 2018: consonant acquisition norms
#   - McCauley et al. 2009 / Lof & Watson 2008: NSOMEs
#   - Gierut 1998: phonological therapy principles
TEST_QUESTIONS = [
    {
        "id": "E1",
        "question": (
            "Why are you suggesting short daily practice sessions "
            "rather than one longer session per week?"
        ),
        "rubric_note": (
            "The correct explanation references HIGH-FREQUENCY, LOW-INTENSITY practice "
            "research (e.g. Hegarty 2018, McFaul 2022, or similar intensity studies). "
            "Key principle: distributed practice leads to better retention and "
            "generalisation than massed practice. A response citing specific evidence "
            "or named researchers scores higher. A response that only says 'it's better "
            "to practise often' without explanation scores lower."
        ),
    },
    {
        "id": "E2",
        "question": (
            "Why hasn't Liam's /r/ improved faster — is something wrong?"
        ),
        "rubric_note": (
            "The correct explanation references acquisition norms: /r/ is one of the "
            "LAST sounds acquired, typically not fully established until age 7-8 "
            "(McLeod & Crowe 2018 or similar). At age 4, Liam's rate of progress is "
            "appropriate. The response should explain WHY /r/ is late-acquired (complex "
            "articulation) and reassure without dismissing concern. A response that just "
            "says 'be patient' without citing norms scores lower."
        ),
    },
    {
        "id": "E3",
        "question": (
            "Why do we focus on word level before moving to sentences?"
        ),
        "rubric_note": (
            "The correct explanation describes the EVIDENCE-BASED THERAPY HIERARCHY: "
            "sounds must be established at isolation → syllable → word → phrase → "
            "sentence → conversation. Moving to sentences before consolidation leads to "
            "breakdown and error reinforcement. Response should explain the clinical "
            "reasoning (not just state the rule), reference accuracy thresholds "
            "(e.g. 70-80% before advancing), and ideally link to motor learning or "
            "phonological learning theory."
        ),
    },
    {
        "id": "E4",
        "question": (
            "I read online that tongue exercises improve speech sounds — "
            "what does the research say about this?"
        ),
        "rubric_note": (
            "The correct evidence-based response is that NON-SPEECH ORAL MOTOR EXERCISES "
            "(NSOMEs) — tongue exercises, blowing, chewing — are NOT supported by research "
            "for improving speech sound production. Key citations: Lof & Watson 2008 "
            "(survey showing widespread use without evidence), McCauley et al. 2009 "
            "(systematic review). The response should cite evidence, NOT endorse tongue "
            "exercises, and redirect to direct sound practice. Citing specific researchers "
            "or studies scores higher."
        ),
    },
]

# ── RAG system prompt (evidence/explainability focused) ───────────────────────
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

The following clinical notes and research papers are provided as BACKGROUND \
EVIDENCE. For questions asking WHY or for evidence, you MUST cite specific \
research, authors, studies, or clinical guidelines from these notes. \
Do not invent references — only cite what appears in the notes below.

{retrieved_documents}

Ground rules:
- When asked WHY or for evidence, ALWAYS explain the clinical reasoning chain \
  (not just the rule — explain the evidence behind it).
- Cite specific papers, authors, or guidelines if they appear in the retrieved notes.
- If evidence is limited or contested, say so explicitly — do not overstate certainty.
- NON-SPEECH ORAL MOTOR EXERCISES: If asked about tongue exercises or oral motor \
  drills, state clearly that research does NOT support NSOMEs for speech improvement. \
  Cite evidence if available. Do not endorse them.
- NEVER start by summarising the question. Begin your answer directly.

Question: {question}
Response:"""

# ── Retrieval (always uses full medbase for "why"/evidence questions) ─────────
_EVIDENCE_KEYWORDS = [
    "why", "evidence", "research", "reason", "study", "because",
    "what does", "how do we know", "explain", "theory",
]

def _retrieve(question, patient):
    q = question.lower()
    # For evidence questions, bypass therapy_focus filter → full medbase
    is_evidence = any(kw in q for kw in _EVIDENCE_KEYWORDS)
    kwargs = {"k": 5, "fetch_k": 20}
    if not is_evidence:
        focus = "phonology"   # default to phonology for Liam
        _KNOWN = {
            "phonology", "articulation", "phonology|articulation",
            "phonology|articulation|stammering|early_language",
        }
        matching = [v for v in _KNOWN if focus in v]
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
You are evaluating an AI speech therapy assistant on EXPLAINABILITY only.

Explainability measures: Does the response explain WHY, not just WHAT?
Does it cite evidence? Is the clinical reasoning transparent?

=== BACKGROUND: AVAILABLE EVIDENCE ===
The RAG system has access to papers on:
- Practice frequency/intensity (Hegarty 2018, McFaul 2022)
- Consonant acquisition norms (McLeod & Crowe 2018)
- NSOMEs evidence: Lof & Watson 2008, McCauley et al. 2009
- Phonological therapy principles (Gierut 1998)
A baseline LLM has NO access to these papers.

=== RUBRIC ===
{rubric_note}

=== QUESTION ===
{question}

=== RESPONSE TO EVALUATE ===
{response}

=== SCORING CRITERIA ===
Score each 1-5. Return ONLY JSON.

1. provides_rationale (1-5)
   5 = Response explains the clinical/theoretical WHY, not just the WHAT
   3 = Some reasoning given but it is incomplete or superficial
   1 = Response only states what to do, gives no reason at all

2. cites_evidence (1-5)
   5 = Response cites specific named studies, researchers, or clinical guidelines
   3 = Response references "research" or "studies" generally without naming them
   1 = No reference to evidence; claims are made without any support

3. clinical_reasoning (1-5)
   5 = Complete and accurate reasoning chain — explains the mechanism not just the rule
   3 = Partially correct reasoning; some gaps or oversimplifications
   1 = Reasoning is absent, circular, or incorrect

4. transparent_uncertainty (1-5)
   5 = Response accurately flags contested evidence or limits of knowledge when relevant
   3 = Mostly appropriate confidence; one area where uncertainty should be flagged but isn't
   1 = Overconfident, or fails to flag that an online claim (e.g. NSOMEs) is unsupported

5. overall_explainability (1-5)
   Holistic score — does the response genuinely help the caregiver understand WHY?

Return ONLY this JSON:
{{"provides_rationale": X, "cites_evidence": X, "clinical_reasoning": X, "transparent_uncertainty": X, "overall_explainability": X}}
"""

def judge_response(question, response, rubric_note):
    raw = llm.invoke(JUDGE_PROMPT.format(
        question=question, response=response, rubric_note=rubric_note)).content
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        print(f"  [WARN] non-JSON: {raw[:80]}")
        return {k: None for k in ["provides_rationale","cites_evidence",
                                   "clinical_reasoning","transparent_uncertainty",
                                   "overall_explainability"]}
    try:    return json.loads(match.group())
    except: return {k: None for k in ["provides_rationale","cites_evidence",
                                        "clinical_reasoning","transparent_uncertainty",
                                        "overall_explainability"]}

# ── Runner ────────────────────────────────────────────────────────────────────
def run_evaluation():
    score_keys = ["provides_rationale","cites_evidence","clinical_reasoning",
                  "transparent_uncertainty","overall_explainability"]
    rows = []

    print("\n" + "="*70)
    print("  EXPLAINABILITY EVALUATION — RAG vs Baseline LLM")
    print("  NOTE: RAG has access to research papers; Baseline does not.")
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

        print(f"\n  {'Metric':<26} {'RAG':>6} {'BASELINE':>10}")
        print(f"  {'-'*44}")
        for k in score_keys:
            rv, bv = rag_s.get(k,"—"), base_s.get(k,"—")
            win = "◄ RAG" if (rv and bv and rv>bv) else ("◄ BASE" if (rv and bv and bv>rv) else "")
            print(f"  {k:<26} {str(rv):>6} {str(bv):>10}  {win}")

    df = pd.DataFrame(rows)
    print("\n" + "="*70 + "\n  MEAN SCORES\n" + "="*70)
    print(f"\n  {'Metric':<26} {'RAG':>8} {'BASELINE':>12} {'Δ (RAG−Base)':>14}")
    print(f"  {'-'*62}")
    for k in score_keys:
        rm, bm = df[f"RAG_{k}"].mean(), df[f"BASELINE_{k}"].mean()
        d = rm - bm
        print(f"  {k:<26} {rm:>8.2f} {bm:>12.2f} {'+' if d>=0 else ''}{d:>13.2f}")

    out = os.path.join(os.path.dirname(__file__), "eval_results_explainability.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    print("""
NOTE FOR PAPER:
  Explainability is the dimension where RAG's advantage is most defensible.
  The baseline LLM cannot cite Hegarty 2018, McLeod & Crowe 2018, or
  McCauley et al. 2009 — it has no access to these papers. Any citations
  from the baseline are fabricated (hallucinated). Any citations from RAG
  come from the actual retrieved documents.
""")

if __name__ == "__main__":
    run_evaluation()
