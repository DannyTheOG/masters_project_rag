#####################  1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ######################
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader
import glob
from langchain_core.documents import Document


load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  ##################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

##################  DELETE CHROMA DB IF EXISTS AND INITIALIZE   ###################################

if os.path.exists(os.getenv("DATABASE_LOCATION")):
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

#########################  INITIALIZE TEXT SPLITTER   ##################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

######################## 2.  PROCESSING THE DOCS(PDFs) RESPONSE LINE BY LINE   ####################
#########################  FUNCTION TO EXTRACT RESPONSE LINE BY LINE   ###########################

# ---------------------------------------------------------------------------
# Per-document metadata lookup table.
# Keys are the bare PDF filenames (without the leading "medbase/" prefix).
# Add or edit entries here as new documents are added to the medbase/ folder.
# ---------------------------------------------------------------------------
DOCUMENT_METADATA = {
    # ── existing entries ──────────────────────────────────────────────────────
    "Intensive-interaction-updated-Nov-25.pdf": {
        "title": "Intensive Interaction",
        "section_heading": "Intensive Interaction Guidance",
        "therapy_focus": ["early_language"],
        "age_range": ["0-2", "2-5"],
        "target_skills": ["language_development", "AAC"],
        "link": "medbase/Intensive-interaction-updated-Nov-25.pdf",
    },
    "Language-Top-Tips.pdf": {
        "title": "Language Top Tips",
        "section_heading": "Language Development Top Tips",
        "therapy_focus": ["early_language"],
        "age_range": ["0-2", "2-5", "5-8"],
        "target_skills": ["language_development"],
        "link": "medbase/Language-Top-Tips.pdf",
    },
    "STRATEGIES TO HELP YOUR CHILD.pdf": {
        "title": "Strategies to Help Your Child",
        "section_heading": "Parent Strategies",
        "therapy_focus": ["articulation", "early_language"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound", "language_development"],
        "link": "medbase/STRATEGIES TO HELP YOUR CHILD.pdf",
    },
    "patient-parent-data.pdf": {
        "title": "Patient and Parent Data",
        "section_heading": "Patient and Parent Information",
        "therapy_focus": ["phonology", "articulation", "stammering", "early_language"],
        "age_range": ["0-2", "2-5", "5-8", "8-12"],
        "target_skills": ["speech_sound", "language_development", "fluency", "AAC"],
        "link": "medbase/patient-parent-data.pdf",
    },
    # ── newly added entries ───────────────────────────────────────────────────
    "Advice-to-Promote-Early-Language-Development.pdf": {
        "title": "Advice to Promote Early Language Development",
        "section_heading": "Early Language Development Advice",
        "therapy_focus": ["early_language"],
        "age_range": ["0-2", "2-5"],
        "target_skills": ["language_development"],
        "link": "medbase/Advice-to-Promote-Early-Language-Development.pdf",
    },
    "Compound-words.pdf": {
        "title": "Compound Words",
        "section_heading": "Compound Words Activity",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["5-8", "8-12"],
        "target_skills": ["speech_sound", "language_development"],
        "link": "medbase/Compound-words.pdf",
    },
    "Identifying-sounds-leaflet.pdf": {
        "title": "Identifying Sounds Leaflet",
        "section_heading": "Sound Identification",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Identifying-sounds-leaflet.pdf",
    },
    "Minimal-Pairs.pdf": {
        "title": "Minimal Pairs",
        "section_heading": "Minimal Pairs Therapy",
        "therapy_focus": ["phonology"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Minimal-Pairs.pdf",
    },
    "Speech-discrimination-games.pdf": {
        "title": "Speech Discrimination Games",
        "section_heading": "Speech Discrimination Activities",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Speech-discrimination-games.pdf",
    },
    "Strategies-to-support-your-child-with-speech-sound-difficulties.pdf": {
        "title": "Strategies to Support Your Child with Speech Sound Difficulties",
        "section_heading": "Speech Sound Support Strategies",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Strategies-to-support-your-child-with-speech-sound-difficulties.pdf",
    },
    "Syllable-clapping-programme.pdf": {
        "title": "Syllable Clapping Programme",
        "section_heading": "Syllable Awareness Activities",
        "therapy_focus": ["phonology"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Syllable-clapping-programme.pdf",
    },
    "Typical-speech-sound-development.pdf": {
        "title": "Typical Speech Sound Development",
        "section_heading": "Speech Sound Development Milestones",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["0-2", "2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Typical-speech-sound-development.pdf",
    },
    "Where-in-the-word.pdf": {
        "title": "Where in the Word",
        "section_heading": "Word Position Activities",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/Where-in-the-word.pdf",
    },
    "english_words_1000.pdf": {
        "title": "1000 Most Common English Words",
        "section_heading": "English Vocabulary Reference",
        "therapy_focus": ["early_language"],
        "age_range": ["2-5", "5-8", "8-12"],
        "target_skills": ["language_development"],
        "link": "medbase/english_words_1000.pdf",
    },
    "phoneme_minimal_pair.pdf": {
        "title": "Phoneme Minimal Pairs",
        "section_heading": "Phoneme Minimal Pair Therapy",
        "therapy_focus": ["phonology"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/phoneme_minimal_pair.pdf",
    },
    "phonological_words_5000.pdf": {
        "title": "5000 Phonological Words",
        "section_heading": "Phonological Word Reference",
        "therapy_focus": ["phonology"],
        "age_range": ["5-8", "8-12"],
        "target_skills": ["speech_sound", "language_development"],
        "link": "medbase/phonological_words_5000.pdf",
    },
    "transition-changes.pdf": {
        "title": "Transition and Changes",
        "section_heading": "Managing Transitions and Changes",
        "therapy_focus": ["early_language", "stammering"],
        "age_range": ["5-8", "8-12"],
        "target_skills": ["language_development", "fluency"],
        "link": "medbase/transition-changes.pdf",
    },
    # ── research papers added 2026-04-05 ─────────────────────────────────────
    "mcleod-crowe-2018-children-s-consonant-acquisition-in-27-languages-a-cross-linguistic-review.pdf": {
        "title": "Children's Consonant Acquisition in 27 Languages: A Cross-Linguistic Review",
        "section_heading": "Speech Sound Development Norms — Cross-Linguistic",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["0-2", "2-5", "5-8"],
        "target_skills": ["speech_sound"],
        "link": "medbase/mcleod-crowe-2018-children-s-consonant-acquisition-in-27-languages-a-cross-linguistic-review.pdf",
    },
    "Sugdenetal_2019_JEI.pdf": {
        "title": "Parents' Experiences of Completing Home Practice for Speech Sound Disorders",
        "section_heading": "Parent Home Practice — Qualitative Study",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound", "parent_support"],
        "link": "medbase/Sugdenetal_2019_JEI.pdf",
    },
    "sugden-et-al-2020-evaluation-of-parent-and-speech-language-pathologist-delivered-multiple-oppositions-intervention-for.pdf": {
        "title": "Evaluation of Parent- and SLP-Delivered Multiple Oppositions Intervention for Children with Phonological Impairment",
        "section_heading": "Parent-Delivered Intervention — Multiple Oppositions",
        "therapy_focus": ["phonology"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound", "parent_support"],
        "link": "medbase/sugden-et-al-2020-evaluation-of-parent-and-speech-language-pathologist-delivered-multiple-oppositions-intervention-for.pdf",
    },
    "Intl J Lang   Comm Disor - 2024 - Pritchard - How speech and language therapists and parents work together in the.pdf": {
        "title": "How Speech and Language Therapists and Parents Work Together in the Therapeutic Process for Children with SSD: A Scoping Review",
        "section_heading": "SLT-Parent Collaboration — Scoping Review",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["0-2", "2-5"],
        "target_skills": ["speech_sound", "parent_support"],
        "link": "medbase/Intl J Lang   Comm Disor - 2024 - Pritchard - How speech and language therapists and parents work together in the.pdf",
    },
    "EBSCO-FullText-04_05_2026.pdf": {
        "title": "A Qualitative Exploration of SLPs' Intervention and Intensity Provision for Children with Phonological Impairment",
        "section_heading": "Treatment Intensity — Qualitative Study (Hegarty et al. 2021)",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["0-2", "2-5", "5-8", "8-12"],
        "target_skills": ["speech_sound"],
        "link": "medbase/EBSCO-FullText-04_05_2026.pdf",
    },
    "e081446.full.pdf": {
        "title": "Outcome Measures for Children with Speech Sound Disorder: An Umbrella Review (Harding et al. 2024)",
        "section_heading": "Assessment & Outcome Measures — Umbrella Review",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["0-2", "2-5", "5-8", "8-12"],
        "target_skills": ["speech_sound"],
        "link": "medbase/e081446.full.pdf",
    },
    "e001761.full.pdf": {
        "title": "Applying Evidence to Practice by Increasing Intensity for Children with Severe SSD: A QI Project (McFaul et al. 2022)",
        "section_heading": "Treatment Intensity — Quality Improvement Project",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8", "8-12"],
        "target_skills": ["speech_sound"],
        "link": "medbase/e001761.full.pdf",
    },
    "e074272.full.pdf": {
        "title": "Supporting Parents to Implement Intensive SLT at Home for Children with SSD: A Realist Review Protocol (Leafe et al. 2024)",
        "section_heading": "Home Practice Intensity — Realist Review Protocol",
        "therapy_focus": ["phonology", "articulation"],
        "age_range": ["2-5", "5-8"],
        "target_skills": ["speech_sound", "parent_support"],
        "link": "medbase/e074272.full.pdf",
    },
    "EBSCO-FullText-02_22_2026.cleaned.pdf": {
        "title": "Clinical Application of Large Language Models for Intervention Plan Development in Speech-Language Pathology (Kim et al. 2025)",
        "section_heading": "AI and LLMs in SLT — Clinical Application",
        "therapy_focus": ["phonology", "articulation", "early_language", "stammering"],
        "age_range": ["0-2", "2-5", "5-8", "8-12"],
        "target_skills": ["speech_sound", "language_development"],
        "link": "medbase/EBSCO-FullText-02_22_2026.cleaned.pdf",
    },
}

# Fallback metadata used when a PDF is not in the lookup table above.
DEFAULT_METADATA = {
    "title": "",
    "section_heading": "",
    "therapy_focus": [],
    "age_range": [],
    "target_skills": [],
    "link": "",
}


def get_document_metadata(pdf_path: str) -> dict:
    """Return the metadata dict for a given PDF path, falling back to defaults."""
    filename = os.path.basename(pdf_path)
    return DOCUMENT_METADATA.get(filename, {**DEFAULT_METADATA, "title": filename, "link": pdf_path})


def process_pdf_files(file_path):
    """Process each pdf line and extract relevant information."""
    extracted = []

    for pdf in glob.glob("medbase/*.pdf", recursive=True):
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        # Attach the source PDF path so we can look it up later.
        for doc in docs:
            doc.metadata["pdf_path"] = pdf
        extracted.extend(docs)

    return extracted


file_content = process_pdf_files(os.getenv("DATASET_STORAGE_FOLDER") + "data.txt")


#########################  3.  CHUNKING, EMBEDDING AND INGESTION   ##################################

for line in file_content:

    if len(line.page_content) < 10:
        continue

    pdf_path = line.metadata.get("pdf_path", line.metadata.get("source", ""))
    doc_meta = get_document_metadata(pdf_path)

    # Build the rich metadata dict for this page's chunks.
    # Fields that are lists (therapy_focus, age_range, target_skills) are
    # stored as pipe-delimited strings because ChromaDB only supports scalar
    # metadata values.
    base_metadata = {
        "title": doc_meta["title"],
        "section_heading": doc_meta["section_heading"],
        "therapy_focus": "|".join(doc_meta["therapy_focus"]),
        "age_range": "|".join(doc_meta["age_range"]),
        "target_skills": "|".join(doc_meta["target_skills"]),
        "source": "CPFT Toolkit",
        "link": doc_meta["link"],
        # Page number (0-indexed) from PyPDFLoader – useful for section tracing.
        "page": line.metadata.get("page", 0),
    }

    texts = text_splitter.create_documents(
        texts=[line.page_content],
        metadatas=[base_metadata],
    )

    # Assign a unique UUID to every chunk and stamp it into the metadata too.
    uuids = [str(uuid4()) for _ in range(len(texts))]
    for chunk, uid in zip(texts, uuids):
        chunk.metadata["id"] = uid

    vector_store.add_documents(documents=texts, ids=uuids)

    print(pdf_path, "| page", base_metadata["page"], "| chunks", len(texts))
