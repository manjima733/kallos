from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download required NLTK data silently
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# ---------------- Data ----------------
actions = ["create", "view", "search", "delete", "update", "modify", "approve"]
processes = [
    "customer", "salesorder", "shipment", "invoice", "receipt",
    "credit memo", "debit memo", "purchase order", "payment", "job offer",
    "objective", "key result", "key result checkin", "initiative",
    "copy okr", "view okr"
]

action_synonyms = {
    "create": ["create", "add", "make", "generate", "build"],
    "view": ["view", "show", "display", "see", "list", "open"],
    "search": ["search", "find", "lookup", "locate", "filter", "retrieve", "get", "fetch"],
    "delete": ["delete", "remove", "discard", "erase", "drop", "deleted", "eliminate"],
    "update": ["update", "refresh", "revise", "amend", "change"],
    "modify": ["modify", "edit", "adjust", "alter"],
    "approve": ["approve", "authorize", "accept", "validate"],
}

process_synonyms = {
    "customer": ["customer", "client", "buyer"],
    "salesorder": ["salesorder", "sales order", "sales orders"],
    "shipment": ["shipment", "delivery", "dispatch", "shipments"],
    "invoice": ["invoice", "bill", "billing", "invoices"],
    "receipt": ["receipt", "proof of payment", "receipts"],
    "credit memo": ["credit memo", "credit note", "credit memos"],
    "debit memo": ["debit memo", "debit note"],
    "purchase order": ["purchase order", "purchaseorder", "purchase-orders", "po", "buy order", "purchase orders"],
    "payment": ["payment", "transaction", "remittance", "payments"],
    "job offer": ["job offer", "offer letter", "employment offer"],
    "objective": ["objective", "goal", "target"],
    "key result": ["key result", "kr", "results"],
    "key result checkin": ["key result checkin", "kr checkin", "check-in", "check in"],
    "initiative": ["initiative", "project", "task"],
    "copy okr": ["copy okr", "duplicate okr", "clone okr"],
    "view okr": ["view okr", "see okr", "display okr", "show okr"]
}

# Generate all possible intents
intents = [f"{action} {process}" for action in actions for process in processes]

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute intent embeddings
intent_embeddings = model.encode(intents, convert_to_tensor=False)

# Similarity acceptance threshold
EMBEDDING_ACCEPT_THRESHOLD = 0.9

# ---------------- Utility Functions ----------------
def normalize_word(word):
    return lemmatizer.lemmatize(word.lower(), pos='v')

def match_with_synonyms(sentence, synonyms_dict):
    sent_lemmas = [normalize_word(w) for w in re.findall(r'\w+', sentence.lower())]
    sent_text = " ".join(sent_lemmas)
    for key, synonyms in synonyms_dict.items():
        for syn in sorted(synonyms, key=len, reverse=True):
            syn_lemma = " ".join(normalize_word(w) for w in syn.split())
            if re.search(rf"\b{re.escape(syn_lemma)}s?\b", sent_text):
                return key
    return None

def normalize_action(sentence):
    return match_with_synonyms(sentence, action_synonyms)

def normalize_process(sentence):
    sentence_lower = sentence.lower()
    for process, synonyms in process_synonyms.items():
        for syn in sorted(synonyms, key=len, reverse=True):
            if process == "salesorder" and syn == "order" and "purchase" in sentence_lower:
                continue
            if re.search(rf"\b{re.escape(syn)}s?\b", sentence_lower):
                return process
    return match_with_synonyms(sentence, process_synonyms)

def get_intent_with_score(user_sentence):
    user_emb = model.encode([user_sentence.lower().strip()], convert_to_tensor=False)[0]
    sims = [
        float(np.dot(user_emb, emb) / (np.linalg.norm(user_emb) * np.linalg.norm(emb)))
        for emb in intent_embeddings
    ]
    top_idx = int(np.argmax(sims))
    return intents[top_idx], sims[top_idx]

def truncate(text, max_len=50):
    if not text:
        return "not found"
    text = str(text)
    return text if len(text) <= max_len else text[:max_len-3] + "..."

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Intent Recognition API")

class Query(BaseModel):
    sentence: str

@app.get("/")
def root():
    return {
        "message": "Intent Recognition API is running",
        "usage": "POST a sentence to /predict/ to get action & process",
        "docs": "/docs"
    }

@app.post("/predict/")
def predict(query: Query):
    sentence = query.sentence.strip()
    action_found = normalize_action(sentence)
    process_found = normalize_process(sentence)

    # BOTH FOUND
    if action_found and process_found:
        return {
            "status": "success",
            "action": truncate(action_found),
            "process": truncate(process_found),
            "error": "no error detected"
        }

    # ACTION FOUND, PROCESS MISSING
    if action_found and not process_found:
        predicted_intent, score = get_intent_with_score(sentence)
        _, predicted_process = predicted_intent.split(' ', 1)
        if score >= EMBEDDING_ACCEPT_THRESHOLD:
            return {
                "status": "success",
                "action": truncate(action_found),
                "process": truncate(predicted_process),
                "error": "no error detected"
            }
        else:
            return {
                "status": "failure",
                "action": truncate(action_found),
                "process": "not found",
                "error": f"I understand you want to {truncate(action_found, 35)} something, but couldn’t figure out what."
            }

    # PROCESS FOUND, ACTION MISSING
    if process_found and not action_found:
        predicted_intent, score = get_intent_with_score(sentence)
        predicted_action, _ = predicted_intent.split(' ', 1)
        if score >= EMBEDDING_ACCEPT_THRESHOLD:
            return {
                "status": "success",
                "action": truncate(predicted_action),
                "process": truncate(process_found),
                "error": "no error detected"
            }
        else:
            return {
                "status": "failure",
                "action": "not found",
                "process": truncate(process_found),
                "error": f"Recognized process '{truncate(process_found, 40)}', but not action."
            }

    # NEITHER FOUND
    return {
        "status": "failure",
        "action": "not found",
        "process": "not found",
        "error": "Could not determine intent. Please rephrase."
    }
