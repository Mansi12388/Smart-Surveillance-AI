"""
Incident-description classification with a fine-tuned BERT (PyTorch + Hugging Face).

Model contract (created in notebook 06):
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
with labels in the order of INCIDENT_CLASSES.
"""
from functools import lru_cache
from typing import Dict, List

from src.config import BERT_MAX_LEN, BERT_MODEL_DIR, INCIDENT_CLASSES
from src.text_preprocessing import basic_clean


@lru_cache(maxsize=1)
def load_bert():
    """Return (tokenizer, model, device, label_names)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (BERT_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Fine-tuned BERT not found in {BERT_MODEL_DIR}. Run notebook 06 first."
        )
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n = model.config.num_labels
    id2label = model.config.id2label
    names = [str(id2label[i]) for i in range(n)]
    if all(name.startswith("LABEL_") for name in names):  # generic names -> use ours
        if n != len(INCIDENT_CLASSES):
            raise ValueError(f"Model has {n} labels but INCIDENT_CLASSES has {len(INCIDENT_CLASSES)}.")
        names = list(INCIDENT_CLASSES)
    return tokenizer, model, device, names


def predict_incidents(texts: List[str]) -> List[Dict]:
    """Classify several incident descriptions in one batch."""
    import torch

    cleaned = [basic_clean(t) for t in texts]
    if not cleaned or any(not t for t in cleaned):
        raise ValueError("Incident description must not be empty.")

    tokenizer, model, device, names = load_bert()
    enc = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=BERT_MAX_LEN,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for text, p in zip(cleaned, probs):
        idx = int(p.argmax())
        results.append(
            {
                "text": text,
                "label": names[idx],
                "confidence": float(p[idx]),
                "probabilities": {n: float(v) for n, v in zip(names, p)},
            }
        )
    return results


def predict_incident(text: str) -> Dict:
    """Classify one incident description. Adds an automatic one-line summary."""
    result = predict_incidents([text])[0]
    result["summary"] = generate_summary(result["text"], result["label"], result["confidence"])
    return result


def generate_summary(text: str, label: str, confidence: float) -> str:
    """Template-based incident summary (rule-based, not a generative model)."""
    return (
        f"Incident classified as '{label}' with {confidence * 100:.1f}% confidence. "
        f"Report: \"{text}\""
    )