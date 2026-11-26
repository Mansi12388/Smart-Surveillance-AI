"""
Text preprocessing for incident descriptions (NLTK + spaCy).

  * clean_text / basic_clean   : light regex cleaning
  * preprocess_tokens / _text  : spaCy tokenisation + lemmatisation + NLTK stop-words
                                 (used for classic LSTM / BiLSTM text baselines)

BERT does NOT need stop-word removal or lemmatisation - use basic_clean() for it.

One-time setup:
    python -m spacy download en_core_web_sm
    python -m nltk.downloader stopwords
"""
import re
from functools import lru_cache
from typing import Iterable, List, Set

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# negations carry meaning ("not moving") so they are kept even though NLTK lists them
_KEEP_WORDS = {"no", "nor", "not"}


def basic_clean(text: str) -> str:
    """Minimal cleaning for BERT: strip and collapse whitespace (case/punctuation kept)."""
    if text is None:
        return ""
    return _WHITESPACE.sub(" ", str(text)).strip()


def clean_text(text: str) -> str:
    """Lower-case, replace punctuation with spaces, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is missing. Run:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


@lru_cache(maxsize=1)
def get_stopwords() -> Set[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
    return words - _KEEP_WORDS


def _doc_to_tokens(doc, remove_stopwords: bool, lemmatize: bool) -> List[str]:
    stop = get_stopwords() if remove_stopwords else set()
    tokens = []
    for tok in doc:
        if tok.is_space:
            continue
        surface = tok.text.lower()
        word = tok.lemma_.lower() if lemmatize else surface
        if remove_stopwords and (surface in stop or word in stop):
            continue
        tokens.append(word)
    return tokens


def preprocess_tokens(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
    """Raw text -> list of cleaned tokens."""
    doc = _get_nlp()(clean_text(text))
    return _doc_to_tokens(doc, remove_stopwords, lemmatize)


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Raw text -> cleaned string (tokens joined by single spaces)."""
    return " ".join(preprocess_tokens(text, remove_stopwords, lemmatize))


def preprocess_batch(
    texts: Iterable[str], remove_stopwords: bool = True, lemmatize: bool = True
) -> List[str]:
    """Fast preprocessing for many texts (uses spaCy's nlp.pipe)."""
    cleaned = [clean_text(t) for t in texts]
    docs = _get_nlp().pipe(cleaned, batch_size=256)
    return [" ".join(_doc_to_tokens(d, remove_stopwords, lemmatize)) for d in docs]