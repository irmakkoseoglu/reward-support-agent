"""Finds the help-centre articles relevant to a question (the "R" in RAG).

The help centre is small, so a TF-IDF index is enough: it runs offline, costs
nothing, needs no API key and is fully deterministic, which keeps the eval
reproducible. The interface (search -> list of docs) stays the same if it is
swapped for an embedding model later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_DIR = Path(__file__).resolve().parent.parent / "data" / "faq"


@dataclass
class Doc:
    id: str
    title: str
    text: str


class Retriever:
    def __init__(self, faq_dir: Path = FAQ_DIR):
        self.docs = []
        for p in sorted(faq_dir.glob("*.md")):
            text = p.read_text()
            title = text.splitlines()[0].lstrip("# ").strip()
            self.docs.append(Doc(p.stem, title, text))
        # titles are repeated so they weigh more than body text
        corpus = [f"{d.title} {d.title} {d.text}" for d in self.docs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(corpus)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Doc, float]]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return [(d, round(float(s), 3)) for d, s in ranked[:k] if s >= min_score]
