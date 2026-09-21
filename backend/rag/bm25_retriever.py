"""
bm25_retriever.py
-----------------
Simple BM25 keyword search over a list of LangChain Documents.
No external dependencies required.
"""

import math
import re
from collections import Counter
from langchain_core.documents import Document


def _tokenize(text: str) -> list[str]:
    """Lowercase + remove punctuation + split into words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Simple BM25 retriever.

    Parameters
    ----------
    documents : list[Document]  – the chunks to search over
    k1        : float           – term frequency saturation (default 1.5)
    b         : float           – length normalization (default 0.75)
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        # Tokenize every document once
        self.tokenized = [_tokenize(d.page_content) for d in documents]

        # Average document length
        self.avgdl = sum(len(t) for t in self.tokenized) / self.N

        # Document frequency: how many docs contain each term
        self.df: dict[str, int] = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for one document."""
        tf = Counter(self.tokenized[doc_idx])
        dl = len(self.tokenized[doc_idx])
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = math.log((self.N - self.df.get(term, 0) + 0.5) /
                           (self.df.get(term, 0) + 0.5) + 1)
            numerator = tf[term] * (self.k1 + 1)
            denominator = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def get_top_k(self, query: str, k: int = 4) -> list[Document]:
        """Return the top-k documents ranked by BM25 score."""
        query_tokens = _tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.docs[i] for i, _ in scores[:k]]
