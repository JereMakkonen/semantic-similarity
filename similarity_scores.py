import numpy as np
from functools import lru_cache
import nltk
from nltk.corpus import wordnet as wn
from util import extract_synsets


@lru_cache(maxsize=100_000)
def wup(syn1, syn2):
    return syn1.wup_similarity(syn2) or 0

def max_wup(syn1, syn2):
    return max((wup(s1, s2) for s1 in syn1 for s2 in syn2), default=0)


def mean_wup(s1: list[str], s2: list[str]):
    scores = []

    for a, b in zip(s1, s2):
        syn1, syn2 = extract_synsets(a), extract_synsets(b)

        noun_score = max_wup(syn1[wn.NOUN], syn2[wn.NOUN])
        verb_score = max_wup(syn1[wn.VERB], syn2[wn.VERB])

        scores.append((noun_score + verb_score) / 2)
    
    return scores


def embedding_similarity(model, s1: list[str], s2: list[str]):
    def vectorize(sentences):
        return np.array([
            np.mean(v, axis=0) if v else np.zeros(model.vector_size)
            for s in sentences for v in 
            [[model[t] for t in nltk.word_tokenize(s) if t in model]]
        ])
        
    v1, v2 = vectorize(s1), vectorize(s2)
    d = np.vecdot(v1, v2, axis=1)
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

    return np.divide(d, n, out=np.zeros_like(d), where=n != 0)


def bert_similarity(model, s1: list[str], s2: list[str]):
    e = model.encode(s1 + s2, normalize_embeddings=True)
    return np.vecdot(e[:len(s1)], e[len(s1):], axis=1)

