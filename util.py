import random
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


def get_wn_tag(pos_tag):
    pos_map = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADJ, "RB": wn.ADV}
    return pos_map.get(pos_tag[:2], None)


@lru_cache(maxsize=50_000)
def extract_synsets(sentence):
    """Extract noun and verb synsets for a given sentence."""
    result = {wn.NOUN: [], wn.VERB: []}
    
    for w, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if (pos := get_wn_tag(tag)) in result and (syn := wn.synsets(w, pos)):
            result[pos].append(syn[0])

    return {pos: tuple(syn) for pos, syn in result.items()}


def lemma_filter(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    sentence = [
        lemmatizer.lemmatize(token.lower(), get_wn_tag(pos) or wn.NOUN)
        for token, pos in nltk.pos_tag(nltk.word_tokenize(sentence))
    ]
    return " ".join(sentence)
  

def stopword_filter(sentence):
    stop_words = set(stopwords.words("english"))
    sentence = [
        token.lower() for token in nltk.word_tokenize(sentence) 
        if token.lower() not in stop_words
    ]
    return " ".join(sentence)
    

def stop_lemma_filter(sentence):
    stop_words = set(stopwords.words("english"))
    lemmatizer = nltk.WordNetLemmatizer()
    sentence = [
        lemmatizer.lemmatize(token.lower(), get_wn_tag(pos) or wn.NOUN)
        for token, pos in nltk.pos_tag(nltk.word_tokenize(sentence))
        if token not in stop_words
    ]
    return " ".join(sentence)


def alphabet_filter(sentence):
    sentence = [t.lower() for t in nltk.word_tokenize(sentence) if t.isalpha()]
    return " ".join(sentence)


def get_synonyms(word):
    """get_synonyms(word: (token, wn_tag)) -> (word, synonyms) or None."""
    token, tag = word
    
    synonyms = {
        syn for s in wn.synsets(token.lower(), pos=tag) for l in s.lemmas()
        if (syn := l.name().replace("_", " ").lower()) != token.lower()
    }
    return synonyms or None


def words_with_tag(sentence, tagset):
    """Returns words: (word, wn_tag) in sentence grouped by WordNet POS tag."""
    candidates = []
    stop_words = set(stopwords.words("english"))
    
    for token, pos_tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        wn_tag = get_wn_tag(pos_tag[:2])
        if wn_tag in tagset and token not in stop_words:
            candidates.append((token, wn_tag))

    return candidates


def get_verb_adjective(sentence):
    """Returns a random verb and adjective/adverb (word, wn_tag, pos_tag)."""
    
    verb_candidates = words_with_tag(sentence, [wn.VERB])
    adj_candidates = words_with_tag(sentence, [wn.ADJ, wn.ADV])
    
    if verb_candidates and adj_candidates:
        return random.choice(verb_candidates), random.choice(adj_candidates)
    
    return None

