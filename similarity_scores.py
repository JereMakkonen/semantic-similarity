import nltk
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr


def get_wn_tag(pos_tag):
    pos_map = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADV, "RB": wn.ADJ}
    return pos_map.get(pos_tag[:2], None)


def get_synsets(sentence: str, pos_tag):
    """Returns the first synset for each word with the specified POS tag, if available."""
    return [
        synsets[0] for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence))
        if get_wn_tag(tag) == pos_tag and (synsets := wn.synsets(word, pos=pos_tag))
    ]


def calculate_max_wup(s1: str, s2: str, pos_tag):
    """Returns the maximum Wu-Palmer similarity between two strings for the pos tag."""
    return max((
        syn1.wup_similarity(syn2) or 0.0 
        for syn1 in get_synsets(s1, pos_tag) for syn2 in get_synsets(s2, pos_tag)
        ), default=0.0
    )


def noun_verb_wup(s1: str, s2: str):
    """For sentence pair S1 and S2 calculates: [(max noun WUP similarity) + (max verb WUP similarity)] / 2"""
    
    max_noun_score = calculate_max_wup(s1, s2, wn.NOUN)
    max_verb_score = calculate_max_wup(s1, s2, wn.VERB)
    
    return round((max_noun_score + max_verb_score) / 2, 3)

