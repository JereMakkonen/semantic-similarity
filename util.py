import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


def get_wn_tag(pos_tag):
    pos_map = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADJ, "RB": wn.ADV}
    return pos_map.get(pos_tag[:2], None)


def get_synsets(sentence, pos_tag):
    """Returns the first synset for each word with the specified POS tag, if available."""
    return [
        synsets[0] for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence))
        if get_wn_tag(tag) == pos_tag and (synsets := wn.synsets(word, pos=pos_tag))
    ]


def lemma_filter(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    sentence = [
        lemmatizer.lemmatize(token.lower(), get_wn_tag(pos) or wn.NOUN)
        for token, pos in nltk.pos_tag(nltk.word_tokenize(sentence))
    ]
    return " ".join(sentence)
  

def stopword_filter(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = [
        token.lower() for token in nltk.word_tokenize(sentence) 
        if token.lower() not in stop_words
    ]
    return " ".join(sentence)
    

def stop_lemma_filter(sentence):
    stop_words = set(stopwords.words('english'))
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
