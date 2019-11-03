import urllib.request, os, shutil, glob, re, nltk, gensim, scipy.stats, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PortugueseStemmer
from nltk import FreqDist

class CorpusHandler:

    def __init__(self, folder_path, language):
        self.folder_path = folder_path
        self.output_path = os.getcwd() + "\\Outputs"
        os.makedirs("Output",exist_ok=True)
        self.list_of_files = glob.glob(self.folder_path)
        # set the language for the stopwords
        self.stop_words = set(stopwords.words(language))
        # set the punctuactions
        symbols = {"º", "ª", "“", "”", "–", "¾", ',', '„', '⅞', '°', '\'', '£', '…', '’', '½'}
        self.punctuations = set(string.punctuation).union(symbols)
        # get the lemmas
        self.lemmas = PortugueseStemmer()
        # load and clean the texts
        self.texts = []
        for fil in self.list_of_files:
            with open(fil, "r", encoding='utf-8') as f:
                txt = f.read()
                self.texts.append(self._clean_text(txt))
        # create tokens
        self.tokens = [[w.lower() for w in word_tokenize(text)] for text in self.texts]
        # create dictionary
        self.dictionary = gensim.corpora.Dictionary(self.tokens)
        # create bag-of-words
        self.bag_of_words = [self.dictionary.doc2bow(token) for token in self.tokens]
        # create tf-idf
        self.tf_idf = gensim.models.TfidfModel(self.bag_of_words)

    def _clean_text(self, text):
        stop_free = " ".join([i for i in text.lower().split() if i not in self.stop_words])
        punc_free = ''.join(ch for ch in stop_free if ch not in self.punctuations)
        normalized = " ".join(self.lemmas.stem(word) for word in punc_free.split())
        return normalized

    def get_LDA_modelling(self):
        LDA = gensim.models.ldamodel.LdaModel
        return LDA(self.bag_of_words, num_topics=3, id2word = self.dictionary, passes=50)
    
    def get_corpus_frequent_terms(self, num_words):
        all_texts = ""
        for text in self.texts:
            all_texts += text
        fdist = FreqDist(all_texts.split())
        return fdist.most_common(num_words)

    def get_bigrams(self, doc_num):
        return nltk.bigrams(self.texts[doc_num].split())

    def get_trigrams(self, doc_num):
        return nltk.trigrams(self.texts[doc_num].split())
        

