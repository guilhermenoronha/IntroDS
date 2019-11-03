import os
from corpus_handler import CorpusHandler

if __name__ == "__main__":

    ch = CorpusHandler(os.getcwd() + "\\Corpus\\*.txt", 'portuguese')
    
    print("Bigrams of Document 1")
    print(*map(' '.join, ch.get_bigrams(1)), sep=', ')

    print("Trigrams of Document 1")
    print(*map(' '.join, ch.get_trigrams(1)), sep=', ')

    print("10 most frequent terms")
    print(ch.get_corpus_frequent_terms(2))

    print("First document LDA model")
    lda = ch.get_LDA_modelling()
    print(lda.print_topics(num_topics=1, num_words=3))
    
    
    