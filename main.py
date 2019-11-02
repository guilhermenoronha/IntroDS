from corpus_handler import CorpusHandler
import glob, os
import pandas as pd

import gensim

if __name__ == "__main__":

    ch = CorpusHandler(os.getcwd() + "\\Corpus\\*.txt", 'portuguese')
    lda = ch.LDA_modelling()
    print(lda.print_topics(num_topics=3, num_words=3))
    