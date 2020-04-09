'''
Created on March 28, 2020
MFin NLP Project _ FedViewer
Group members:
Xin         Hao
Jiayi       Chen
Haili       Wang
Shaokang    Wang
Zhaokai     Wang
Jingyu      Zhao, Jingyuz@connect.hku.hk 3035644697
'''

import os
import codecs
import datetime as dt
import nltk
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import pyLDAvis.gensim
'''-----------------------------------------------------------------------------------------------------------------'''


def LDA_3scenarios():
    df_data = pd.read_pickle(os.getcwd() + os.sep + 'tokenized_data.pkl')
    df_data = df_data.set_index('dates')  # set dates to be the index for easier checking the data alignment

    df_minute = df_data[['tokenized_minute', 'diff_1']].dropna()
    tokenized_minutes = df_minute['tokenized_minute']
    # df_statement = df_data[['tokenized_statement', 'diff_1']].dropna()
    # tokenized_statements = df_statement['tokenized_statement']

    # change the target rate difference into categories (1,0,-1)
    y_threshold = 0.25
    df_minute['y'] = df_minute['diff_1']
    df_minute['y'].mask(df_minute['y'] >= y_threshold, 1, inplace=True)
    df_minute['y'].mask(df_minute['y'] <= -y_threshold, -1, inplace=True)
    df_minute['y'].mask((df_minute['y'] > -y_threshold) & (df_minute['y'] < y_threshold), 0, inplace=True)
    # print(df_minute)

    # split the data into 3 groups by direction of the Fed's target rate movement
    df_hawkish = df_minute[df_minute['y'] == 1]
    df_neutral = df_minute[df_minute['y'] == 0]
    df_dovish = df_minute[df_minute['y'] == -1]

    # Instantiate a TFIDF vectorizer
    vect = TfidfVectorizer(stop_words='english',
                           max_features=3000,
                           ngram_range=(2, 4),
                           token_pattern='[a-zA-Z]+')

    df_tfidf_hawkish = vect.fit_transform(df_hawkish['tokenized_minute'])
    text_hawkish = pd.DataFrame(df_tfidf_hawkish.toarray(), columns=vect.get_feature_names())
    df_tfidf_neutral = vect.fit_transform(df_neutral['tokenized_minute'])
    text_neutral = pd.DataFrame(df_tfidf_neutral.toarray(), columns=vect.get_feature_names())
    df_tfidf_dovish = vect.fit_transform(df_dovish['tokenized_minute'])
    text_dovish = pd.DataFrame(df_tfidf_dovish.toarray(), columns=vect.get_feature_names())
    # print(df_minute['y'].value_counts())

    # Now we have the Term Document Matrix as the input for the LDA model
    # LDA for hawkish scenario
    token_text_hawkish = [text_hawkish.columns[text_hawkish.loc[index, :].to_numpy().nonzero()] for index in
                          text_hawkish.index]
    dictionary = corpora.Dictionary(token_text_hawkish)
    corpus = [dictionary.doc2bow(text) for text in token_text_hawkish]
    lda_hawkish = models.ldamodel.LdaModel(corpus,
                                           id2word=dictionary,
                                           # Matches each word to its "number" or "spot" in the dictionary
                                           num_topics=2,  # Number of topics T to find
                                           passes=5,  # Number of passes through corpus; similar to number of epochs
                                           minimum_probability=0.01)  # Only include topics above this probability threshold
    graph_hawkish = pyLDAvis.gensim.prepare(lda_hawkish, corpus, dictionary)
    pyLDAvis.show(graph_hawkish)

    # LDA for neutral scenario # have to do this repetitively otherwise the graph won't show
    token_text_neutral = [text_neutral.columns[text_neutral.loc[index, :].to_numpy().nonzero()] for index in
                          text_neutral.index]
    dictionary = corpora.Dictionary(token_text_neutral)
    corpus = [dictionary.doc2bow(text) for text in token_text_neutral]
    lda_neutral = models.ldamodel.LdaModel(corpus,
                                           id2word=dictionary,
                                           # Matches each word to its "number" or "spot" in the dictionary
                                           num_topics=2,  # Number of topics T to find
                                           passes=5,  # Number of passes through corpus; similar to number of epochs
                                           minimum_probability=0.01)  # Only include topics above this probability threshold
    graph_neutral = pyLDAvis.gensim.prepare(lda_neutral, corpus, dictionary)
    pyLDAvis.show(graph_neutral)

    # LDA for dovish scenario # have to do this repetitively otherwise the graph won't show
    token_text_dovish = [text_dovish.columns[text_dovish.loc[index, :].to_numpy().nonzero()] for index in
                         text_dovish.index]
    dictionary = corpora.Dictionary(token_text_dovish)
    corpus = [dictionary.doc2bow(text) for text in token_text_dovish]
    lda_dovish = models.ldamodel.LdaModel(corpus,
                                          id2word=dictionary,
                                          # Matches each word to its "number" or "spot" in the dictionary
                                          num_topics=2,  # Number of topics T to find
                                          passes=5,  # Number of passes through corpus; similar to number of epochs
                                          minimum_probability=0.01)  # Only include topics above this probability threshold
    graph_dovish = pyLDAvis.gensim.prepare(lda_dovish, corpus, dictionary)
    pyLDAvis.show(graph_dovish)


def LDA():
    df_data = pd.read_pickle(os.getcwd() + os.sep + 'tokenized_data.pkl')
    df_data = df_data.set_index('dates')  # set dates to be the index for easier checking the data alignment
    print(df_data.head(5).to_string())

    df_minute = df_data[['tokenized_minute', 'diff_1']].dropna()
    tokenized_minutes = df_minute['tokenized_minute']

    # change the target rate difference into categories (1,0,-1)
    y_threshold = 0.25
    df_minute['y'] = df_minute['diff_1']
    df_minute['y'].mask(df_minute['y'] >= y_threshold, 1, inplace=True)
    df_minute['y'].mask(df_minute['y'] <= -y_threshold, -1, inplace=True)
    df_minute['y'].mask((df_minute['y'] > -y_threshold) & (df_minute['y'] < y_threshold), 0, inplace=True)
    # print(df_minute)

    # Instantiate a TFIDF vectorizer
    vect = TfidfVectorizer(stop_words='english',
                           max_features=3000,
                           ngram_range=(2, 5),
                           token_pattern='[a-zA-Z]+')

    df_tfidf = vect.fit_transform(df_minute['tokenized_minute'])
    text = pd.DataFrame(df_tfidf.toarray(), columns=vect.get_feature_names())

    token_text = [text.columns[text.loc[index, :].to_numpy().nonzero()] for index in text.index]
    dictionary = corpora.Dictionary(token_text)
    corpus = [dictionary.doc2bow(text) for text in token_text]
    lda = models.ldamodel.LdaModel(corpus,
                                   id2word=dictionary,  # Matches each word to its "number" or "spot" in the dictionary
                                   num_topics=7,  # Number of topics T to find
                                   passes=5,  # Number of passes through corpus; similar to number of epochs
                                   minimum_probability=0.01)  # Only include topics above this probability threshold
    # graph = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    for topic in lda.print_topics():
        print(topic)
    # pyLDAvis.show(graph)


def Noun_Phrase_Preprocessing():
    '''tokenize minutes into noun phrases'''
    from textblob import TextBlob

    df_data = pd.read_pickle(os.getcwd() + os.sep + 'tokenized_data.pkl')
    df_data = df_data.set_index('dates')  # set dates to be the index for easier checking the data alignment
    df_minute = df_data[['minute', 'diff_1', 'tokenized_minute']].dropna()
    df_minute['minute'] = df_minute['minute'].apply(lambda x: x.lower())
    print(df_data.head(5).to_string())
    print('Tokenizing noun phrase......................................')
    df_minute['tokenized_noun_phrase'] = df_minute['minute'].apply(lambda x: TextBlob(x).noun_phrases)

    df_minute.to_pickle(os.getcwd() + os.sep + 'tokenized_noun_phrase.pkl')


def LDA_Noun_Phrase():
    '''try to parse out only nouns phrases from the Fed minutes as tokens'''
    '''helper function. dummy tokenizer. Doesn't do anything'''

    def dummy_fun(doc):
        return doc

    vect = TfidfVectorizer(  # This way we can fit a collections of documents already tokenized
        max_features=3000,
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
    )
    df_minute = pd.read_pickle(os.getcwd() + os.sep + 'tokenized_noun_phrase.pkl')
    df_tfidf = vect.fit_transform(df_minute['tokenized_noun_phrase'])
    text = pd.DataFrame(df_tfidf.toarray(), columns=vect.get_feature_names())
    print(text.head().to_string())

    token_text = [text.columns[text.loc[index, :].to_numpy().nonzero()] for index in text.index]
    dictionary = corpora.Dictionary(token_text)
    corpus = [dictionary.doc2bow(text) for text in token_text]
    lda = models.ldamodel.LdaModel(corpus,
                                   id2word=dictionary,  # Matches each word to its "number" or "spot" in the dictionary
                                   num_topics=6,  # Number of topics T to find
                                   passes=6,  # Number of passes through corpus; similar to number of epochs
                                   random_state=6,
                                   minimum_probability=0.01)  # Only include topics above this probability threshold
    for topic in lda.print_topics():
        print(topic)
    # graph = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # pyLDAvis.show(graph)

    # Look at which topic each minutes contains
    corpus_transformed = lda[corpus]
    print(list(zip([a for [(a, b)] in corpus_transformed], text.index)))


if __name__ == "__main__":
    # LDA_3scenarios()
    # LDA()
    # Noun_Phrase_Preprocessing()
    LDA_Noun_Phrase()
