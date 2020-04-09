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

import codecs
import datetime as dt
import gc
import os

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

'''-----------------------------------------------------------------------------------------------------------------'''
count = 0


## Preprocessing

def Preprocess():
    '''
    Read in files of Fed Minutes, Statements, and target rates.
    Merge files into a Dataframe with dates aligned.
    Calc the diff of target rates.
    :return: Merged Dataframe of all data
    '''
    path_dir_minutes = os.getcwd() + os.sep + 'minutes'
    path_dir_statements = os.getcwd() + os.sep + 'statements'
    path_target_rate_1 = os.getcwd() + os.sep + 'rates' + os.sep + 'DFEDTAR.csv'
    path_target_rate_2 = os.getcwd() + os.sep + 'rates' + os.sep + 'DFEDTARL(1).csv'

    '''concate statements files'''
    file_in_statement_folder = os.listdir(path_dir_statements)
    statements, statement_dates = [], []  # list of statements' content and correspondent date
    for file_name in file_in_statement_folder:
        if file_name[-3:] == 'txt':
            path_file = path_dir_statements + os.sep + file_name
            statement = codecs.open(path_file,
                                    encoding='utf-8').read().strip()  # This way of reading .txt files does not generate garbled text
            date = file_name[:8]
            statements.append(statement)
            statement_dates.append(dt.datetime.strptime(date, '%Y%m%d'))
    df_statements = pd.DataFrame(list(zip(statement_dates, statements)), columns=['dates', 'statement'])
    print(df_statements)

    '''concate statements files'''
    file_in_minute_folder = os.listdir(path_dir_minutes)
    minutes, minute_dates = [], []
    for file_name in file_in_minute_folder:
        if file_name[-3:] == 'txt':
            path_file = path_dir_minutes + os.sep + file_name
            minute = codecs.open(path_file, encoding='utf-8').read().strip()
            date = file_name[:8]
            minutes.append(minute)
            minute_dates.append(dt.datetime.strptime(date, '%Y%m%d'))
    df_minutes = pd.DataFrame(list(zip(minute_dates, minutes)), columns=['dates', 'minute'])
    print(df_minutes)

    '''concate Fed Target rates '''
    df_target_rate_1 = pd.read_csv(path_target_rate_1, index_col=False)
    df_target_rate_2 = pd.read_csv(path_target_rate_2, index_col=False)
    df_target_rate_1 = df_target_rate_1.rename(columns={'DFEDTAR': 'target_rate'})
    df_target_rate_2 = df_target_rate_2.rename(columns={'DFEDTARL': 'target_rate'})
    df_target_rate = pd.concat([df_target_rate_1, df_target_rate_2])
    df_target_rate = df_target_rate.rename(columns={'DATE': 'dates'})
    df_target_rate['dates'] = pd.to_datetime(df_target_rate['dates'])
    print(df_target_rate)

    df_merged = pd.merge(df_statements, df_minutes, how='outer', on='dates', sort=True)
    df_merged = pd.merge(df_merged, df_target_rate, how='inner', on='dates', sort=True)
    print(df_merged.to_string())

    df_merged['target_rate_lead1'] = df_merged['target_rate'].shift(-1)
    df_merged['target_rate_lead3'] = df_merged['target_rate'].shift(-3)
    df_merged['target_rate_lead6'] = df_merged['target_rate'].shift(-6)
    df_merged['diff_1'] = df_merged['target_rate_lead1'] - df_merged['target_rate']
    df_merged['diff_3'] = df_merged['target_rate_lead3'] - df_merged['target_rate']
    df_merged['diff_6'] = df_merged['target_rate_lead6'] - df_merged['target_rate']
    print(df_merged.to_string())

    df_merged.to_pickle(os.getcwd() + os.sep + 'merged.pkl')
    return df_merged


def tokenize(corpus):
    global count
    count += 1
    print(count)
    tokenized_corpus = word_tokenize(corpus.lower())
    wnl = WordNetLemmatizer()
    new_list = []
    for word in tokenized_corpus:

        if word not in stopwords.words('english'):
            lower_word = word.lower()
            lem_word = wnl.lemmatize(lower_word)

            if lem_word not in stopwords.words('english'):
                new_list.append(lem_word)

    new_corpus = ' '.join(new_list)
    return new_corpus


def Process():
    '''
    Stemming, tf-idf vectorization, modelling
    :return:
    '''

    df_data = pd.read_pickle(os.getcwd() + os.sep + 'merged.pkl')

    df_data['tokenized_statement'] = [tokenize(i) if pd.notna(i) else np.nan for i in df_data['statement']]
    df_data.to_pickle(os.getcwd() + os.sep + 's_tokenized_data.pkl')
    print("Done tokenizing statement.")
    global count
    count = 0
    gc.collect()

    df_data = pd.read_pickle(os.getcwd() + os.sep + 's_tokenized_data.pkl')
    df_data['tokenized_minute'] = [tokenize(i) if pd.notna(i) else np.nan for i in df_data['minute']]

    df_data.to_pickle(os.getcwd() + os.sep + 'tokenized_data.pkl')
    print("Done tokenizing minute.")


if __name__ == "__main__":
    Preprocess()
    Process()
