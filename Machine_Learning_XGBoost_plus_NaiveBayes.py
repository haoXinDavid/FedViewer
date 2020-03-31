import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.stem import SnowballStemmer
from xgboost import XGBClassifier, plot_importance
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
import pyLDAvis.gensim
from nltk.corpus import stopwords
import pickle
import warnings
warnings.filterwarnings('ignore')


file = open('tokenized_data.pkl', 'rb')
df = pickle.load(file)
file.close()


# Label the training data
label_range = 0.25

DiffLevel = 'diff_1'
df2= df[df[DiffLevel].notna()]
df2['y'] = df2[DiffLevel].apply(lambda x: -1 if x <= -label_range else
                                 1 if x>=label_range else 0 )

df2['y'].value_counts()


df3_stat = df2[df2['tokenized_statement'].notna()]
df3_min = df2[df2['tokenized_minute'].notna()]

# TFIDF vectorizer
vect = TfidfVectorizer(stop_words='english',
                       max_features = 1000,
                       ngram_range=(2,8),
                       token_pattern='[a-zA-Z]+')
X_statements = vect.fit_transform(df3_stat['tokenized_statement'])
X_minutes = vect.fit_transform(df3_min['tokenized_minute'])


# Create a DataFrame to examine
text_statements = pd.DataFrame(X_statements.toarray(), columns=vect.get_feature_names())
text_statements_tfidf = text_statements.copy() # create backup of TFIDF text for later use
text_statements.head()

text_minutes = pd.DataFrame(X_minutes.toarray(), columns=vect.get_feature_names())
text_minutes_tfidf = text_minutes.copy() # create backup of TFIDF text for later use
text_minutes.head()

#----------------------- Machine Learning --------------------------#
# ------------------------------------------------------------------#
## XGBoost model for statements
xgb_model = XGBClassifier(objective='multi:softmax',
                          max_depth=3,
                          num_classes=3)

train_scores = []
test_scores = []
for x in range(1,3):
    X_train, X_test, y_train, y_test = train_test_split(text_statements, df3_stat['y'])
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    test_score = round(xgb_model.score(X_test, y_test),2)
    train_score = round(xgb_model.score(X_train, y_train),2)
    # print(f'{xgb_model.get_params}')
    print(f'Run {x}... train: {train_score}, test: {test_score}')
    train_scores.append(train_score)
    test_scores.append(test_score)
print(f' Statements training average: {np.mean(train_scores)}')
print(f'Statements test average: {np.mean(test_scores)}')

#  XGBoost model for minutes
xgb_model2 = XGBClassifier(objective='multi:softmax',
                          max_depth=3,
                          num_classes=3)

train_scores2 = []
test_scores2 = []
for x in range(1,3):
    X_train, X_test, y_train, y_test = train_test_split(text_minutes, df3_min['y'])
    xgb_model2.fit(X_train, y_train)
    y_pred = xgb_model2.predict(X_test)

    test_score2 = round(xgb_model2.score(X_test, y_test),2)
    train_score2 = round(xgb_model2.score(X_train, y_train),2)
    # print(f'{xgb_model.get_params}')
    print(f'Run {x}... train: {train_score2}, test: {test_score2}')
    train_scores2.append(train_score2)
    test_scores2.append(test_score2)
print(f'Minutes Train average: {np.mean(train_scores2)}')
print(f'Minutes Test average: {np.mean(test_scores2)}')



# Feature importance

fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams.update({'font.size': 8})
ax.yaxis.label.set_size(2)
plot_importance(xgb_model, max_num_features = 15, ax=ax);

df_inc = df3_stat[df3_stat['y'] == 1]

#-------------- Naive Bayes model -----------------#
for i in range(1,5):
    X_train, X_test, y_train, y_test = train_test_split(text_statements, df3_stat['y'])
    nb = MultinomialNB()
    nb.fit(X_train,y_train)

    nb.classes_
    nb.coef_[0]
    pred = nb.predict(X_test)
    print(metrics.accuracy_score(y_test,pred))

for i in range(1,5):
    X_train, X_test, y_train, y_test = train_test_split(text_minutes, df3_min['y'])
    nb = MultinomialNB()
    nb.fit(X_train,y_train)

    nb.classes_
    nb.coef_[0]
    pred = nb.predict(X_test)
    print(metrics.accuracy_score(y_test,pred))