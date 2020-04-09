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

# file = open('tokenized_data.pkl', 'rb')
file = open('tokenized_data3.pkl', 'rb')
df = pickle.load(file)
file.close()

# Label the training data
label_range = 0.25

DiffLevel = 'diff_1'
df2 = df.copy()
df2['y'] = df2[DiffLevel].apply(lambda x: -1 if x <= -label_range else
1 if x >= label_range else 0)

df2['y'].value_counts()

df3_stat = df2[df2['tokenized_statement'].notna()]
df3_min = df2[df2['tokenized_minute'].notna()]

# TFIDF vectorizer
vect = TfidfVectorizer(stop_words='english',
                       max_features=100,
                       ngram_range=(2, 8),
                       token_pattern='[a-zA-Z]+')
X_statements = vect.fit_transform(df3_stat['tokenized_statement'])
X_minutes = vect.fit_transform(df3_min['tokenized_minute'])

# Create a DataFrame to examine
text_statements = pd.DataFrame(X_statements.toarray(), columns=vect.get_feature_names())

text_minutes = pd.DataFrame(X_minutes.toarray(), columns=vect.get_feature_names())

# ----------------------- Machine Learning --------------------------#
# ------------------------------------------------------------------#
## XGBoost model for statements
xgb_model = XGBClassifier(objective='multi:softmax',
                          max_depth=3,
                          num_classes=3)

test_scores = []
for i in range(1, 20):
    X_train, X_test, y_train, y_test = train_test_split(text_statements[:-3], df3_stat.iloc[:-3]['y'])
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    print("XGBoost prediction for the most recent statements")
    print(xgb_model.predict(text_statements[-3:]))

    test_score = xgb_model.score(X_test, y_test)
    # print(f'{xgb_model.get_params}')
    print(f'Run {i}... ,XGBoost test score: {test_score}')
    test_scores.append(test_score)

print(f'XGBoost Statements test average: {np.mean(test_scores)}')

#  XGBoost model for minutes
xgb_model2 = XGBClassifier(objective='multi:softmax',
                           max_depth=3,
                           num_classes=3)

test_scores2 = []
for i in range(1, 6):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(text_minutes, df3_min['y'])
    xgb_model2.fit(X_train2, y_train2)
    y_pred2 = xgb_model2.predict(X_test2)

    test_score2 = xgb_model2.score(X_test2, y_test2)
    # print(f'{xgb_model.get_params}')
    print(f'Run {i}... , XGBoost test score: {test_score2}')
    test_scores2.append(test_score2)

print(f'XGBoost Minutes Test average: {np.mean(test_scores2)}')

# Prediction given most recent document
print("XGBoost prediction for the most recent statements")
print(xgb_model.predict(text_statements[-3:]))

# Feature importance

fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams.update({'font.size': 8})
ax.yaxis.label.set_size(2)
plot_importance(xgb_model, max_num_features=15);
plot_importance(xgb_model2, max_num_features=15);

feature_importances = pd.DataFrame(xgb_model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
for i in range(len(feature_importances)): print(i + 1, ' ', feature_importances.index[i])

feature_importances2 = pd.DataFrame(xgb_model2.feature_importances_,
                                    index=X_train2.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

for i in range(len(feature_importances2)): print(i + 1, ' ', feature_importances2.index[i])

# -------------- Naive Bayes model -----------------#
test_scoresNB = []
for i in range(1, 20):
    X_trainN, X_testN, y_trainN, y_testN = train_test_split(text_statements[:-3], df3_stat.iloc[:-3]['y'])
    nb = MultinomialNB()
    nb.fit(X_trainN, y_trainN)
    # nb.classes_
    # nb.coef_[0]
    pred = nb.predict(X_testN)
    print("Naive Bayes prediction for the most recent statements")
    print(nb.predict(text_statements[-3:]))
    score = metrics.accuracy_score(y_testN, pred)
    test_scoresNB.append(score)
    print(f'Run {i}... , Naive Bayes test score: {score}')

print(f'Naive Bayes Statements test average: {np.mean(test_scoresNB)}')

# Prediction given most recent document
print("Naive Bayes prediction for the most recent statements")
print(nb.predict(text_statements[-1:]))

test_scoresNB2 = []
for i in range(1, 6):
    X_train, X_test, y_train, y_test = train_test_split(text_minutes, df3_min['y'])
    nb2 = MultinomialNB()
    nb2.fit(X_train, y_train)
    pred = nb2.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    test_scoresNB2.append(score)
    print(f'Run {i}... , Naive Bayes test score: {score}')
print(f'Naive Bayes Minutes test average: {np.mean(test_scoresNB2)}')


def show_most_informative_features(vectorizer, nb, n=20):
    feature_names = vect.get_feature_names()
    coefs_with_fns = sorted(zip(nb.coef_[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    print("Top 20 most important terms")
    for (coef_2, fn_2) in top:
        #     print("\t%.4f\t%-15s" % (coef_2, fn_2))
        print(fn_2)


show_most_informative_features(vect, nb)
show_most_informative_features(vect, nb2)


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
                          "\n".join(feature_names[j] for j in top10)))


print_top10(vect, nb, nb.classes_)
print_top10(vect, nb2, nb2.classes_)
