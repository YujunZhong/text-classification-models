import os
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

nltk.data.path.append('/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/text-classification-models/nltk_data/')

from utils import save_model


class StemmedCountVectorizer(CountVectorizer):
    """ count vectorizer with stemming"""
    def build_analyzer(self):
        """ build analyzer.
        :return: stemmed text
        """
        stemmer = SnowballStemmer("english", ignore_stopwords=True)

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def train_nb(train_data, val_data):
    """ Training Naive Bayesian Models.
    :param train_data: transformed training data
    :param val_data: transformed validation data
    :return:
    """
    Train_X, Train_Y, Test_X, Test_Y = train_data['text'], train_data['label'], val_data['text'], val_data['label']

    text_clf_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('clf-nb', MultinomialNB()),
    ])

    _ = text_clf_nb.fit(Train_X, Train_Y)

    preds = text_clf_nb.predict(Test_X)
    acc = np.mean(preds == Test_Y)
    print(f'Test accurary of Naive Bayes model is: {acc}')

    save_model(text_clf_nb, './save/models/nb_trained_model.pkl')


def train_svm(train_data, val_data):
    """ Training SVM Models.
    :param train_data: transformed training data
    :param val_data: transformed validation data
    :return:
    """
    Train_X, Train_Y, Test_X, Test_Y = train_data['text'], train_data['label'], val_data['text'], val_data['label']

    # linearSVC
    text_clf_svm = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('clf-svm', LinearSVC(loss='hinge', C=0.5, class_weight='balanced')),
    ])

    _ = text_clf_svm.fit(Train_X, Train_Y)
    preds = text_clf_svm.predict(Test_X)
    acc = np.mean(preds == Test_Y)
    print(f'Test accurary of SVM model is: {acc}')

    save_model(text_clf_svm, './save/models/svm_trained_model.pkl')


if __name__ == "__main__":
    data_folder = "/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/data/kaggle-competition-2/new"
    train_data = pd.read_csv(os.path.join(data_folder, "train.csv"), encoding='latin-1')
    val_data = pd.read_csv(os.path.join(data_folder, "val.csv"), encoding='latin-1')
    # train_nb_gs(train_data, val_data)
    train_svm(train_data, val_data)
