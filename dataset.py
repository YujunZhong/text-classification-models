import pandas as pd
from torch.utils.data import Dataset

import contractions
import re
import string
# lemmatizer = WordNetLemmatizer()


def text_preprocessing(df):
    #1) Expand contractions in Text Processing
    df['new_text']=df['text'].apply(lambda x: contractions.fix(x, slang=True))
    #2) Lower Case
    df['new_text'] = df['new_text'].str.lower()
    #3) Remove punctuations
    df['new_text'] = df['new_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    #4) Remove words containing digits
    df['new_text'] = df['new_text'].apply(lambda x: re.sub(r'\w*\d\w*', '', x))
    #5) Remove Stopwords
    #def remove_stopwords(text):
    #    return " ".join([word for word in str(text).split() if word not in stop_words])
    #df['new_text'] = df['new_text'].apply(lambda x: remove_stopwords(x))
    #6) Lemmatization
    # def lemmatize_words(text):
    #     return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    # df['new_text'] = df['new_text'].apply(lambda text: lemmatize_words(text))
    #7) Remove Extra Spaces
    df['new_text'] = df['new_text'].apply(lambda x: re.sub(' +', ' ', x))
    return df