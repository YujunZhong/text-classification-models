import pandas as pd
from torch.utils.data import Dataset

import contractions
import re
import string
# lemmatizer = WordNetLemmatizer()


def text_preprocessing(df):
    """ Includes all data preprocessing.
    :param df: raw dataframe
    :param df: preprocessed data
    :return:
    """
    df['new_text']=df['text'].apply(lambda x: contractions.fix(x, slang=True))
    df['new_text'] = df['new_text'].str.lower()
    df['new_text'] = df['new_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    df['new_text'] = df['new_text'].apply(lambda x: re.sub(r'\w*\d\w*', '', x))
    df['new_text'] = df['new_text'].apply(lambda x: re.sub(' +', ' ', x))
    return df