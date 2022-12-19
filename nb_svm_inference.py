import os
import pandas as pd
import nltk

from tools.utils import load_model

nltk.data.path.append('/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/text-classification-models/nltk_data/') 


def test_nb_svm(test_data):
    """ Model inference and save the results.
    :param test_data: transformed test data
    :return:
    """
    text_clf_svm = load_model('./save/models/svm_trained_model.pkl')
    preds = text_clf_svm.predict(test_data['text'])

    df = pd.DataFrame({'target': preds})
    df['id'] = df.index
    df.to_csv('./save/outputs/test_results_bi.csv', columns=['id', 'target'], index=False)


if __name__ == "__main__":
    data_folder = "/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/data/kaggle-competition-2"
    test_data = pd.read_csv(os.path.join(data_folder, "test_data.csv"), encoding='latin-1')
    test_nb_svm(test_data)