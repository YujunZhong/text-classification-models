import pickle


def save_model(clf, filename='svm_trained_model.pkl'):
    """ Interface to quickly save models
    :param clf: classifier to save
    :param filename: filename to save
    :return:
    """
    pickle.dump(clf, open(filename, 'wb'))


def load_model(model_path):
    """ Interface to quickly load models
    :param model_path: model path
    :return: the loaded classifier
    """
    clf = pickle.load(open(model_path, 'rb'))

    return clf