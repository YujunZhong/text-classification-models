import pickle


def save_model(clf, filename='svm_trained_model.pkl'):
    pickle.dump(clf, open(filename, 'wb'))


def load_model(model_path):
    clf = pickle.load(open(model_path, 'rb'))

    return clf