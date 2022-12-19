import pandas as pd
import os


def gen_train_val(data_path, label_path, output_folder):
    """ Split the training data and validation data, and convert data format.
    :param data_path: raw data path
    :param label_path: raw label path
    :param output_folder: root directory for storing newly generated data
    :return:
    """
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

    df = pd.read_csv(data_path)
    df_y = pd.read_csv(label_path)

    df['label'] = df_y.apply(lambda row: label_dict[row['target']], axis=1)

    df = df[['text', 'label']]    
    df.to_csv(os.path.join(output_folder, 'all.csv'), index=False)
    val_df = df.sample(frac=0.06)
    train_df = df.drop(val_df.index)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_folder, 'val.csv'), index=False)


def gen_test(data_path, output_folder):
    """ Convert test data format.
    :param data_path: raw test data path
    :param output_folder: root directory for storing newly generated data
    :return:
    """
    df = pd.read_csv(data_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(os.path.join(output_folder, 'test.csv'))


if __name__ == "__main__":
    data_path = "/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/data/kaggle-competition-2/train_data.csv"
    label_path = "/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/data/kaggle-competition-2/train_results.csv"
    output_folder = "/Users/yujunzhong/Documents/study/UdeM_Mila/courses/IFT6390/competition/comp2/data/kaggle-competition-2/new"
    gen_train_val(data_path, label_path, output_folder)
