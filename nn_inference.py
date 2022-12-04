import pandas as pd
import time
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from dataset import text_preprocessing
from model import TextClassificationModel


def test(dataloader, model, device):
    model.eval()

    pred_list = []
    all_outputs = torch.Tensor([], device=device)
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            outputs = model(text, offsets)
            all_outputs = torch.cat((all_outputs, outputs), 0)
            # pred_list.append(predicted_label.argmax(1))
    
    # pred_labels = torch.stack(pred_list, 0)

    return all_outputs


def main(exp_name, test_data_path, vocab_path, model_path):
    # prepare data processing pipelines
    tokenizer = get_tokenizer('basic_english')
    vocab = torch.load(vocab_path)

    print(vocab(['here', 'is', 'an', 'example']))    # [475, 21, 30, 5297]

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    BATCH_SIZE = 64

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    test_df = pd.read_csv(test_data_path)
    test_df['label'] = 0
    test_df_new = text_preprocessing(test_df)

    test_label_iter = iter(test_df.label)
    test_text_iter = iter(test_df_new.new_text)
    test_iter = zip(test_label_iter, test_text_iter)
    test_dataset = to_map_style_dataset(test_iter)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_batch)
    all_outputs = test(test_dataloader, model, device)
    pred_classes = torch.argmax(all_outputs, 1).numpy()


    df = pd.DataFrame({'target': pred_classes})
    df['id'] = df.index
    df.to_csv(f'./save/outputs/{exp_name}_results.csv', columns=['id', 'target'], index=False)


if __name__ == "__main__":
    exp_name = "nn_model_1203"
    test_data_path = "../data/kaggle-competition-2/test_data.csv"
    vocab_path = "./save/vocab.pth"
    model_path = f"./save/models/{exp_name}/epoch_6.pth"

    main(exp_name, test_data_path, vocab_path, model_path)