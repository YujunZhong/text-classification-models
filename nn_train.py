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


def build_vocab(labels, texts, tokenizer):
    """ Build vocabulary
    :param labels: text labels
    :param texts: test data
    :param tokenizer: tokenizer to use
    :return: generated vocabulary
    """
    target_iter = iter(labels)
    text_iter = iter(texts)
    data_iter = zip(target_iter, text_iter)
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    torch.save(vocab, 'vocab.pth')

    return vocab


def train(epoch, dataloader, model, optimizer, criterion, exp_name):
    """ Main function for training.
    :param epoch: total number of epochs
    :param dataloader: training dataloader
    :param model: model to train
    :param optimizer: defined optimizer
    :param criterion: loss function
    :param exp_name: experiment name
    :return:
    """
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                            total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
    
    save_dir = f'./save/models/{exp_name}'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    torch.save(model, f'{save_dir}/epoch_{epoch}.pth')


def evaluate(dataloader, model, criterion):
    """ Main function for evaluation.
    :param model: model to evaluate
    :param criterion: loss function
    :param exp_name: experiment name
    :return: model accuracy
    """
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            outputs = model(text, offsets)
            loss = criterion(outputs, label)
            total_acc += (outputs.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def main(exp_name, data_path, label_path, vocab_path):
    # load data
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    texts_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(label_path)
    texts_df['label'] = labels_df.apply(lambda row: label_dict[row['target']], axis=1)
    texts_df_new = text_preprocessing(texts_df)

    # prepare data processing pipelines
    tokenizer = get_tokenizer('basic_english')
    if os.path.isfile(vocab_path):
        vocab = torch.load(vocab_path)
    else:
        vocab = build_vocab(texts_df.label, texts_df_new.new_text, tokenizer)

    print(vocab(['here', 'is', 'an', 'example']))    # [475, 21, 30, 5297]

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    # train and evaluate
    num_class = 3
    vocab_size = len(vocab)
    emsize = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    # Hyperparameters
    EPOCHS = 20 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    total_accu = None

    target_iter = iter(texts_df.label)
    text_iter = iter(texts_df_new.new_text)
    train_iter = zip(target_iter, text_iter)
    train_dataset = to_map_style_dataset(train_iter)

    num_train = int(len(train_dataset) * 0.9)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

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

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(epoch, train_dataloader, model, optimizer, criterion, exp_name)
        accu_val = evaluate(valid_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)


if __name__ == "__main__":
    exp_name = "nn_model_1208"
    data_path = "../data/kaggle-competition-2/train_data.csv" # 0.796 (val) if 3 classes, 0.794 if 2 classes
    label_path = "../data/kaggle-competition-2/train_results.csv"
    vocab_path = "./save/vocab.pth"

    main(exp_name, data_path, label_path, vocab_path)