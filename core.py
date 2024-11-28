import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
import pickle
import re


def build_vocab(texts, max_word_num=50000, path="./vocab.pkl"):
    vocab = {}
    # 构建词表
    for text in texts:
        for word in text:
            vocab[word] = vocab.get(word, 0) + 1
    # 进行排序
    vocab_items = (list(vocab.items()))
    vocab_items.sort(key=lambda x: x[1], reverse=True)
    vocab_items = vocab_items[:max_word_num]
    # 设置特殊词
    vocab_items = [('<PAD>', 0), ('<UNK>', 1)] + vocab_items
    # 将词表转换为字典
    vocab = {word: index for index, (word, _) in enumerate(vocab_items)}
    with open(path, "wb") as f:
        pickle.dump(vocab, f)
    return vocab


def data_clean(data_path, stop_words_path):
    """
    从指定位置加载数据
    :param path:
    :return:
    """
    # 加载停用词列表
    with open(stop_words_path, "r") as f:
        stop_words = set(f.readlines())

    def sentence2words(sentence, stop_words):
        # 使用链式操作处理句子，并通过列表推导式去除停用词
        return [word for word in re.sub(r'[^\w\s]', '', sentence.strip().lower()).split() if word not in stop_words]

    df = pd.read_csv(data_path, sep=";")
    df = df.loc[:, ["title", "label"]]
    df['title'] = df['title'].apply(lambda x: sentence2words(x, stop_words))
    return df['title'].to_numpy(), df['label'].to_numpy()


def load_vocab(path="./vocab.pkl"):
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


class DataSet(Dataset):
    def __init__(self, texts, labels, vocab, max_len=20):
        """
        :param data: 需要加载的数据
        :param vocabulary: 词表
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # 将文本转化为索引序列
        text = self.texts[index]  # 这里只是访问文本，尚未处理
        label = self.labels[index]  # 获取标签
        # 处理文本：将其转换为索引并填充或截断
        indexed_text = [self.vocab.get(word, self.vocab.get('<UNK>')) for word in text.split()]
        indexed_text = indexed_text[:self.max_len]  # 截断
        indexed_text += [self.vocab.get('<PAD>')] * (self.max_len - len(indexed_text))  # 填充
        # 将文本和标签转换为张量
        indexed_text = torch.tensor(indexed_text)
        label = torch.tensor(label)
        return indexed_text, label


if __name__ == '__main__':
    texts, labels = data_clean("./data/train.csv", "./en_stopwords.txt")
    vocab = load_vocab("./vocab.pkl")
    data_set = DataSet(texts, labels, vocab=vocab)
