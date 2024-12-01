import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


def build_vocab(texts, max_word_num=50000, path="./vocab.pkl"):
    """
    构建词表
    :param texts:
    :param max_word_num:
    :param path:
    :return:
    """
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
    """
    从指定位置加载词表
    :param path:
    :return:
    """
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def build_label(labels):
    """
    将标签0-1编码化
    :return:
    """
    new_labels = [None] * len(labels)
    for i, label in enumerate(labels):
        if label == 0:
            new_labels[i] = np.array([1, 0])
        else:
            new_labels[i] = np.array([0, 1])
    return np.array(new_labels, dtype=np.float32)


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
        words = self.texts[index]  # 这里只是访问文本，尚未处理
        label = self.labels[index]  # 获取标签
        # 处理文本：将其转换为索引并填充或截断
        indexed_text = [self.vocab.get(word, self.vocab.get('<UNK>')) for word in words]
        indexed_text = indexed_text[:self.max_len]  # 截断
        indexed_text += [self.vocab.get('<PAD>')] * (self.max_len - len(indexed_text))  # 填充
        # 将文本和标签转换为张量
        indexed_text = torch.tensor(indexed_text)
        label = torch.tensor(label)
        return indexed_text, label


def load_data(data_path, stop_words_path, vocab_path, batch_size, max_len=20):
    """
    加载数据
    :param data_path:
    :param stop_words_path:
    :param vocab_path:
    :param max_len:
    :return:
    """
    texts, labels = data_clean(data_path, stop_words_path)
    labels = build_label(labels)
    vocab = load_vocab(vocab_path)
    data_set = DataSet(texts, labels, vocab=vocab, max_len=max_len)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             prefetch_factor=5, persistent_workers=True)
    return data_loader


def train(model, data_loader, max_epoch, lr, device, save_path="./model", log_interval=10):
    # 将模型设置为训练模式
    model = model.to(device)
    model.train()
    # 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss().to(device)
    # 训练过程的信息
    print("Start training...")
    print("Max epoch:{}".format(max_epoch))
    print("Learning rate:{}".format(lr))
    print("Batch size:{}".format(data_loader.batch_size))
    print("Device:{}".format(device))
    print("Save path:{}".format(save_path))
    # 开始训练
    loss_record = []
    acc_record = []
    for i in range(1, max_epoch + 1):
        loss_total = 0
        acc_total = 0
        num_total = 0
        for batch_idx, (texts, labels) in enumerate(data_loader):
            # 将数据移动到GPU上
            texts, labels = texts.to(device), labels.to(device)
            # 前向传播
            label_predict = model(texts)
            # 计算损失
            loss_value = loss(label_predict, labels)
            # 计算正确率
            acc_num = (label_predict.argmax(dim=1, keepdim=True) == labels.argmax(dim=1, keepdim=True)).sum().item()
            acc_total += acc_num
            num_total += len(labels)
            # 反向传播
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_total += loss_value.item()
        # 记录损失和准确率
        acc_rate = acc_total / num_total
        loss_record.append(loss_total)
        acc_record.append(acc_rate)
        print(f'Epoch {i}, Total Loss: {loss_total:.4f}, Accuracy: {acc_rate * 100:.2f}%')
    # 判断存储位置是否存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    return loss_record, acc_record

def model_eval(model, data_loader, device):
    # 将模型设置为评估模式
    model.eval()
    model.to(device)
    # 测试过程的信息
    print("Start testing...")
    print("Batch size:{}".format(data_loader.batch_size))
    print("Device:{}".format(device))
    # 开始测试
    acc_total = 0
    num_total = 0
    predict = []
    with torch.no_grad():
        for texts, labels in data_loader:
            # 将数据移动到GPU上
            texts, labels = texts.to(device), labels.to(device)
            # 前向传播
            label_predict = model(texts)
            # 计算正确率
            acc_num = (label_predict.argmax(dim=1, keepdim=True) == labels.argmax(dim=1, keepdim=True)).sum().item()
            acc_total += acc_num
            num_total += len(labels)
            # 记录预测出的标签
            label_predict = label_predict.argmax(dim=1).cpu().numpy().tolist()
            predict.extend(label_predict)
    # 计算准确率
    acc_rate = acc_total / num_total
    print("Test acc:{}".format(acc_rate))
    return predict


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, length):
        super(TextCNN, self).__init__()
        # 定义超参数
        self.embedding_dim = embedding_dim
        self.kernel_size = [3, 4, 5]
        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 定义卷积层
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=length, kernel_size=(i, embedding_dim)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(length - i + 1, 1))
            ) for i in self.kernel_size
        ])
        # 定义全连接层
        self.fc = nn.Linear(length * 3, 2)

    def forward(self, x):
        # 增加数据的一个维度（模仿图片中的通道）
        x = x.unsqueeze(1)
        # 将嵌入层应用于输入数据
        x = self.embedding(x)
        # 将卷积层应用于输入数据
        x = [conv(x).squeeze(3) for conv in self.conv]
        # 将卷积层的输出进行拼接
        x = torch.cat(x, dim=1)
        # 将全连接层应用于输入数据
        x = x.squeeze(2)
        x = self.fc(x)
        return x


def plot_train_info(loss, acc):
    """
    绘制模型训练过程中的详细信息
    :param loss:
    :param acc:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 创建图形和第一个y轴
    fig, ax1 = plt.subplots()
    x = list(range(len(loss)))
    # 在第一个y轴上绘制数据
    ax1.plot(x, loss, 'g-', label='损失')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 在第二个y轴上绘制数据
    ax2.plot(x, acc, 'b-', label='正确率')
    ax2.set_ylabel('正确率', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # 显示图形
    plt.title('训练信息')
    plt.show()


if __name__ == '__main__':
    texts, labels = data_clean("./data/train.csv", "./en_stopwords.txt")
    labels = build_label(labels)
    vocab = load_vocab("./vocab.pkl")
    data_set = DataSet(texts, labels, vocab=vocab)
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True, num_workers=8)
    net = TextCNN(vocab_size=len(vocab), embedding_dim=100, length=20)
