
import pandas as pd
import re

from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 加载停用词列表
    with open("./en_stopwords.txt", "r") as f:
        stop_words = set(f.readlines())


    def sentence2words(sentence, stop_words):
        # 使用链式操作处理句子，并通过列表推导式去除停用词
        return [word for word in re.sub(r'[^\w\s]', '', sentence.strip().lower()).split() if word not in stop_words]
    df = pd.read_csv("./data/train.csv", sep=";")
    df = df.loc[:, ["title", "label"]]
    df['title'] = df['title'].apply(lambda x: sentence2words(x, stop_words))
    df['cnt'] = df['title'].apply(lambda x: len(x))
    # 查看句子长度的分布情况
    df['cnt'].plot(kind='hist', bins=6, edgecolor='black', alpha=0.7)
    plt.savefig("./data/length.png")