from core import *
import click
@click.command()
@click.option("--mode", help="train or test", type=str)
def main(mode):
    # 定义超参数
    max_epoch = 500
    batch_size = 1024
    lr = 0.0001
    vocab_size = len(load_vocab())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        # 数据预处理
        train_data = load_data('./data/train.csv', "./en_stopwords.txt", "./vocab.pkl", batch_size)
        # 训练模型
        net = TextCNN(vocab_size=vocab_size, embedding_dim=100, length=20)
        loss, acc = train(net, train_data, max_epoch=max_epoch, lr=lr, device=device)
        plot_train_info(loss, acc)
    elif mode == 'test':
        model = TextCNN(vocab_size=vocab_size, embedding_dim=100, length=20)
        model.load_state_dict(torch.load('./model/model.pth'))
        test_data = load_data('./data/test.csv', "./en_stopwords.txt", "./vocab.pkl", batch_size)
        model_eval(model, test_data, device=device)
    else:
        model = TextCNN(vocab_size=vocab_size, embedding_dim=100, length=20)
        model.load_state_dict(torch.load('./model/model.pth'))
        test_data = load_data('./data/evaluation.csv', "./en_stopwords.txt", "./vocab.pkl", batch_size)
        model_eval(model, test_data, device=device)
if __name__ == '__main__':
    main()