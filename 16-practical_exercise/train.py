import argparse  # 사용자의 입력을 configuration으로 받아오는 역할

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes


def define_argparser():
    p = argparse.ArgumentParser()  # ArgumentParser 사용자의 입력을 configuration으로 받아오는 역할

    p.add_argument('--model_fn', required=True)
    # pth 파일이 저장되는 위치, 최종 모델
    p.add_argument('--gpu_id', type=int,
                   default=0 if torch.cuda.is_available() else -1)
    # gpu사용가능?

    p.add_argument('--train_ratio', type=float, default=.8)
    # train, valid 나누는 비율

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    # layer개수, 자동으로 hidden size가 정해짐, 유틸에 있음 (등차수열)
    p.add_argument('--use_dropout', action='store_true')
    # dropout쓰실? default는 batch norm
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)
    # 수다스러움의 정도?

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(
        'cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)  # 얘는 커스텀
    x, y = split_data(x.to(device), y.to(device),
                      train_ratio=config.train_ratio)
    # 얘는 train data, vaild data 나누기

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])  # 맨 마지막 차원이 입력 차원이어야 한다.
    output_size = int(max(y[0])) + 1  # 이건 긍 중 부에 따라 바꿔줘야겠다 3으로

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    # 보통 이런식으로 저장함
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
