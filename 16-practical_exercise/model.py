import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):  # 배치노멀이 default
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            # 한 block 즉, 신경망 한층의 기본 set?개념
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            # linear Layer의 output_size가 batch_norm의 input_size
            get_regularizer(use_batch_norm, output_size),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)

        return y


class ImageClassifier(nn.Module):  # 이름 바꿔주자

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):  # 레이어 개수에 따른 hidden_sizes(중간 레이어의 사이즈, 위의 값은 default)만 잘 설정하면 다른 문제에도 적용 가능
        # hidden_sizes의 default값은 내가 하고자하는 분석의 목적에 맞게 작성
        # hidde_sizes는 중간 신경망의 node수라고 생각

        super().__init__()

        # len(hidden_sizes) > 0가 False면 AssertError를 발생 시킴
        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size  # 최초 input_size는 raw data의 크기대로(column수)
        blocks = []
        for hidden_size in hidden_sizes:  # 여기서 hidden size의 len대로 block을 쌓아줌
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size) 즉 mini_batch내의 각 sample의 class별 log확률값

        return y
