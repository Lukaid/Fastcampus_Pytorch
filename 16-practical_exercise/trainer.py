from copy import deepcopy
import numpy as np

import torch


class Trainer():

    # 학습에 필요한 것들을 가져와서 저장

    def __init__(self, model, optimizer, crit):  # 모델, 아마 아담, 목적에 맞는 loss func
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True):  # 미니배치 분할
        if random_split:  # validate에선 false
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):  # 작은 for문, epoch 안의 iteration
        self.model.train()  # 학습모드, 까먹지 말자

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):

            # |x_i| = (batch_size, input_size)
            # |y_i| = (batch_size, |class|), |class|는 out_put size
            # # of class 즉, 분류하고 싶은 아웃풋의 개수? 긍정, 중립, 부정 이면 3

            y_hat_i = self.model(x_i)  # feed_forward
            # y_hat_i은 확률분포 vector
            loss_i = self.crit(y_hat_i, y_i.squeeze())  # loss
            # y_i는 long type tensor, one hot의 index만 들어있는 vector

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()  # 그래디언트 초기화
            loss_i.backward()  # 로스 미분

            self.optimizer.step()  # 얜 adam이 알아서 할것, update

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" %
                      (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):  # 작은 for문, epoch 안의 iteration
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            # random_shuffling없이 쪼개기
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)  # feed_forward
                loss_i = self.crit(y_hat_i, y_i.squeeze())  # loss

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" %
                          (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):  # 전체 for문, epoch
        lowest_loss = np.inf  # default는 무한
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:  # valid loss가 낮아졌다면
                lowest_loss = valid_loss  # loss 저장 후
                # 해당 model도 저장, snapshot
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model. valid loss가 가장 낮은 모델을 저장
        self.model.load_state_dict(best_model)
