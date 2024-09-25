import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import Tensor, nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh, ReLU
from torch.nn.functional import softmax
from miqsar.estimators.base_nets import BaseRegressor
from typing import List, Optional, Sequence, Tuple, Union
import torch_optimizer as optim

from miqsar.estimators.utils import get_mini_batches, train_val_split

class MainNet:

    def __new__(cls, ndim: Sequence):

        ind, hd1, hd2, hd3 = ndim
        net = Sequential(Linear(ind, hd1),
                         ReLU(),
                         Linear(hd1, hd2),
                         ReLU(),
                         Linear(hd2, hd3),
                         ReLU())

        return net
class BagAttention(nn.Module):

    def __init__(self, ndim: Sequence, det_ndim: Sequence, init_cuda: bool = False):

        super().__init__()
        self.init_cuda = init_cuda
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)
        #
        input_dim = ndim[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def name(self):
        return self.__class__.__name__
    
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:

        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = nn.functional.gumbel_softmax(x_det, tau=self.instance_dropout, dim=2)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        out = out.view(-1, 1)
        return w, out

    def add_padding(self, x: Union[Sequence, np.array]) -> Tuple[np.array,np.array]:

        bag_size = max(len(i) for i in x)
        mask = np.ones((len(x), bag_size, 1))

        out = []
        for i, bag in enumerate(x):
            bag = np.asarray(bag)
            if len(bag) < bag_size:
                mask[i][len(bag):] = 0
                padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
                bag = np.vstack((bag, padding))
            out.append(bag)
        out_bags = np.asarray(out)
        return out_bags, mask

    def loss_batch(self, x_mb: Tensor, y_mb: Tensor, m_mb: Tensor, optimizer: Optional[torch.optim.Optimizer]=None) -> float:

        w_out, y_out = self.forward(x_mb, m_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def fit(self, x: Union[Sequence[Union[Sequence, np.array]], np.array], y: Union[Sequence, np.array],
            n_epoch: int = 100, batch_size: int = 128, lr: float = 0.001,
            weight_decay: float = 0, instance_dropout: float = 0.95, verbose: bool = False)-> 'BagAttention':

        self.instance_dropout = instance_dropout
        x, m = self.add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_val_split(x, y, m) 
        print(f'train: {len(x_train)},val: {len(x_val)}')
        if y_train.ndim == 1: # convert 1d array into 2d ("column-vector")
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:  # convert 1d array into 2d ("column-vector")
            y_val = y_val.reshape(-1, 1)
        if self.init_cuda:
            x_train, x_val, y_train, y_val, m_train, m_val  = x_train.cuda(), x_val.cuda(), y_train.cuda(), y_val.cuda(), \
                                                              m_train.cuda(), m_val.cuda()
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            mb = get_mini_batches(x_train, y_train, m_train, batch_size=batch_size)
            self.train()
            for x_mb, y_mb, m_mb in mb:
                loss = self.loss_batch(x_mb, y_mb, m_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(x_val, y_val, m_val, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)

        y_val_pred = self.predict(x_val.cpu())
        print('3D/MI/AttentionNet: r2_score val = {:.2f}'.format(r2_score(y_val.cpu(), y_val_pred)))
        return self

    def predict(self, x: Union[Sequence[Union[Sequence, np.array]], np.array]) -> np.array:
        
        x, m = self.add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        return np.asarray(y_pred.cpu())

    def get_instance_weights(self, x: Union[Sequence[Union[Sequence, np.array]], np.array]) -> List[np.array]:

        x, m = self.add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        w = w.view(w.shape[0], w.shape[-1]).cpu()
        w = [np.asarray(i[j.bool().flatten()]) for i, j in zip(w, m)]
        return w
