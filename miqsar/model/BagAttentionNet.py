import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from typing import Sequence, Tuple

class MainNet:
    """
    Abstract class not intended to be invoked directly.
    Main net for multiple-instance learning via convolution on conformers. (conformation ensembles are hereafter termed "bags".)
    Linear transform maps each conformer to new hidden space of dimensionality hd (hd1,hd2,hd3..) This is perfomed seqentially ndim-1 times,
    with RelU nonlinearity after each linear layer (linear layer can be viewed as a set
    of 1-dimensional convolution filters). The result is learnt representation of bags of shape Nmols*Nconf*Nhlast, where
    Nmols - number of molecules, Nconf -number of conformers and Nhlast - dimensionality of last hidden layer.

    This learnt representation (as opposed to original one) is assumed to help better predict property studied.
    THe same net can be used for single instance learning, in which case it essentially is MLP.
    """

    def __new__(cls, ndim: Sequence):
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        Returns
        --------
        MainNet instance
        """
        ind, hd1, hd2, hd3 = ndim
        net = Sequential(Linear(ind, hd1),
                         ReLU(),
                         Linear(hd1, hd2),
                         ReLU(),
                         Linear(hd2, hd3),
                         ReLU())

        return net
    
class Detector(nn.Module):
    def __new__(cls, input_dim:int, det_dim: Sequence):
        input_dim = input_dim
        attention = []
        for dim in det_dim[1:]:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        net = Sequential(*attention)
        return net
    
class Estimator(nn.Module):
    def __new__(cls, input_dim:int):
        net = Linear(input_dim, 1)
        return net


    
class BagAttentionNet(nn.Module):
    def __init__(self, ndim: Sequence, det_ndim: Sequence, instance_dropout:int=0.05):
        """
        Parameters
        ----------
        ndim: Sequence
        Hyperparameter for MainNet: each entry of sequence specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        det_ndim: Sequence
        Hyperparameter for attention subnet: each entry of sequence specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        init_cuda: bool, default is False
        Use Cuda GPU or not?

        """
        super().__init__()
        input_dim = ndim[-1]
        self.main_net = MainNet(ndim)
        self.estimator = Estimator(input_dim)
        self.detector = Detector(input_dim, det_ndim)
        self.instance_dropout = instance_dropout


    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Feed forward input data.
        Parameters
        ----------
        x: torch.Tensor
        m: torch.Tensor
        Returns
        --------
        Tuple with weights of conformers and tensor
        of shape Nmol*1, where Nmol is the number of molecules. The tensor is final output y, but it needs to be passed
        to sigmoid to obtain final class probabilities in case of classification (recall, this classs shouldnt be called directly,
        call regressor/classifier subclass to obtain final y).

        Examples
        --------
        >> > import torch
        >> > import numpy as np
        >> > from torch import randn
        >> > from miqsar.estimators.attention_nets import AttentionNet
        >> > x_train = randn((3, 3, 3))
        >> > at_net = AttentionNet(ndim=(x_train[0].shape[-1], 4, 6, 4), det_ndim = (4,4), init_cuda=False)
        >> > _, m = at_net.add_padding(x_train)
        >> > m = torch.from_numpy(m.astype('float32'))
        >> > _ = at_net.forward(x_train, m)  # (assign result to a variable to supress std output)

        """
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = nn.functional.gumbel_softmax(x_det, tau=self.instance_dropout, dim=2)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        out = out.view(-1, 1)
        return w, out
    
if __name__ == "__main__":
    from torch import randn
    import numpy as np
    x_train = randn((7,2,3))
    model = BagAttentionNet(ndim=(x_train[0].shape[-1], 4, 6, 4), det_ndim = (4,4))
    m = torch.ones((7,2,1))
    w,o = model(x_train,m)
    w = w.view(w.shape[0], w.shape[-1]).cpu()
    w = [i[j.bool().flatten()].detach().numpy() for i, j in zip(w, m)]
    print(w)
    #print(w.shape)
    pass
