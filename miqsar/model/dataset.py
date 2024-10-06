from typing import Sequence, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader,Dataset
def scale_data(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = np.array(x_train)
    x_test_scaled = np.array(x_test)
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)
    return np.array(x_train_scaled), np.array(x_test_scaled)
class MolDataSet(Dataset):
    def __init__(self,bags,labels) -> None:
        bags,mask = self.add_padding(bags)
        # bags: Nmol*Nconf*Ndesc 训练数据
        self.bags = torch.from_numpy(bags)
        # mask: Nmol*Nconf*1 标记哪些构象是有效的，在训练过程中去除噪点
        self.mask = torch.from_numpy(mask)
        # labels: Nmol
        labels = labels.reshape(-1,1)
        if isinstance(labels,torch.Tensor):
            self.labels = labels 
        else:
            self.labels = torch.from_numpy(labels)
                   
    def __len__(self):
        return self.bags.shape[0]
    def __getitem__(self, index):
        return (self.bags[index],self.mask[index]),self.labels[index]
    def add_padding(self, x: Union[Sequence, np.array]) -> Tuple[np.array,np.array]:
        """
        Adds zero-padding to each  bag in x (sequence of bags) to bring x to tensor of shape Nmol*max(Nconf)*Ndescr,
        where: Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string, Nmol - number of molecules  in dataset
        Examples
        --------
        >>> import numpy as np
        >>> from miqsar.estimators.base_nets import BaseNet
        >>> net = BaseNet()
        >>> x_train = [[[1, 1],[1,1]],[[1, 1]]] # 2 molecules, one with 2 conformers and the other with only 1 conformer
        >>> _, m = net.add_padding(x_train)
        >>> m
        array([[[1.],
                [1.]],
        <BLANKLINE>
               [[1.],
                [0.]]])

        Parameters
         -----------
         x:  Union[Sequence, np.array]
         Sequence of bags (sets of conformers, one set for each molecule)
         Returns
         -----------
         Tuple of 2 tensors: new padded  tensor x and   mask tensor m (shape of m: Nmol*max(Nconf)*1): each row populated with
          either 1 where conformer exists, or 0 where conformer didnt exist and zeros were added.
            """
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
    def preprocess(self):
        x_train, x_test, y_train, y_test = train_test_split(self.bags, self.labels, test_size=0.1,random_state=45)
        x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
        x_train_scaled, x_val_scaled, y_train, y_val = train_test_split(x_train_scaled, y_train, test_size=0.1,random_state=43)
        return MolDataSet(x_train_scaled,y_train),MolDataSet(x_val_scaled,y_val),MolDataSet(x_test_scaled,y_test)