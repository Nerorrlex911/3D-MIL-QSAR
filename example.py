import os
from miqsar.utils import calc_3d_pmapper
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from miqsar.estimators.wrappers import InstanceWrapperMLPRegressor
from miqsar.estimators.attention_nets import AttentionNetRegressor
from miqsar.estimators.utils import set_seed

def process_data(file_name:str='alltrain',max_conf:int=50):
    #Choose dataset to be modeled and create a folder where the descriptors will be stored

    nconfs_list = [1,max_conf] #number of conformations to generate; calculation is time consuming, so here we set 5, for real tasks set 25..100
    ncpu = 50 # set number of CPU cores 

    dataset_file = os.path.join('datasets', f'{file_name}.smi')
    descriptors_folder = os.path.join('descriptors')
    # os.mkdir(descriptors_folder)

    out_fname = calc_3d_pmapper(input_fname=dataset_file, nconfs_list=nconfs_list, energy=100,  descr_num=[4],path=descriptors_folder, ncpu=ncpu)

def load_data(file_name:str='alltrain',nconf:int=20):
    def load_svm_data(fname):
        
        def str_to_vec(dsc_str, dsc_num):

            tmp = {}
            for i in dsc_str.split(' '):
                tmp[int(i.split(':')[0])] = int(i.split(':')[1])
            #
            tmp_sorted = {}
            for i in range(dsc_num):
                tmp_sorted[i] = tmp.get(i, 0)
            vec = list(tmp_sorted.values())

            return vec
        
        #
        with open(fname) as f:
            dsc_tmp = [i.strip() for i in f.readlines()]

        with open(fname.replace('txt', 'rownames')) as f:
            mol_names = [i.strip() for i in f.readlines()]
        #
        labels_tmp = [float(i.split(':')[1]) for i in mol_names]
        idx_tmp = [i.split(':')[0] for i in mol_names]
        dsc_num = max([max([int(j.split(':')[0]) for j in i.strip().split(' ')]) for i in dsc_tmp])
        #
        bags, labels, idx = [], [], []
        for mol_idx in list(np.unique(idx_tmp)):
            bag, labels_, idx_ = [], [], []
            for dsc_str, label, i in zip(dsc_tmp, labels_tmp, idx_tmp):
                if i == mol_idx:
                    bag.append(str_to_vec(dsc_str, dsc_num))
                    labels_.append(label)
                    idx_.append(i)
                    
            bags.append(np.array(bag).astype('uint8'))
            labels.append(labels_[0])
            idx.append(idx_[0])           

        return np.array(bags,dtype=object), np.array(labels), np.array(idx)

    # split data into a training and test set
    dsc_fname = os.path.join('descriptors', f'PhFprPmapper_conf-{file_name}_{nconf}.txt') # descriptors file
    bags, labels, idx = load_svm_data(dsc_fname)
    print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx)
    print(f'There are {len(x_train)} training molecules and {len(x_test)} test molecules')

    def scale_data(x_train, x_test):
        scaler = MinMaxScaler()
        scaler.fit(np.vstack(x_train))
        x_train_scaled = x_train.copy()
        x_test_scaled = x_test.copy()
        for i, bag in enumerate(x_train):
            x_train_scaled[i] = scaler.transform(bag)
        for i, bag in enumerate(x_test):
            x_test_scaled[i] = scaler.transform(bag)
        return np.array(x_train_scaled), np.array(x_test_scaled)


    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    return x_train_scaled,x_test_scaled, y_train, y_test, idx_train, idx_test
def train(x_train_scaled,x_test_scaled, y_train, y_test, idx_train, idx_test):
    
    # One should implement a protocol for optimizing the hyperparameters of the neural network. Here we assign the default hyperparameters found with the grid search technique.

    # In[ ]:

    set_seed(43)

    ndim = (x_train_scaled[0].shape[1], 256, 128, 64) # number of hidden layers and neurons in the main network
    det_dim = (64,64)
    pool = 'mean'                                     # type of pulling of instance descriptors
    n_epoch = 1000                                    # maximum number of learning epochs
    lr = 0.001                                      # learning rate
    weight_decay = 0.001                              # l2 regularization
    batch_size = 16                           # batch size
    init_cuda = True                                  # True if GPU is available


    net = AttentionNetRegressor(ndim=ndim, det_ndim=det_dim, init_cuda=init_cuda)
    net.fit(x_train_scaled, y_train, 
            n_epoch=n_epoch, 
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            verbose=True)
    return net

if __name__ == '__main__':
    file_name = 'alltrain'#'CHEMBL1075104'#'alltrain'
    max_conf = 5
    process_data(file_name,max_conf)
    x_train_scaled,x_test_scaled, y_train, y_test, idx_train, idx_test=load_data(file_name,max_conf)
    net=train(x_train_scaled,x_test_scaled, y_train, y_test, idx_train, idx_test)


    from sklearn.metrics import r2_score, mean_absolute_error

    y_pred = net.predict(x_test_scaled)

    print('3D/MI/AttentionNet: r2_score test = {:.2f}'.format(r2_score(y_test, y_pred)))

    y_train_pred = net.predict(x_train_scaled)

    print('3D/MI/AttentionNet: r2_score test = {:.2f}'.format(r2_score(y_train, y_train_pred)))




