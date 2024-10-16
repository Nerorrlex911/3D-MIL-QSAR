from datetime import datetime
import logging
import os
import sys

from sklearn.metrics import r2_score,mean_squared_error
import torch
import torch_optimizer as optim
from tqdm import tqdm
from miqsar.model.BagAttentionNet import BagAttentionNet
from miqsar.model.dataset import MolDataSet
from miqsar.model.earlystop import EarlyStopping
from miqsar.utils import calc_3d_pmapper
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from miqsar.estimators.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

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

    dataset = MolDataSet(bags,labels)

    return dataset
def train(
        dataset:MolDataSet,
        batch_size = 1,
        instance_dropout = 0.95,
        lr=0.01,
        gamma=0.95,
        step=10,
        weight_decay=0.001,
        earlystop=False,
        patience=30,
        epochs = 1000,
        save_path='train'
    ):

    set_seed(43)
    train_dataset,val_dataset,test_dataset = dataset.preprocess()
    logging.info(f'train: {len(train_dataset)},val: {len(val_dataset)},test: {len(test_dataset)}')
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    batch_amount = len(train_dataloader)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=True)

    # 初始化模型
    model = BagAttentionNet(ndim=(train_dataset[0][0][0].shape[1],256,128,64),det_ndim=(64,64),instance_dropout=instance_dropout)

    model = model.double()

    print(model)

    # 检查是否有 CUDA 设备可用
    if torch.cuda.is_available():
        # 将模型转移到 CUDA
        model = model.cuda()
        # 包装成 DataParallel 模型以支持多张 GPU
        model = torch.nn.DataParallel(model)

    # 创建 TensorBoard 的 SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(save_path,'tensorboard'))
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.Yogi(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=gamma, patience=step,)

    # 初始化用于保存loss的列表
    train_losses = []
    val_losses = []
    test_losses = []

    # 早停
    earlystopping = EarlyStopping(patience=patience,verbose=True)

    # 训练模型
    for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
        model.train()
        train_loss = 0
        train_progress = tqdm(enumerate(train_dataloader), desc="Batches", position=0, leave=True)
        for i,((bags,mask),labels) in train_progress:

            bags = bags.cuda()
            mask = mask.cuda()
            labels = labels.cuda()
            weight,outputs = model(bags,mask)
            loss = criterion(outputs, labels)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()  # 更新学习率

            if i % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
        # 记录损失和学习率到 TensorBoard
        # writer.add_scalar('Train Loss', loss.item(), i+(epoch+1)*batch_amount)
        # train_losses.append(train_loss/len(train_dataloader))
        # 验证模型
        model.eval()
        with torch.no_grad():

            def calc_loss(dataset, name):
                weights, outputs = model(dataset.bags.cuda(), dataset.mask.cuda())
                loss = criterion(outputs, dataset.labels.cuda())
                logging.info(f'Epoch [{epoch + 1}/{epochs}], {name} Loss: {loss.item():.4f}')
                writer.add_scalar(f'{name} Loss MSE',loss.item(),epoch)
                return loss.item()
            
            train_loss = calc_loss(train_dataset, 'Train')
            train_losses.append(train_loss)
            
            val_loss = calc_loss(val_dataset, 'Val')
            val_losses.append(val_loss)

            test_loss = calc_loss(test_dataset, 'Test')
            test_losses.append(test_loss)

        min_loss_idx = val_losses.index(min(val_losses))
        if min_loss_idx == epoch:
            best_parameters = model.state_dict()
            logging.fatal(f'loss decreased, epoch: {epoch},loss: {val_loss}')
        scheduler.step(val_loss)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if earlystop:
            earlystopping(val_loss,model)
            if earlystopping.early_stop:
                logging.info(f'Early stopping, loss: {str(val_loss)}')
                break
    model.load_state_dict(best_parameters, strict=True)
    loss_data = pd.DataFrame({'train_loss':train_losses,'val_loss':val_losses,'test_loss':test_losses})
    loss_data.to_csv(os.path.join(save_path,'loss.csv'),header=False,index=False)
    writer.close()
    model.eval()
    with torch.no_grad():
        def eval_model(dataloader, model, save_path, file_name):
            progress = tqdm(enumerate(dataloader), desc=file_name, position=0, leave=True)
            weights = []
            y_pred = []
            y_label = []
            for i, ((bags, mask), labels) in progress:
                bags = bags.cuda()
                mask = mask.cuda()
                labels = labels.cuda()
                weight, outputs = model(bags, mask)
                w = weight.view(weight.shape[0], weight.shape[-1]).cpu()
                w = [i[j.bool().flatten()].detach().numpy() for i, j in zip(w, mask.cpu())]
                weights.extend(w)
                y_pred.extend(outputs.cpu().detach().numpy())
                y_label.extend(labels.cpu().detach().numpy())
            # weights = np.array(weights)
            # weight_data = pd.DataFrame(weights)
            # weight_data.to_csv(os.path.join(save_path, f'{file_name}_weight.csv'))
            y_pred = np.array(y_pred)
            y_label = np.array(y_label)
            np.savetxt(os.path.join(save_path, f'{file_name}_pred.csv'), np.column_stack((y_label,y_pred)), delimiter=',')
            r2 = r2_score(y_label, y_pred)
            mse = mean_squared_error(y_label, y_pred)
            # 将R2 score保存到txt文件
            with open(os.path.join(save_path, f'{file_name}_r2_score.txt'), 'w') as f:
                f.write(f'R2 score {file_name}: {r2}\n')
                f.write(f'MSE Loss {file_name}: {mse}\n')
            logging.info(f'R2 score {file_name}:{r2}')
            logging.info(f'MSE Loss {file_name}:{mse}')

        # 使用新的函数来进行训练、测试和验证
        eval_model(train_dataloader, model, save_path, 'train')
        eval_model(test_dataloader, model, save_path, 'test')
        eval_model(val_dataloader, model, save_path, 'val')
    
    return model

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('debug.log')]  # 添加这一行
    )
    logging.info('------------start------------')
    np.random.seed(42)
    file_name = 'alltrain'#'CHEMBL1075104'#'alltrain'
    max_conf = 50
    #process_data(file_name,max_conf)
    dataset=load_data(file_name,max_conf)
    lr_list = [ 0.005,0.01,0.02,0.05 ]
    gamma_list = [ 0.1,0.2,0.3 ]
    step_list = [ 10,20,30 ]
    for lr in lr_list:
        for gamma in gamma_list:
            for step in step_list:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
                save_path = os.path.join('train', f'ReduceLROnPlateau_lr={str(lr)}_step={str(step)}_gamma={str(gamma)}_{current_time}')
                model = train(dataset, lr=lr, gamma=gamma, step=step, save_path=save_path,earlystop=True,patience=40,weight_decay=0.01)




