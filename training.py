import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv import GINConvNet
from utils import *
from create_data import create


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)

        optimizer.zero_grad()
        output,loss_con = model(data_mol,data_pro)
        loss = loss_fn(output, ((data_mol.y + data_pro.y)/2).view(-1, 1).float().to(device))+loss_con
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data_mol.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output ,losslala = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, ((data_mol.y + data_pro.y)/2).view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

# device=os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
datasets = [['kiba','davis'][int(sys.argv[1])]]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
modeling = GINConvNet
# modeling = torch.nn.DistributedDataParallel(modeling)
model_st = modeling.__name__
# modeling = nn.DataParallel(modeling)
# modeling = modeling.cuda()

# device_ids = [0, 1]
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

# modeling=torch.nn.DataParallel(modeling,device_ids=device_ids)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    # processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    # processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    # if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    #     print('please run create_data.py to prepare data in pytorch format!')
    # else:
    if(True):
        x_data = []
        y_data = []
        train_data,test_data = create(int(sys.argv[1]))
        # train_data = TestbedDataset(root= 'data', dataset=dataset+'_train')
        # test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        # model = modeling().cuda()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),rm2(G,P)]
            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-2]
                best_rm2 = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse,best_ci,best_rm2,model_st,dataset)
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse,best_ci,best_rm2,model_st,dataset)

            plt.figure(figsize=(10, 6))
            y_data.append(ret[1])
            x_data.append(epoch)
            plt.plot(x_data, y_data, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注

            plt.legend()  # 显示上面的label
            plt.xlabel('time')  # x_label
            plt.ylabel('number')  # y_label

            plt.savefig('result_'+dataset +'2quan.jpg',  dpi=450)