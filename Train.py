import argparse
import os
import json
import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
from datetime import datetime

from model.net_cell import MSTF_NAS_NETWORK
from DataSets import *


def Train(train_loader, model, criterion, optimizer, device, new_iter, num_class, Args):
    model.train()
    Loss_list = []
    Acc_list  = [[],[]]
    now_iter  = 0
    for (Inputs, Labels) in train_loader:
        Labels = Labels.view(-1)
        Inputs = Inputs.to(device)
        Labels = Labels.to(device)
        optimizer.zero_grad()
        Logits = model(Inputs)
        Loss = criterion(Logits, Labels)
        Loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        
        # 计算topk的准确率，此处topk取值为(1,k), 其中k=num_class//2
        Batch_Size = Inputs.size(0)
        topk = (1, num_class//2)
        _, Pred = torch.topk(input=Logits.data, k=max(topk), dim=1, largest=True, sorted=True)
        Pred = Pred.T   # Pred.size() = [max(topk), batch_size]
        Correct = torch.eq(Pred, Labels.data.view(1,-1))
        for index, k in enumerate(topk):
            Correct_k = Correct[0:k].contiguous().view(-1).float().sum()
            Acc_list[index].append(Correct_k.mul(100.0/Batch_Size).item())
        Loss_list.append(Loss.item())
        
        now_iter = now_iter + 1
        if now_iter >= new_iter: break

    # 返回平均损失、平均Top-1准确率、平均Top-k准确率
    return [sum(Loss_list)/len(Loss_list), sum(Acc_list[0])/len(Acc_list[0]), sum(Acc_list[1])/len(Acc_list[1])]

def Test(test_loader, model, criterion, device, num_class, Args):
    model.eval()
    Loss_list = []
    Acc_list  = [[],[]]
    with torch.no_grad():
        for (Inputs, Labels) in test_loader:
            Labels = Labels.view(-1)
            Inputs = Inputs.to(device)
            Labels = Labels.to(device)
            Logits = model(Inputs)
            Loss = criterion(Logits, Labels)
            
            # 计算topk的准确率，此处topk取值为(1,k), 其中k=num_class//2
            Batch_Size = Inputs.size(0)
            topk = (1, num_class//2)
            _, Pred = torch.topk(input=Logits.data, k=max(topk), dim=1, largest=True, sorted=True)
            Pred = Pred.T   # Pred.size() = [max(topk), batch_size]
            Correct = torch.eq(Pred, Labels.data.view(1,-1))
            for index, k in enumerate(topk):
                Correct_k = Correct[0:k].contiguous().view(-1).float().sum()
                Acc_list[index].append(Correct_k.mul(100.0/Batch_Size).item())
            Loss_list.append(Loss.item())

    # 返回平均损失、平均Top-1准确率、平均Top-5准确率
    return [sum(Loss_list)/len(Loss_list), sum(Acc_list[0])/len(Acc_list[0]), sum(Acc_list[1])/len(Acc_list[1])]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', default=102400000, type=int)   # 此处控制使用训练集的大小，如果想要只使用部分训练集，达到缓解过拟合的方法，请自己调节
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--now_times', default=0, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    args = parser.parse_args()
    return args                                                                 

if __name__ == '__main__':
    # 输入运行参数
    Args      = parse_arguments()
    Args.data = "PathMNIST"
    Args.type = "None"

    # 验证cuda是否可用
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = torch.backends.cudnn.enabled
    print(f'Is cuda  available?\t{torch.cuda.is_available()}')
    print(f'Is cudnn available?\t{torch.backends.cudnn.enabled}')

    print(f'Data Set is {Args.data}')

    # 加载MedMNIST
    Train_Data, Test_Data, num_class = DataSet[Args.data](size=28)
    Train_Loader  = torch.utils.data.DataLoader(dataset=Train_Data, batch_size=Args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    Test_Loader   = torch.utils.data.DataLoader(dataset=Test_Data, batch_size=512, shuffle=False, drop_last=True, pin_memory=True, num_workers=8)
    Args.new_iter = min(Args.data_num // Args.batch_size, len(Train_Loader))

    # Log Config
    Log_File = f"/home/username/MSTF-NAS/log/{'_'.join(Args.type)}_{Args.now_times}.json"
    if not os.path.exists(Log_File):
        os.makedirs(os.path.dirname(Log_File), exist_ok=True)
    Log_Data = {
        'Train':['float',[]],
        'Test':['float',[]],
    }

    # Training Config
    N_N = 5     # Normel_Cell中节点个数
    N_R = 4     # Reduct_Cell中节点个数
    N_B = 3     # Block的个数
    Genotype     =  [ 4,0,2,1,4,2,5,0,5,0,4,1,5,4,3,0,6,1,3,2,0,1,5,1,3,0,0,2,7,0,7,1,7,2,2,2,2,3,0,4,2,5 ]
    # Genotype  = [2,0, 4,1, 3,0, 2,0, 2,4,  0,5,5,2] + [7,0, 2,1, 7,0, 2,3, 1,4,  5,2,5,5] + [4,0, 5,0, 4,2, 3,3, 1,0,  4,1,2,4] # Author's Genotype

    Network   = MSTF_NAS_NETWORK(num_class=num_class, genotype=Genotype, n_n=N_N, n_r=N_R, num_block=N_B).to(device)
    Network   = nn.DataParallel(Network).to(device)
    Optimizer = torch.optim.SGD(params=Network.parameters(), lr=Args.lr, momentum=0.9, weight_decay=0)
    # Optimizer = torch.optim.Adam(params=Network.parameters(), lr=Args.lr, weight_decay=3e-4)

    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=Optimizer, T_max=Args.max_epoch)
    # Scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=Optimizer, gamma=0.9)  # PneumoniaMNIST用

    Criterion = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(Args.max_epoch):
        Train_Loss, Train_Acc_1, Train_Acc_5 = Train(train_loader=Train_Loader, model=Network, criterion=Criterion, optimizer=Optimizer, device=device, new_iter=Args.new_iter, num_class=num_class, Args=Args)
        Test_Loss, Test_Acc_1, Test_Acc_5 = Test(test_loader=Test_Loader, model=Network, criterion=Criterion, device=device, num_class=num_class, Args=Args)
        Scheduler.step()
        
        print(epoch, '_'.join(Args.type), datetime.now().strftime('%Y-%m-%d %H:%M:%S'),)
        print(f"{Train_Loss}, {Train_Acc_1}, {Train_Acc_5}")
        print(f"{Test_Loss}, {Test_Acc_1}, {Test_Acc_5}")
        print()

        Log_Data['Train'][1].append([Train_Loss,Train_Acc_1,Train_Acc_5])
        Log_Data['Test'][1].append([Test_Loss,Test_Acc_1,Test_Acc_5])
    
    with open(Log_File, 'w') as Log:
        json.dump(Log_Data, Log, indent=4)

