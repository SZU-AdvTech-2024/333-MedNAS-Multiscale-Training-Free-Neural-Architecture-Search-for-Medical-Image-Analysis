import torch
import torch.utils.data
import torch.nn as nn

import gc
import os
import random
import warnings
import argparse
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import IntegerRandomSampling 
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from DataSets import *
from Metrics.complexity import COMPLEXITY
from pruners import predictive
from model.operators import MSTF_NAS_SPACE_NORMAL, MSTF_NAS_SPACE_REDUCT
from model.net_cell import MSTF_NAS_NETWORK


max_values = None
min_values = None
ranges     = None
record     = {}   # Recording the evaluated neural architecture's objective value

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='OrganSMNIST')
    parser.add_argument('--batch_size', default=24, type=int, help="The batch size for training-free metric.")
    parser.add_argument('--repeat', default=20, type=int, help="The frequency of calculating the training-free metric.")
    args = parser.parse_args()
    return args

def network_weight_init(net: nn.Module):
    with torch.no_grad():
        for layer in net.modules():
            try: layer.reset_parameters()
            except AttributeError: pass
        return net

# Defining the optimization problem
class NAS(Problem):
    def __init__(self, n_var, xl, xu, repeat:int, net_info:dict):
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=np.int16)
        self.repeat     = repeat
        self.net_info   = net_info  # "Train_Data","Test_Data","num_class","batch_size","device","N_N","N_R","N_B"
        self.data_idx   = list(range(len(self.net_info["Train_Data"])))

    def load_data(self):
        gc.collect()
        torch.cuda.empty_cache()
        rand_idx   = random.sample(population=self.data_idx, k=self.net_info["batch_size"]*self.repeat)
        sampler    = torch.utils.data.SubsetRandomSampler(rand_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset=self.net_info["Train_Data"], batch_size=self.net_info["batch_size"], sampler=sampler,
            shuffle=False, drop_last=True, pin_memory=True, num_workers=4
        )
        return dataloader
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Tips: np.shape(x) = [pop_size, n_var]
        global epoch_counter
        print(f"Epoch: {epoch_counter}")
        epoch_counter += 1

        genotypes = x.tolist()
        objects   = np.zeros((len(genotypes), 3))
        for idx, geno in enumerate(genotypes):
            if tuple(geno) in record.keys():
                objects[idx,:] = record[tuple(geno)]   # 把list转换为tuple，才可以用字典来搜索
            else:
                Network   = MSTF_NAS_NETWORK(
                    num_class=self.net_info["num_class"], genotype=geno, n_n=self.net_info["N_N"], 
                    n_r=self.net_info["N_R"], num_block=self.net_info["N_B"]
                ).to(self.net_info["device"])

                # 1、calculate flops, macs and params
                flops, macs, params = COMPLEXITY(model=Network, input_shape=list(self.net_info["Train_Data"][0][0].shape))
                # 2、calculate jacob_cov and synflow
                Temp_Dict = predictive.find_measures(net_orig=Network, dataloader=self.load_data(),
                                                     dataload_info=('random', self.repeat, self.net_info["num_class"]),
                                                     device=self.net_info["device"],
                                                     measure_names=["jacob_cov", "synflow"])    # return Dict
                jacob_cov, synflow = Temp_Dict["jacob_cov"], Temp_Dict["synflow"]

                objects[idx,:] = [flops, jacob_cov, synflow]
                record[tuple(geno)] = objects[idx,:]
                del Network
                gc.collect()
                torch.cuda.empty_cache()
        
        # Fake objective
        global min_values, ranges
        if ranges is not None:
            objects    = np.where(ranges!=0, (objects-min_values)/ranges, 0)
        f1         = -1*objects[:,1]-objects[:,0]  # 第一个目标函数
        f2         = -1*objects[:,2]+objects[:,0]  # 第二个目标函数
        out["F"]   = np.column_stack([f1, f2])
        out["F"][np.isnan(out["F"])] = np.inf
        print(out["F"])

class NormalizeCallback(Callback):
    def __call__(self, algorithm):
        # Calculate real objective value
        # Get training-free metrics
        X          = algorithm.pop.get("X")  # 获取种群基因型
        genotypes  = X.tolist()
        objects    = np.zeros((len(genotypes), 3))
        for idx, geno in enumerate(genotypes):
            objects[idx,:] = record[tuple(geno)]   # 把list转换为tuple，才可以用字典来搜索
        
        # Normalize
        global max_values, min_values, ranges
        max_values = np.nanmax(objects, axis=0)
        min_values = np.nanmin(objects, axis=0)
        ranges     = max_values - min_values
        objects    = np.where(ranges!=0, (objects-min_values)/ranges, 0)
        f1         = -1*objects[:,1]-objects[:,0]  # 第一个目标函数
        f2         = -1*objects[:,2]+objects[:,0]  # 第二个目标函数
        objects    = np.column_stack([f1, f2])
        objects[np.isnan(objects)] = np.inf

        # Update objective
        algorithm.pop.set("F", objects)
        print(objects)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # 输入运行参数
    Args = parse_arguments()

    # 验证cuda是否可用
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    print(f'Is cuda  available?\t{torch.cuda.is_available()}')
    print(f'Is cudnn available?\t{torch.backends.cudnn.enabled}')

    # 加载数据集
    print(f'Data Set is {Args.dataset}')
    if Args.dataset not in DataSet.keys():
        ValueError("Unsupported 'Data_Type'")
    elif Args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST']:
        Train_Data, Test_Data, num_class = DataSet[Args.dataset]()
    else:   # MedMNIST
        Train_Data, Test_Data, num_class = DataSet[Args.dataset](size=28)


    # Network Config
    N_N          = 5     # Normel_Cell中节点个数
    N_R          = 4     # Reduct_Cell中节点个数
    N_B          = 3     # Block的个数
    Network_Info = {
        "Train_Data":Train_Data, "Test_Data":Test_Data, "num_class":num_class,
        "batch_size":Args.batch_size, "device":device, 
        "N_N":N_N, "N_R":N_R, "N_B":N_B
    }
    # Genotype Config
    n_var           = N_B * (N_N*2 + N_R) # 决策变量的个数
    Whole_LB        = np.zeros(n_var)
    Normal_UB       = np.ones(N_N*2) * len(MSTF_NAS_SPACE_NORMAL) - 1
    Normal_UB[1::2] = range(N_N)
    Reduct_UB       = np.ones(N_R) * len(MSTF_NAS_SPACE_REDUCT) - 1
    Whole_UB        = np.concatenate([Normal_UB, Reduct_UB], axis=0)
    Whole_UB        = np.tile(Whole_UB, N_B)

    # Optimizing Problem
    epoch_counter = 1
    NASProblem    = NAS(n_var=n_var, xl=Whole_LB, xu=Whole_UB, repeat=Args.repeat, net_info=Network_Info)

    # Algorithm     = NSGA2(pop_size=50,
    #     sampling=IntegerRandomSampling(),
    #     crossover=SimulatedBinaryCrossover(prob_exch=0.3, prob_bin=0.2, prob_var=0.01),
    #     mutation=PolynomialMutation(repair=RoundingRepair())
    # )

    ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
    Algorithm     = MOEAD(ref_dirs=ref_dirs, n_neighbors=20, callback=NormalizeCallback(),
        sampling=IntegerRandomSampling(),
        crossover=SimulatedBinaryCrossover(prob_exch=0.3, prob_bin=0.2, prob_var=0.01),
        mutation=PolynomialMutation(repair=RoundingRepair())
    )

    Result     = minimize(
        problem=NASProblem, algorithm=Algorithm,
        termination=('n_gen', 10)
    )
    
    Best_Value = -1 * np.inf
    Best_X     = []
    for i in Result.pop:
        Now_Value = -1*(i.F[0]+i.F[1])
        if(Now_Value > Best_Value):
            Best_Value = Now_Value
            Best_X     = i.X
    print("[",",".join([str(i) for i in Best_X]),"]")
    print(Best_Value)
