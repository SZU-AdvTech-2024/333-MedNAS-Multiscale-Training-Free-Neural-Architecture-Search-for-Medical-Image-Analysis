import torch
import torch.nn as nn

# Created by myself
from .Feature_Extract import *
from .operators import *
from .operators import ReLUConvBN

# genotype是一个有整数组成的定长列表：在Normal中，genotype=[ops_idx, input_idx]*n；在Reduct中，genotype=[ops_idx]*n；n为Cell中的节点数

# 形状不变，通道数不变
class Normal_Cell(nn.Module):
    def __init__(self, in_c:int, genotype:list):
        super().__init__()
        self.num_node = len(genotype)//2    # 计算Normal Cell中的节点个数
        self.ops_idx  = genotype[0::2]      # ops_idx中元素的取值范围从0开始
        self.in_idx   = genotype[1::2]      # in_idx中元素的取值范围从1开始，0代表的是上一层的输出

        # 解码genotype为Normal Cell的具体operation
        self.OPS_List = nn.ModuleList()
        for idx in self.ops_idx:
            self.OPS_List.append(OPS[MSTF_NAS_SPACE_NORMAL[idx]](c=in_c, affine=True))

        # 记录Node的出度情况，得到self.out_c和要执行concat的Node编号
        self.recoder  = torch.zeros(self.num_node+1)
        for idx in set(self.in_idx):    self.recoder[idx] = 1
        self._concat  = torch.where(self.recoder==0)[0].numpy().tolist()     # 出度为0的Node需要进行concat
        # self.out_c    = in_c * len(self._concat)    # Concat Mode
        self.out_c    = in_c                        # Add Mode
        # print(self.recoder, '\n', self._concat, '\n', self.out_c)

    def forward(self, Inputs):
        State   = [Inputs]
        for idx in range(self.num_node):
            op    = self.OPS_List[idx]          # 从self.OPS_List中取当前节点的输入操作
            input = State[self.in_idx[idx]]     # 从State中取当前节点的输入值
            State.append(op(input))
        # Outputs = torch.concat([State[idx] for idx in self._concat], dim=1)             # Concat Mode
        Outputs = torch.stack([State[idx] for idx in self._concat], dim=0).mean(dim=0)  # Add Mode
        return Outputs


# 形状减半，通道数翻倍
class Reduct_Cell(nn.Module):
    def __init__(self, in_c:int, genotype:list):
        super().__init__()
        self.num_node     = len(genotype)   # 计算Reduction Cell中的节点个数
        self.out_c        = 2 * in_c
        if not (in_c % self.num_node): ValueError("in_c should be divisible by num_node.")

        # 解码genotype为Reduction Cell的具体operation
        self.POOL_List    = nn.ModuleList()
        for idx in genotype:
            self.POOL_List.append(OPS[MSTF_NAS_SPACE_REDUCT[idx]](in_c=in_c, out_c=in_c//self.num_node, affine=True))

        # 使用1*1卷积，令通道数翻倍
        self.Post_Process = ReLUConvBN(in_c=in_c, out_c=self.out_c, kernel_s=1, stride=1, padding=0)
    
    def forward(self, Inputs):
        Temp_List = []
        for op in self.POOL_List:   Temp_List.append(op(Inputs))
        Inputs    = torch.concat(Temp_List, dim=1)
        Outputs   = self.Post_Process(Inputs)
        return Outputs


class Block(nn.Module):
    def __init__(self, in_c:int, n_n:int, n_r:int, genotype:list):
        super().__init__()
        # 划分得到Normal_Cell和Reduct_Cell的genotype
        self.normal_genotype, self.reduct_genotype = genotype[:2*n_n], genotype[-1*n_r:]

        # 声明Normal_Cell和Reduct_Cell各一个，并得到最终的输出通道数self.out_c
        self.N_C   = Normal_Cell(in_c=in_c, genotype=self.normal_genotype)
        self.out_c = self.N_C.out_c
        self.R_C   = Reduct_Cell(in_c=self.out_c, genotype=self.reduct_genotype)
        self.out_c = self.R_C.out_c
    
    def forward(self, Inputs):
        Outputs = self.N_C(Inputs)
        Outputs = self.R_C(Outputs)
        return Outputs


class MSTF_NAS_NETWORK(nn.Module):
    def __init__(self, num_class:int, genotype:list, n_n:int=5, n_r:int=4, num_block:int=3, in_c=3, init_c:int=128):
        super().__init__()
        # 计算Block的genotype长度
        self.len_block_genotype = n_n * 2 + n_r
        if(self.len_block_genotype*num_block != len(genotype)): ValueError("The length of genotype does not match with n_n and n_r!")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=init_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=init_c),
        )
        self.in_c = init_c

        self.main_stream = nn.ModuleList()
        for idx in range(num_block):
            block_genotype = genotype[idx*self.len_block_genotype: (idx+1)*self.len_block_genotype]
            self.main_stream.append(Block(in_c=self.in_c, n_n=n_n, n_r=n_r, genotype=block_genotype))
            self.in_c      = self.main_stream[idx].out_c
        self.main_stream = nn.Sequential(*self.main_stream)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(in_features=self.in_c, out_features=num_class, bias=True)
        # print(self)
    
    def forward(self, Inputs):
        Outputs = self.stem(Inputs)
        Outputs = self.main_stream(Outputs)
        Outputs = self.global_pool(Outputs)
        Outputs = Outputs.view(Outputs.size(0), -1)
        Outputs = self.classifier(Outputs)
        return Outputs
