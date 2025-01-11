import torch.nn as nn

__all__ = ["OPS", "MSTF_NAS_SPACE_NORMAL", "MSTF_NAS_SPACE_REDUCT"]

# 使用字典构建Lambda函数，调用相应Operations
OPS = {
    'zero'             : lambda c, affine : Zero(),
    'identity'         : lambda c, affine : nn.Identity(),
    'sep_conv_3x3'     : lambda c, affine : SepConv(
        in_c=c, out_c=c, kernel_s=3, stride=1, padding=1, affine=affine
    ),
    'sep_conv_5x5'     : lambda c, affine : SepConv(
        in_c=c, out_c=c, kernel_s=5, stride=1, padding=2, affine=affine
    ),
    'sep_conv_7x7'     : lambda c, affine : SepConv(
        in_c=c, out_c=c, kernel_s=7, stride=1, padding=3, affine=affine
    ),
    'dil_sep_conv_3x3' : lambda c, affine : DilSepConv(
        in_c=c, out_c=c, kernel_s=3, stride=1, padding=2, dilation=2, affine=affine
    ),
    'dil_sep_conv_5x5' : lambda c, affine : DilSepConv(
        in_c=c, out_c=c, kernel_s=5, stride=1, padding=4, dilation=2, affine=affine
    ),
    'dual_conv_7x7'    : lambda c, affine :  ReLUDualConvBN(
        in_c=c, out_c=c, kernel_s=7, stride=1, padding=3, affine=affine
    ),
    
    'avg_com_pool_2x2' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=2, stride=2, padding=0, in_c=in_c, out_c=out_c, pool_type='Avg', affine=affine
    ),
    'max_com_pool_2x2' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=2, stride=2, padding=0, in_c=in_c, out_c=out_c, pool_type='Max', affine=affine
    ),
    'avg_com_pool_3x3' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=3, stride=2, padding=1, in_c=in_c, out_c=out_c, pool_type='Avg', affine=affine
    ),
    'max_com_pool_3x3' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=3, stride=2, padding=1, in_c=in_c, out_c=out_c, pool_type='Max', affine=affine
    ),
    'avg_com_pool_4x4' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=4, stride=2, padding=1, in_c=in_c, out_c=out_c, pool_type='Avg', affine=affine
    ),
    'max_com_pool_4x4' : lambda in_c, out_c, affine : CompressPooling(
        kernel_size=4, stride=2, padding=1, in_c=in_c, out_c=out_c, pool_type='Max', affine=affine
    ),
}

MSTF_NAS_SPACE_NORMAL = [
    # 以下OPS，形状不变，通道不变
    'zero',
    'identity',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_sep_conv_3x3',
    'dil_sep_conv_5x5',
    'dual_conv_7x7',
]
MSTF_NAS_SPACE_REDUCT = [
    # 以下OPS，形状减半，通道不定
    'avg_com_pool_2x2',
    'max_com_pool_2x2',
    'avg_com_pool_3x3',
    'max_com_pool_3x3',
    'avg_com_pool_4x4',
    'max_com_pool_4x4',
]


'''Will be used in the Normal Cell'''
# ReLU + Conv + BN
class ReLUConvBN(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=kernel_s, 
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# ReLU + Conv + Conv + BN:
class ReLUDualConvBN(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=in_c, kernel_size=(1,kernel_s), 
                stride=(1,stride), padding=(0,padding), bias=False
            ),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=(kernel_s,1),
                stride=(stride,1), padding=(padding,0), bias=False
            ),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# Dilation Separable Convolution
class DilSepConv(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, dilation=1, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=in_c, dilation=dilation, groups=in_c,       # 当groups等于通道数是，即可分离卷积
                kernel_size=kernel_s, stride=stride, padding=padding, bias=False
            ),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# Separable Convolution
class SepConv(nn.Module):
    def __init__(self, 
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super(SepConv, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=kernel_s, stride=stride, padding=padding, groups=in_c, bias=False),
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_c, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=kernel_s, stride=1, padding=padding, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )

    def forward(self, Inputs):
        return self.main(Inputs)

class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Inputs):
        return Inputs.mul(0.0)


'''Will be used in the Reduct Cell'''
class CompressPooling(nn.Module):
    def __init__(self,
        kernel_size, stride, padding, in_c, out_c, 
        pool_type:str='Max',affine=True,
    ):
        super().__init__()
        if pool_type not in ['Avg', 'Max']:
            ValueError("The type of pooling isn't valid! It only can be 'Avg' or 'Max'!")
        self.pool     = nn.ModuleList()
        self.pool.append(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if pool_type == 'Max'
            else nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False),
        )
        if kernel_size % 2:     # 当kernel_size为奇数时，增添一个向下取整的pool操作
            self.pool.append(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding-1) if pool_type == 'Max'
                else nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding-1, count_include_pad=False),
            )
        self.compress = nn.Identity() if in_c == out_c else ReLUConvBN(in_c=in_c, out_c=out_c, kernel_s=1, stride=1, padding=0, affine=affine)
    
    def forward(self, Inputs):
        # 当height和width为奇数时，选择向下取整的pool操作
        Inputs = self.pool[-1](Inputs) if (Inputs.shape[-1] % 2) else self.pool[0](Inputs)
        return self.compress(Inputs)
