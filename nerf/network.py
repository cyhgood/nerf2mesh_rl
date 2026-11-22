import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

#NeRFNetwork 是项目中 NeRF 的核心网络，主要负责把三维空间坐标和观察方向转换成密度、颜色等信息，
# 供体渲染和网格提取使用。

#编码器提取特征，MLP解码特征为密度，颜色和高光
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 specular_dim=3,
                 ):

        super().__init__(opt)

        # sigma and feature network 主要决定几何形状 encoder：对空间坐标编码，提取几何特征
        self.encoder, self.in_dim_density = get_encoder("hashgrid", level_dim=1, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        #提取颜色相关的空间特征
        self.encoder_color, self.in_dim_color = get_encoder("hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')
        #根据几何特征预测密度σ，决定物体的形状和表面位置 
        self.sigma_net = MLP(self.in_dim_density, 1, 32, 2, bias=False) #密度σ：决定几何表面

        # color network 主要决定颜色
        self.encoder_dir, self.in_dim_dir = get_encoder("None")
        #预测 基础颜色 和 specular feature
        self.color_net = MLP(self.in_dim_color + self.individual_dim, 3 + specular_dim, 64, 3, bias=False)#rgb颜色
        #根据观察方向等信息预测高光，使颜色具有视角相关性
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=False)#主要决定视角相关外观 高光等

    # 综合密度和颜色，输出用于渲染的结果
    def forward(self, x, d, c=None, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # c: [1/N, individual_dim]

        sigma = self.density(x)['sigma']
        color, specular = self.rgb(x, d, c, shading)

        return sigma, color, specular

    #输出密度和几何特征，主要用于体渲染及Stage 0网格提取
    def density(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound)
        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0])

        return {
            'sigma': sigma,
        }
    
    #根据坐标x，提取颜色相关特征 c:可选附加条件，eg.视角、索引或潜在编码
    #x：三维空间坐标，通常形状为[N,3]
    def geo_feat(self, x, c=None):
        #对空间坐标进行哈希网格编码，提取颜色特征
        h = self.encoder_color(x, bound=self.bound)
        if c is not None:
          #c.shape[0] == 1所有点共享同一个特征，将它复制N次；否则认为c已经与每个空间点一一对应；dim=-1，在特征维度当拼接
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        #特征输入color_net 通过MLP进行非线性变换，学习颜色相关表达
        h = self.color_net(h)
        #使用sigmod将输出限制到[0,1]，便于表示归一化后的颜色或特征值
        geo_feat = torch.sigmoid(h)

        return geo_feat #用color_feat命名更合适

    #外观分解为 diffuse 和 specular
    def rgb(self, x, d, c=None, shading='full'):

        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        d = self.encoder_dir(d)
        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                color = (specular + diffuse).clamp(0, 1) # specular + albedo

        return color, specular


    # optimizer utils
    def get_params(self, lr):

        params = super().get_params(lr)

        params.extend([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_color.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.specular_net.parameters(), 'lr': lr}, 
        ])

        return params