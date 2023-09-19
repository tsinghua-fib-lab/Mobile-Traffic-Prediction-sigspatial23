import torch
import torch.nn as nn
import math
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv,RelGraphConv
from dgl.nn.pytorch import ChebConv
from torch import nn, Tensor
from math import sqrt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoder(nn.Module):
    def __init__(self,dropout = 0.1,max_seq_len = 100,d_model = 32,batch_first = True):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first
        self.d_model = d_model

        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n

        dist = torch.softmax(dist, dim = -1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att


class SelfAttention_new(nn.Module):
    def __init__(self,device, dim = 64):
        super(SelfAttention_new, self).__init__( )
        self.dim = dim

        self.c = torch.autograd.Variable(torch.rand((1,dim))).to(device)
        self.V = torch.autograd.Variable(torch.rand((dim,dim))).to(device)
        self.b = torch.autograd.Variable(torch.rand(dim)).to(device)


    def forward(self, x,y):
        n, dim_q = x.shape
        V = self.V.repeat(n,1,1)
        b = self.b.repeat(n,1)
        a_x = torch.mm(torch.tanh(torch.matmul(V,x.unsqueeze(-1)).squeeze() + b),self.c.T)
        a_y = torch.mm(torch.tanh(torch.matmul(V, y.unsqueeze(-1)).squeeze( ) + b),self.c.T)

        b_x = torch.exp(a_x) / (torch.exp(a_x) + torch.exp(a_y))
        b_y = torch.exp(a_y) / (torch.exp(a_x) + torch.exp(a_y))
        # print(b_x.shape,x.shape)
        out = b_x*x.squeeze()+b_y*y.squeeze()
        return out

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 dilation = 1,
                 groups = 1,
                 bias = True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0,
            dilation = dilation,
            groups = groups,
            bias = bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels = 1, embedding_size = 256, k = 5):
        super(context_embedding, self).__init__( )
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size = k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return F.tanh(x)


class MRKGCN(nn.Module):
    def __init__(self,input_size = 1,seq_len = 12,n_trans_layers_1 = 1,n_trans_layers_2 = 1,n_rgcn_layers_base = 1,n_rgcn_layers_urban = 1,
                 hidden_dim = 64,urban_emb_size = 32,base_emb_size = 64,num_rel = 34,num_bases = 4,dim_k = 32,dim_v = 16,
                 n_head_1 = 8,n_head_2 = 8,out_dim = 1,kernal_size = 3):
        super(MRKGCN,self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len

        self.n_trans_layers_2 = n_trans_layers_2
        self.n_rgcn_layers_base = n_rgcn_layers_base
        self.n_rgcn_layers_urban = n_rgcn_layers_urban
        self.hidden_dim = hidden_dim
        self.urban_emb_size = urban_emb_size
        self.base_emb_size = base_emb_size

        self.n_head_2 = n_head_2
        self.out_dim = out_dim
        self.conv_layer = context_embedding(1,base_emb_size,kernal_size)

        self.rgcns_base = nn.ModuleList()
        for i in range(n_rgcn_layers_base):
            self.rgcns_base.append(RelGraphConv(self.base_emb_size,self.base_emb_size,num_rel,'basis',num_bases))

        self.positional_encoding_layer_2 = PositionalEncoder(d_model = self.base_emb_size)
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model = self.base_emb_size, nhead = n_head_2, batch_first = True)
        self.encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers = self.n_trans_layers_2, norm = None)


        self.rgcns_urban = nn.ModuleList( )
        for i in range(n_rgcn_layers_urban):
            self.rgcns_urban.append(RelGraphConv(self.urban_emb_size, self.urban_emb_size, num_rel, 'basis', num_bases))

        self.attention = SelfAttention(self.base_emb_size + urban_emb_size, dim_k, dim_v)
        self.output_layer = nn.Linear(dim_v, self.out_dim)
    def forward(self,x,g_base,etype_base,x_urban,g_urban,etype_urban):
        num_nodes = x.shape[1]
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        x = x.reshape(batch_size*num_nodes,seq_len,1)
        x = x.permute(0, 2, 1)
        x = self.conv_layer(x)
        x = x.permute(0, 2, 1)
        x1 = torch.zeros(x.shape).to(device)
        for j in range(seq_len):
            for i in range(self.n_rgcn_layers_base):
                x1[:,j,] = self.rgcns_base[i](g_base,x[:,j,],etype_base)

        x1 = self.positional_encoding_layer_2(x1)
        x1 = self.encoder_2(x1)
        x1 = torch.mean(x1,dim = 1)
        for i in range(self.n_rgcn_layers_urban):
            x_urban = self.rgcns_urban[i](g_urban,x_urban,etype_urban)
        # x = self.gcn1(g,x)
        # print(x.shape,x_urban.shape)
        x1 = torch.concat((x1,x_urban[:num_nodes]),dim = 1)
        x1 = x1.unsqueeze(0)
        x1 = self.attention(x1)
        x1 = x1.squeeze(0)
        x1 = self.output_layer(x1)
        return x1


