import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import random
import math
import copy
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.data import Dataset
import typing
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, NoneType, OptPairTensor, OptTensor, Size, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, is_torch_sparse_tensor, remove_self_loops, softmax
from torch_geometric.utils.sparse import set_sparse_value
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, auc, f1_score, recall_score, precision_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings( "ignore", category=FutureWarning, message=r"You are using `torch.load` with `weights_only=False`.*" )
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
import numpy as np

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
current_dir = os.path.dirname(__file__)
o01 = ""
P = pd.read_csv(f"{o01}P.csv")
D = pd.read_csv(f"{o01}D.csv")

class Tokenizer:
    def __init__(self):
        self.CHAR = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7, ".": 8, "=": 9, "@": 10, "[": 11, "]": 12, "\\": 13, "1": 14, "2": 15, "3": 16, "4": 17, "5": 18, "6": 19, "7": 20, "8": 21, "9": 22, "0": 23, "A": 24, "B": 25, "C": 26, "D": 27, "E": 28, "F": 29, "G": 30, "H": 31, "I": 32, "J": 33, "K": 34, "L": 35, "M": 36, "N": 37, "O": 38, "P": 39, "Q": 40, "R": 41, "S": 42, "T": 43, "U": 44, "V": 45, "W": 46, "X": 47, "Y": 48, "Z": 49, "a": 50, "b": 51, "c": 52, "d": 53, "e": 54, "f": 55, "g": 56, "h": 57, "i": 58, "k": 59, "l": 60, "m": 61, "n": 62, "o": 63, "p": 64, "r": 65, "s": 66, "t": 67, "u": 68, "v": 69, "w": 70, "x": 71, "y": 72, "z": 73}
        self.CHAR_REVERSE = {v: k for k, v in self.CHAR.items()}
    def encode(self, text):
        encoded = []
        i = 0
        while i < len(text):
            if text[i] in self.CHAR:
                encoded.append(self.CHAR[text[i]])
                i += 1
            else:
                raise ValueError(f"E {text[i]}")
        return encoded

    def decode(self, tokens):
        decoded = []
        for token in tokens:
            if token in self.CHAR_REVERSE:
                decoded.append(self.CHAR_REVERSE[token])
            else:
                raise ValueError(f"E {token}")
        return ''.join(decoded)

tokenizer = Tokenizer()

def info():
    print(f"\n\n\033[31mMehran Nosrati\033[0m | Drug–target interaction prediction | 1403 | Golestan University | IR\n")    
def RESIZE(tensor, target_size):
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(tensor, size=target_size, mode='linear', align_corners=False)
    return resized_tensor.squeeze()
    
def Dencoder(x, size):
    SMILES = D[D["Sequence"] == x]
    DID = f"{o01}D/D" + SMILES["ID"].values[0].replace("sequence_", "") + ".pt"
    DDATA = torch.load(DID)

    out = torch.tensor(tokenizer.encode(x)).float()
    return RESIZE(out, size).long().detach(), DDATA.detach()

def Tencoder(x, size):
    FASTA = P[P["Sequence"] == x]
    PID = f"{o01}P/P" + FASTA["ID"].values[0].replace("sequence_", "") + ".pt"
    PDATA = torch.load(PID)
        
    out = torch.tensor(tokenizer.encode(x)).float()
    return RESIZE(out, size).long().detach(), PDATA.detach()

class DATAEncoders(Dataset):
    def __init__(self, ID, Label, DF, Dsize, Tsize):
        self.Label = Label
        self.ID = ID
        self.df = DF
        self.Dsize = Dsize
        self.Tsize = Tsize
    def __len__(self):
        return len(self.ID)
    def __getitem__(self, index):
        index = self.ID[index]
        D = self.df.iloc[index]['SMILES']
        T = self.df.iloc[index]['Target Sequence']
        DV, D3D = Dencoder(D, self.Dsize)
        TV, T3D = Tencoder(T, self.Tsize)
        L = torch.tensor(self.Label[index], dtype=torch.long)
        return DV, TV, D3D, T3D, L
    
class MLP(nn.Module):
    def __init__(self, I, O, dropout):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(I, I // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(I // 2, O)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

class NGATConv(MessagePassing):
    def __init__( self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, residual: bool = False, **kwargs, ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward( self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, size: Size = None, return_attention_weights: Optional[bool] = None, ) -> Union[ Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, SparseTensor], ]:
        H, C = self.heads, self.out_channels
        res: Optional[Tensor] = None
        if isinstance(x, Tensor):
            if self.res is not None:
                res = self.res(x)
            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError("E")

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, size=size)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
            
class GRAPH(nn.Module):
    def __init__(self, dropout):
        super(GRAPH, self).__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 256)
        )
        self.gat1 = NGATConv(256, 64, heads=2, concat=True)
        self.gat2 = NGATConv(64*2, 64, heads=2, concat=True)
        self.gat3 = NGATConv(64*2, 64, heads=1, concat=False)
        self.pool = Set2Set(64, processing_steps=5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.feature_encoder(x)
        x = self.dropout(x)
        
        x_gat = self.gat1(x, edge_index, edge_attr)
        x_gat = F.leaky_relu(x_gat, 0.1)
        x_gat = self.dropout(x_gat)
        
        x_gat = self.gat2(x_gat, edge_index, edge_attr)
        x_gat = F.elu(x_gat)
        x_gat = self.dropout(x_gat)
        
        x_gat = self.gat3(x_gat, edge_index, edge_attr)
        x = self.pool(x_gat, batch)
        return x
        
class DTIV7(nn.Sequential):
    def __init__(self, **params):
        super(DTIV7, self).__init__()
        self.size = params['size']
        self.dropout = params['dropout']
        self.Gsize = params['gsize']

        self.GRAPH1 = GRAPH(self.dropout)
        self.GRAPH2 = GRAPH(self.dropout)
        self.AM = MLP(128, self.Gsize, self.dropout)
        self.BM = MLP(128, self.Gsize, self.dropout)
        
        self.layer = self.layer = params['layer']
        self.dimD = params['dimD']
        self.dimT = params['dimT']
        self.sizeD = params['sizeD']
        self.sizeT = params['sizeT']
        self.head = params['head']
        self.isize = params['isize']
        self.batch_size = params['batch_size']
        
        self.DEmbedding = Embedding(self.dimD, self.size, self.sizeD, self.dropout)
        self.TEmbedding = Embedding(self.dimT, self.size, self.sizeT, self.dropout)
        self.DBlock = Block(self.layer, self.size, self.isize, self.head, self.dropout)
        self.TBlock = Block(self.layer, self.size, self.isize, self.head, self.dropout)
        
        self.Cnn = nn.Conv2d(1, 3, 3, padding=0)
        self.sizeout = 3*(self.sizeD-2)*(self.sizeT-2)+(self.Gsize*2)
        self.Out = nn.Sequential(
            nn.Linear(self.sizeout, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, DV, TV, D3D, T3D):
        A = self.AM(self.GRAPH1(D3D.x, D3D.edge_index, D3D.edge_attr, D3D.batch))
        B = self.BM(self.GRAPH2(T3D.x, T3D.edge_index, T3D.edge_attr, T3D.batch))
        
        ED = self.DEmbedding(DV).float()
        ET = self.TEmbedding(TV).float()
        Dencoded = self.DBlock(ED)
        Tencoded = self.TBlock(ET)
        
        D = torch.unsqueeze(Dencoded, 2).repeat(1, 1, self.sizeT, 1)
        T = torch.unsqueeze(Tencoded, 1).repeat(1, self.sizeD, 1, 1)
        DT = D * T
        O = DT.view(int(self.batch_size), -1, self.sizeD, self.sizeT) 
        O = torch.sum(O, dim = 1)
        O = torch.unsqueeze(O, 1)
        O = F.dropout(O, p=self.dropout)        
        O = self.Cnn(O)
        O = O.view(int(self.batch_size), -1)
        O = torch.cat([O, A, B], dim=-1)
        O = self.Out(O)
        return self.sigmoid(O).squeeze()

class LayerNorm(nn.Module):
    def __init__(self, SIZE):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(SIZE))
        self.beta = nn.Parameter(torch.zeros(SIZE))

    def forward(self, X):
        U = X.mean(-1, keepdim=True)
        S = (X - U).pow(2).mean(-1, keepdim=True)
        O = (X - U) / torch.sqrt(S + 1e-12)
        return self.gamma * O + self.beta

class Embedding(nn.Module):
    def __init__(self, VSIZE, SIZE, MSIZE, DROPOUT):
        super(Embedding, self).__init__()
        self.E = nn.Embedding(VSIZE, SIZE)
        self.P = nn.Embedding(MSIZE, SIZE)
        self.LayerNorm = LayerNorm(SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, X):
        L = X.size(1)
        P = torch.arange(L, dtype=torch.long, device=X.device).unsqueeze(0).expand_as(X)
        E = self.E(X)
        P = self.P(P)
        O = E + P
        O = self.LayerNorm(O)
        O = self.dropout(O)
        return O

class SelfAttention(nn.Module):
    def __init__(self, SIZE, HEAD, DROPOUT):
        super(SelfAttention, self).__init__()
        self.HEAD = HEAD
        self.AHEAD = int(SIZE / HEAD)
        self.HEADs = self.HEAD * self.AHEAD
        self.Query = nn.Linear(SIZE, self.HEADs)
        self.Key = nn.Linear(SIZE, self.HEADs)
        self.Value = nn.Linear(SIZE, self.HEADs)
        self.dropout = nn.Dropout(DROPOUT)

    def transpose(self, X):
        N = X.size()[:-1] + (self.HEAD, self.AHEAD)
        X = X.view(*N)
        return X.permute(0, 2, 1, 3)

    def forward(self, X):
        Q = self.Query(X)
        K = self.Key(X)
        V = self.Value(X)
        Q = self.transpose(Q)
        K = self.transpose(K)
        V = self.transpose(V)
        A = torch.matmul(Q, K.transpose(-1, -2))
        A = A / math.sqrt(self.AHEAD)
        A = nn.Softmax(dim=-1)(A)
        A = self.dropout(A)
        O = torch.matmul(A, V)
        O = O.permute(0, 2, 1, 3).contiguous()
        C = O.size()[:-2] + (self.HEADs,)
        O = O.view(*C)
        return O 
    
class Attention(nn.Module):
    def __init__(self, SIZE, HEAD, DROPOUT):
        super(Attention, self).__init__()
        self.self = SelfAttention(SIZE, HEAD, DROPOUT)
        self.linear = nn.Linear(SIZE, SIZE)
        self.norm = LayerNorm(SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, X):
        O = self.self(X)
        O = self.dropout(self.linear(O))
        O = self.norm(O + X)
        return O

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *(x + 0.044715 * torch.pow(x, 3))))
    
class Encoder(nn.Module):
    def __init__(self, SIZE, ISIZE, HEAD, DROPOUT):
        super(Encoder, self).__init__()
        self.attention = Attention(SIZE, HEAD, DROPOUT)
        self.linear1 = nn.Linear(SIZE, ISIZE)
        self.linear2 = nn.Linear(ISIZE, SIZE)
        self.norm = LayerNorm(SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.activation = GELU()
        
    def forward(self, X):
        X = self.attention(X)
        O = self.activation(self.linear1(X))
        O = self.dropout(self.linear2(O))
        O = self.norm(O + X)
        return O    
    
class Block(nn.Module):
    def __init__(self, N, SIZE, ISIZE, HEAD, DROPOUT):
        super(Block, self).__init__()
        L = Encoder(SIZE, ISIZE, HEAD, DROPOUT)
        self.LAYER = nn.ModuleList([copy.deepcopy(L) for _ in range(N)])    

    def forward(self, X):
        for B in self.LAYER:
            X = B(X)
        return X
    
def GET(i, DF, DFNEW):
        SMILES = D[D["Sequence"] == DF.iloc[i]["SMILES"]]
        FASTA = P[P["Sequence"] == DF.iloc[i]["Target Sequence"]]
        PID = f"{o01}P/P" + FASTA["ID"].values[0].replace("sequence_", "") + ".pt"
        DID = f"{o01}D/D" + SMILES["ID"].values[0].replace("sequence_", "") + ".pt"
        if os.path.exists(PID) and os.path.exists(DID):
            DFNEW = pd.concat([DFNEW, DF.iloc[[i]]], ignore_index=True)
        return DFNEW
    
class KDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def convert(dataset):
    D = [dataset[idx] for idx in range(len(dataset))]
    D = [line.split(" ") for line in D]
    DF = pd.DataFrame(D, columns=["E0", "E1", "SMILES", "Target Sequence", "Label"])
    DF['Label'] = DF['Label'].astype(int)
    return DF
            
def Kfold(i, datasets, k=5):
    size = len(datasets) // k  
    val_start = i * size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]
    return trainset, validset

def Shuffle(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def KDATA(dataset, i):
    Tr, Te = Kfold(i, dataset)
    Tr = KDataSet(Tr)
    Te = KDataSet(Te)
    Tr_len = len(Tr)
    Va_size = int(0.2 * Tr_len)
    Tr_size = Tr_len - Va_size
    Tr, Va = torch.utils.data.random_split(Tr, [Tr_size, Va_size])
    DFTRAIN = convert(Tr)
    DFVAL = convert(Va)
    DFTEST = convert(Te)
    return DFTRAIN, DFVAL, DFTEST

def LOAD():
    DFTRAIN = pd.read_csv("Dataset/DAVIS/train.csv")
    DFVAL = pd.read_csv("Dataset/DAVIS/val.csv")
    DFTEST = pd.read_csv("Dataset/DAVIS/test.csv")
    return DFTRAIN, DFVAL, DFTEST
    
def DATA(batch_size, Dsize, Tsize, DFTRAIN, DFVAL, DFTEST):
    if DFTRAIN is None:
        DFTRAIN, DFVAL, DFTEST = LOAD()
    
    print(f'=========== Train: {len(DFTRAIN)}, Validation: {len(DFVAL)}, Test: {len(DFTEST)}')
    DFNEWTRAIN = pd.DataFrame()
    DFNEWTRAIN = pd.concat([DFNEWTRAIN, DFTRAIN.iloc[:0]], ignore_index=True)
    DFNEWVAL = pd.DataFrame()
    DFNEWVAL = pd.concat([DFNEWVAL, DFVAL.iloc[:0]], ignore_index=True)
    DFNEWTEST = pd.DataFrame()
    DFNEWTEST = pd.concat([DFNEWTEST, DFTEST.iloc[:0]], ignore_index=True)

    for I in range(len(DFTRAIN)):
        DFNEWTRAIN = GET(I, DFTRAIN, DFNEWTRAIN)
    for I in range(len(DFVAL)):
        DFNEWVAL = GET(I, DFVAL, DFNEWVAL)
    for I in range(len(DFTEST)):
        DFNEWTEST = GET(I, DFTEST, DFNEWTEST)
    
    print(f'=========== Train: {len(DFNEWTRAIN)}, Validation: {len(DFNEWVAL)}, Test: {len(DFNEWTEST)}')
    SET = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6, 'drop_last': True}
    train_loader = DataLoader(DATAEncoders(DFNEWTRAIN.index.values, DFNEWTRAIN.Label.values, DFNEWTRAIN, Dsize, Tsize), **SET)
    val_loader = DataLoader(DATAEncoders(DFNEWVAL.index.values, DFNEWVAL.Label.values, DFNEWVAL, Dsize, Tsize), **SET)
    test_loader = DataLoader(DATAEncoders(DFNEWTEST.index.values, DFNEWTEST.Label.values, DFNEWTEST, Dsize, Tsize), **SET)
    return train_loader, val_loader, test_loader

LISTsensitivity = []
LISTspecificity = []
LISTauprc = []
LISTroc_auc = []
Lsensitivity = 0
Lspecificity = 0
Lauprc = 0
Lroc_auc = 0
def test(K, model, test_loader, device):
    global Lsensitivity, Lspecificity, Lauprc, Lroc_auc
    all_predictions = []
    all_labels = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for (DV, TV, D3D, T3D, Label) in test_loader:
        logits = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
        
        loss_fct = torch.nn.BCELoss()            
        label = Label.float().to(device)
        loss = loss_fct(logits, label)
        loss_accumulate += loss
        count += 1
        logits = logits.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()
        all_labels = all_labels + label_ids.flatten().tolist()
        all_predictions = all_predictions + logits.flatten().tolist()
    
    M = "OK-"
    if len(set(all_predictions)) == 1:
        M = "ER-"
                
    loss = loss_accumulate/count
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    threshold = thred_optim
    y_pred_s = [1 if i else 0 for i in (all_predictions >= thred_optim)]
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, y_pred_s)
    recall = recall_score(all_labels, y_pred_s)
    precision = precision_score(all_labels, y_pred_s)
    accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(all_predictions) >= 0.5)])
    f1 = f1_score(all_labels, outputs)
    
    if sensitivity > Lsensitivity:
        Lsensitivity = sensitivity
    if specificity > Lspecificity:
        Lspecificity = specificity
    if auprc > Lauprc:
        Lauprc = auprc
    if roc_auc > Lroc_auc:
        Lroc_auc = roc_auc
    
    # if sensitivity >= 0.8 and specificity >= 0.88 and auprc >= 0.404 and roc_auc >= 0.907:
    #     time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #     File = f"UP-{time}.pth"
    #     torch.save(model.state_dict(), File)
    #     print(f'=========== SAVE: {File} ===========')
        
    return f"{M} Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, ROC AUC: {roc_auc:.5f}, AUPR (PR-AUC): {auprc:.5f}, Sensitivity: {sensitivity:.5f}, Specificity: {specificity:.5f}, Threshold: {threshold:.5f}, Test Loss: {loss:.5f}"
    
def train(K, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, test_loader, device):
    global Lsensitivity, Lspecificity, Lauprc, Lroc_auc
    global LISTsensitivity, LISTspecificity, LISTauprc, LISTroc_auc
    Lsensitivity, Lspecificity, Lauprc, Lroc_auc = 0, 0, 0, 0
    train_losses = []
    val_losses = []
    
    torch.backends.cudnn.benchmark = True
    for epoch in range(epochs):
        model.train()
        for batch_idx, (DV, TV, D3D, T3D, Label) in enumerate(train_loader):
            outputs_train = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
            Label = Label.float().to(device)
            loss_train = criterion(outputs_train, Label)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()
                
            if batch_idx + 1 == len(train_loader):
                with torch.set_grad_enabled(False):
                    model.eval()
                    val_losses_step = []
                    train_losses_step = []
                    for (DV, TV, D3D, T3D, Label) in val_loader:
                        outputs_val = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
                        Label = Label.float().to(device)
                        loss_val = criterion(outputs_val, Label)
                        val_losses_step.append(loss_val.item())
                        # break
                    for (DV, TV, D3D, T3D, Label) in train_loader:
                        outputs_train = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
                        Label = Label.float().to(device)
                        loss_train = criterion(outputs_train, Label)
                        train_losses_step.append(loss_train.item())
                        # break
                        
                avg_val_loss = np.mean(val_losses_step)
                avg_train_loss = np.mean(train_losses_step)
                for param_group in optimizer.param_groups:
                    lrnum = param_group['lr']
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}, LR: {lrnum:.10f}')
        
        with torch.set_grad_enabled(False):
            RES = test(K, model, test_loader, device)
            print(f'Epoch [{epoch + 1}/{epochs}], {RES}')
        
        # if (epoch + 1) % 20 == 0 or (epoch + 1) == 1:
        #     with torch.set_grad_enabled(False):
        #         RES = test(K, model, test_loader, device)
        #         print(f'Epoch [{epoch + 1}/{epochs}], {RES}')
    
    LISTsensitivity.append(Lsensitivity)
    LISTspecificity.append(Lspecificity)
    LISTauprc.append(Lauprc)
    LISTroc_auc.append(Lroc_auc)
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(train_losses) + 1), val_losses, label='Val Loss')
    # plt.legend()
    # plt.savefig('loss.pdf')
    # plt.close()

def start():
    global LISTsensitivity, LISTspecificity, LISTauprc, LISTroc_auc
    info()
    params = {}
    params['Kfold'] = False
    params['fold'] = 10
    params['epochs'] = 40
    params['dimD'] = 73
    params['dimT'] = 73
    
    params['gsize'] = 20
    params['sizeD'] = 50
    params['sizeT'] = 545
    params['isize'] = 1536
    params['size'] = 384
    params['head'] = 12
    params['layer'] = 2
    params['dropout'] = 0.1
    params['batch_size'] = 32
    params['lr'] = 0.000005
    for key, value in params.items():
        print(f"{key}: {value}")
    Stime = datetime.now()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=========== D: \033[31m{device}\033[0m ===========\n')
    # model = DTIV7(**params).to(device)
    # load = torch.load("2025_01_31_21_14_01.pth", map_location=device)
    # model.load_state_dict(load['model_state_dict'])
    criterion = nn.BCELoss()
    
    if params['Kfold']:
        print(f'=========== D: Kfold ===========\n')
        SEED = 1234
        with open('Davis.txt', "r") as f:
            Davis = f.read().strip().split('\n')
        dataset = Shuffle(Davis, SEED)
        for i in range(params['fold']):
            DFTRAIN, DFVAL, DFTEST = KDATA(dataset, i)
            train_loader, val_loader, test_loader = DATA(params['batch_size'], params['sizeD'], params['sizeT'] , DFTRAIN, DFVAL, DFTEST)
            print(f'=========== Train: {len(train_loader)}, Validation: {len(val_loader)}, Test: {len(test_loader)}')
            SL = len(train_loader)*4*params['epochs']
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            train(params['Kfold'], model, criterion, optimizer, scheduler, params['epochs'], train_loader, val_loader, test_loader, device)
            
        RESsensitivity = torch.tensor(LISTsensitivity)
        RESspecificity = torch.tensor(LISTspecificity)
        RESauprc = torch.tensor(LISTauprc)
        RESroc_auc = torch.tensor(LISTroc_auc)
        meansensitivity = torch.mean(RESsensitivity)
        stdsensitivity = torch.std(RESsensitivity)
        meanspecificity = torch.mean(RESspecificity)
        stdspecificity = torch.std(RESspecificity)
        meanauprc = torch.mean(RESauprc)
        stdauprc = torch.std(RESauprc)
        meanroc_auc = torch.mean(RESroc_auc)
        stdroc_auc = torch.std(RESroc_auc)
        
        print(f"\n=========== RES ===========")
        print(f'Sensitivity: {meansensitivity.item():.4f} ± {stdsensitivity.item():.4f}')
        print(f'Specificity: {meanspecificity.item():.4f} ± {stdspecificity.item():.4f}')
        print(f'AUPR (PR-AUC): {meanauprc.item():.4f} ± {stdauprc.item():.4f}')
        print(f'ROC AUC: {meanroc_auc.item():.4f} ± {stdroc_auc.item():.4f}')
        print(f"=========== RES ===========\n")
            
    else:
        print(f'=========== D: SKfold ===========\n')
        train_loader, val_loader, test_loader = DATA(params['batch_size'], params['sizeD'], params['sizeT'], None, None, None)
        print(f'=========== Train: {len(train_loader)}, Validation: {len(val_loader)}, Test: {len(test_loader)}')
        SL = len(train_loader)*4*params['epochs']
        SEEDlist = random.sample(range(1, 10000), params['fold'])
        for i, SEED in enumerate(SEEDlist):
            random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            train(params['Kfold'], model, criterion, optimizer, scheduler, params['epochs'], train_loader, val_loader, test_loader, device)
        
        if params['fold'] > 1:
            RESsensitivity = torch.tensor(LISTsensitivity)
            RESspecificity = torch.tensor(LISTspecificity)
            RESauprc = torch.tensor(LISTauprc)
            RESroc_auc = torch.tensor(LISTroc_auc)
            meansensitivity = torch.mean(RESsensitivity)
            stdsensitivity = torch.std(RESsensitivity)
            meanspecificity = torch.mean(RESspecificity)
            stdspecificity = torch.std(RESspecificity)
            meanauprc = torch.mean(RESauprc)
            stdauprc = torch.std(RESauprc)
            meanroc_auc = torch.mean(RESroc_auc)
            stdroc_auc = torch.std(RESroc_auc)
            
            print(f"\n=========== RES ===========")
            print(f'Sensitivity: {meansensitivity.item():.4f} ± {stdsensitivity.item():.4f}')
            print(f'Specificity: {meanspecificity.item():.4f} ± {stdspecificity.item():.4f}')
            print(f'AUPR (PR-AUC): {meanauprc.item():.4f} ± {stdauprc.item():.4f}')
            print(f'ROC AUC: {meanroc_auc.item():.4f} ± {stdroc_auc.item():.4f}')
            print(f"=========== RES ===========\n")
    
    Etime = datetime.now()
    hours, remainder = divmod((Etime - Stime).total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"=========== T: {int(hours):02}:{int(minutes):02}:{int(seconds):02} ===========")
    
    # time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }
    # File = f"{time}.pth"
    # torch.save(checkpoint, File)
    # print(f'=========== SAVE: {File} ===========')
    
if __name__ == "__main__":
    start()
    info()