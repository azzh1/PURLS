import torch
import torch.nn as nn
from math import sqrt

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_emb, dim_k=150, num_heads=2, dtype=torch.float):
        # config
        super(MultiHeadSelfAttention, self).__init__()
        init_dict = locals().copy()
        init_dict.pop('self')
        self.__dict__.update(init_dict)
        self.total_k = self.dim_k * self.num_heads
        # model
        self.linear_q = nn.Linear(self.dim_emb, self.total_k, bias=False, dtype=dtype)
        self.linear_k = nn.Linear(self.dim_in, self.total_k, bias=False, dtype=dtype)
        self._norm_fact = 1 / sqrt(self.dim_k)
        
    def forward(self, x, desc, resnet=False, return_att = False):
        # x: tensor of shape (batch, tp, 256)
        # desc: tensor of shape (batch, 7, 512)
        batch, p, dim_in = x.shape
        batch, n, dim_emb = desc.shape
        assert dim_in == self.dim_in
        assert dim_emb == self.dim_emb
        q = self.linear_q(desc).reshape(batch, n, self.num_heads, self.dim_k).transpose(1, 2)  # (batch, self.num_heads, n, self.dim_k)
        k = self.linear_k(x).reshape(batch, p, self.num_heads, self.dim_k).transpose(1, 2)  # (batch, self.num_heads, p, self.dim_k)
        v = x
        
        # score = torch.rand((batch, n, p), dtype=torch.float).to(x.device)
        score = (torch.matmul(q, k.transpose(2, 3)) * self._norm_fact).mean(1)  # batch, nh, n, n -> batch, n, n
        score = torch.softmax(score, dim=-1)  # batch, 7, 75 (* batch, 75, 256)
        att = torch.matmul(score, v)  # batch, 7, 256
        # att = v.mean(1).unsqueeze(1).repeat(1, 7, 1)
        
        if resnet:
            att += x.mean(dim=1).unsqueeze(1).repeat(1, att.shape[1], 1)
            
        if return_att:
            return att, score
        return att
    