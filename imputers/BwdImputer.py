from mamba_ssm.modules.mamba2 import Mamba2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import numpy as np
from einops import rearrange

import math

def swish(x):
    return x*torch.sigmoid(x)

def flip(seq):
    fliped_seq = torch.flip(seq,[2])
    return fliped_seq

def Conv1d_with_init(in_channels,out_channels,kernel_size):
    layer = nn.Conv1d(in_channels,out_channels,kernel_size)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def cal_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim-1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),torch.cos(_embed)), 1)
    return diffusion_step_embed

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            'embedding',
            self._build_embedding(num_steps, embedding_dim/2),
            persistent=False,
        )
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1) # [num_steps,1]
        frequencies = 10.0**(torch.arange(dim)/(dim-1)*4.0).unsqueeze(0) # [1,dim]
        table = steps * frequencies # [num_steps, dim]
        table = torch.cat([torch.sin(table),torch.cos(table)],dim=1)
        return table
    
class MambaEncoderForward(nn.Module):
    def __init__(self,input_dim:int,expand:int,headdim:int):
        super().__init__()
        self.input_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.weight_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.fwd_proj = Conv1d_with_init(2*input_dim,2*input_dim,1)
        self.fwd_ssm = Mamba2(d_model=2*input_dim,expand=expand,headdim=headdim)
        self.out_proj = Conv1d_with_init(2*input_dim,input_dim,1)
    def forward(self,x):
        residual = x
        self.norm = RMSNorm(x.shape[2]).cuda()
        x = self.norm(x)
        x = rearrange(x,'b c l -> b l c')
        ssm_input = x
        weight_input = x
        ssm_input = flip(ssm_input)
        ssm_input = self.input_proj(ssm_input)
        ssm_input = self.fwd_proj(ssm_input)
        ssm_input = rearrange(ssm_input, 'b l c -> b c l')
        ssm_input = self.fwd_ssm(ssm_input)
        ssm_input = rearrange(ssm_input, 'b c l -> b l c')
        weight_input = self.weight_proj(weight_input)
        weight_input = swish(weight_input)
        out = weight_input * ssm_input
        out = self.out_proj(out)
        out = rearrange(out,'b l c -> b c l')
        out = out + residual
        return out


class SequentialSSM(nn.Module):
    def __init__(self,num_ch,seq_len,num_ssm,expand_c,expand_s,headdim_c,headdim_s):
        super().__init__()
        self.num_ssm = num_ssm
        self.ssms = nn.ModuleList()
        for _ in range(num_ssm):
            self.ssms.append(MambaEncoderForward(seq_len,expand=expand_s,headdim=headdim_s))
            self.ssms.append(MambaEncoderForward(num_ch,expand=expand_c,headdim=headdim_c))
    def forward(self,x):
        out = x
        for i in range(0,2*self.num_ssm,2):
            x = self.ssms[i](x)
            x = rearrange(x,'b c l -> b l c')
            x = self.ssms[i+1](x)
            x = rearrange(x,'b l c -> b c l')            
        return out + x
    

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,res_channels,diffusion_embedding_dim,seq_dim,num_ssm,num_ch,seq_len,cond_ssm_num,input_ssm_num,expand_c,expand_s,headdim_c,headdim_s):
        super().__init__()
        self.ssm1 = SequentialSSM(num_ch=num_ch,seq_len=seq_dim,num_ssm=num_ssm,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.ssm2 = SequentialSSM(num_ch=2*num_ch,seq_len=seq_dim,num_ssm=num_ssm,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.cond_ssm = SequentialSSM(num_ch=2*num_ch,seq_len=seq_dim,num_ssm=cond_ssm_num,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.num_ch = num_ch
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.seq_proj = Conv1d_with_init(seq_len,seq_dim,1)
        self.input_ssm = SequentialSSM(num_ch=num_ch,seq_len=seq_dim,num_ssm=input_ssm_num,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim,res_channels)
        self.input_proj = Conv1d_with_init(in_channels,num_ch,1)
        self.mid_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.out_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.res_conv = Conv1d_with_init(res_channels,in_channels,1)
        self.res_proj_len = Conv1d_with_init(seq_dim,seq_len,1)
        self.skip_conv = Conv1d_with_init(res_channels,res_channels,1)
        self.cond_conv = Conv1d_with_init(2*in_channels,2*res_channels,1)
        self.cond_proj_len = Conv1d_with_init(seq_len,seq_dim,1)
        self.skip_proj_len = Conv1d_with_init(seq_dim,seq_len,1)
    def forward(self,input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.in_channels
        assert L == self.seq_len
        '''
        inputs
        '''
        h = self.input_proj(h)
        h = rearrange(h,'b c l -> b l c')
        h = self.seq_proj(h)
        h = rearrange(h,'b l c -> b c l')
        h = self.input_ssm(h)
        '''
        diffusion embedding
        '''
        diffemb = self.diffusion_proj(diffusion_step_embed)
        diffemb = diffemb.view([B,self.res_channels,1])
        h = h + diffemb
        h = self.ssm1(h)
        h = self.mid_proj(h)
        '''
        cond embedding
        '''
        assert cond is not None
        cond = self.cond_conv(cond)
        cond = rearrange(cond, 'b c l -> b l c')
        cond = self.cond_proj_len(cond)
        cond = rearrange(cond, 'b l c -> b c l')
        cond = self.cond_ssm(cond)
        h += cond
        h = self.ssm2(h)
        out = torch.tanh(h[:,:self.res_channels,:])*torch.sigmoid(h[:,self.res_channels:,:])
        res = self.res_conv(out)
        res = rearrange(res,'b c l -> b l c')
        res = self.res_proj_len(res)
        res = rearrange(res,'b l c -> b c l')
        assert x.shape == res.shape
        skip = self.skip_conv(out)
        skip = rearrange(skip,'b l c -> b c l')
        skip = self.skip_proj_len(skip)
        skip = rearrange(skip,'b c l -> b l c')
        return (x+res)*math.sqrt(0.5), skip

class BiCoreMImputer(nn.Module):
    def __init__(
        self,
        layers,
        seq_len,
        seq_dim,
        in_channels,
        res_channels,
        diffusion_embedding_dim,
        num_steps,
        num_ssm,
        cond_ssm_num,
        input_ssm_num,
        num_ch,
        expand_c,
        expand_s,
        headdim_c,
        headdim_s
    ):
        super().__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps= num_steps,
            embedding_dim= diffusion_embedding_dim
        )
        self.input_proj = Conv1d_with_init(in_channels,res_channels,1)
        self.out_proj1 = Conv1d_with_init(self.res_channels,self.in_channels,1)
        self.layers = layers
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels,
                    res_channels=res_channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    cond_ssm_num=cond_ssm_num,
                    input_ssm_num=input_ssm_num,
                    num_ssm = num_ssm,
                    num_ch=num_ch,
                    seq_len=seq_len,
                    seq_dim=seq_dim,
                    expand_c=expand_c,
                    expand_s=expand_s,
                    headdim_c=headdim_c,
                    headdim_s=headdim_s
                )
                for _ in range(layers)
            ]
        )
    def forward(self,input_data):
        noise, condition, mask, diffusion_step = input_data
        condition = condition * mask
        condition = torch.cat([condition, mask.float()], dim=1)
        diffusion_step_embed = cal_diffusion_step_embedding(diffusion_step,self.diffusion_embedding_dim)
        h = noise
        skip = 0
        for n in range(self.layers):
            h, skip_n = self.residual_layers[n]((h,condition,diffusion_step_embed))
            skip += skip_n
        x = skip/math.sqrt(self.layers)
        x = self.out_proj1(x)
        return x