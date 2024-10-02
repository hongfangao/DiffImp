from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import math

def swish(x):
    '''
    the swish activation function
    '''
    return x* torch.sigmoid(x)

def flip(seq):
    '''
    Args:
    input: sequence shape:(B,C,L)
    Returns:
    flipped sequence at dimension L
    '''
    fliped_seq = torch.flip(seq,[2])
    return fliped_seq

def Conv1d_with_init(in_channels,out_channels,kernel_size):
    '''
    Returns an initialized conv1d layer with:
    1. kaiming_normal initialization
    2. weight_norm
    '''
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
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

class DownConv1d(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        assert in_channels%2==0
        self.conv = Conv1d_with_init(in_channels,in_channels//2,1)
    def forward(self,x):
        return self.conv(x)
    
class UpConv1d(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv = Conv1d_with_init(in_channels,in_channels*2,1)
    def forward(self,x):
        return self.conv(x)

class PostNormMamba(nn.Module):
    def __init__(self,d_model:int):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
    def forward(self,x):
        self.norm = RMSNorm(x.shape[2]).cuda()
        out = self.mamba(x)
        out = self.norm(out)
        return out + x
    
class MambaEncoderFlip(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.input_proj = Conv1d_with_init(in_dim,2*in_dim,1)
        self.weight_proj = Conv1d_with_init(in_dim,2*in_dim,1)
        self.fwd_proj = Conv1d_with_init(2*in_dim,2*in_dim,1)
        self.fwd_ssm = PostNormMamba(2*in_dim)
        self.bwd_proj = Conv1d_with_init(2*in_dim,2*in_dim,1)
        self.bwd_ssm = PostNormMamba(2*in_dim)
        self.out_proj = Conv1d_with_init(2*in_dim,in_dim,1)
    def forward(self,x):
        self.norm = RMSNorm(x.shape[2]).cuda()
        residual = x
        x = self.norm(x)
        x = rearrange(x,'b c l -> b l c')
        ssm_input = x
        weight_input = x
        ssm_input = self.input_proj(ssm_input)
        fwd_input = self.fwd_proj(ssm_input)
        fwd_input = rearrange(fwd_input,'b l c -> b c l')
        fwd_input = self.fwd_ssm(fwd_input)
        fwd_input = rearrange(fwd_input,'b c l -> b l c')
        bwd_input = flip(ssm_input)
        bwd_input = self.bwd_proj(bwd_input)
        bwd_input = rearrange(bwd_input,'b l c -> b c l')
        bwd_input = self.bwd_ssm(bwd_input)
        bwd_input = rearrange(bwd_input,'b c l -> b l c')
        bwd_input = flip(bwd_input)
        weight_input = self.weight_proj(weight_input)
        weight_input = swish(weight_input)
        fwd_output = weight_input * fwd_input
        bwd_output = weight_input * bwd_input
        out = fwd_output + bwd_output
        out = self.out_proj(out)
        out = rearrange(out,'b l c -> b c l')
        out = out + residual
        return out
    
class MambaEncoderForward(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.weight_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.fwd_proj = Conv1d_with_init(2*input_dim,2*input_dim,1)
        self.fwd_ssm = PostNormMamba(2*input_dim)
        self.out_proj = Conv1d_with_init(2*input_dim,input_dim,1)
    def forward(self,x):
        residual = x
        self.norm = RMSNorm(x.shape[2]).cuda()
        x = self.norm(x)
        x = rearrange(x,'b c l -> b l c')
        ssm_input = x
        weight_input = x
        ssm_input = self.input_proj(ssm_input)
        ssm_input = self.fwd_proj(ssm_input)
        ssm_input = rearrange(ssm_input,'b l c -> b c l')
        ssm_input = self.fwd_ssm(ssm_input)
        ssm_input = rearrange(ssm_input,'b c l -> b l c')
        weight_input = self.weight_proj(weight_input)
        weight_input = swish(weight_input)
        out = weight_input * ssm_input
        out = self.out_proj(out)
        out = rearrange(out,'b l c -> b c l')
        out = out + residual
        return out   

class UResidualMambaSeq(nn.Module):
    def __init__(self,seq_len:int,mid_ssm_num_s:int,n_levels:int=2):
        super().__init__()
        assert seq_len%pow(2,n_levels)==0
        self.level = n_levels
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(self.level):
            self.downs.append(DownConv1d(seq_len//pow(2,i)))
            self.downs.append(MambaEncoderFlip(seq_len//pow(2,i+1)))
        '''
        middle parts
        '''
        mid_ssms = []
        for _ in range(mid_ssm_num_s):
            mid_ssms.append(MambaEncoderFlip(seq_len//pow(2,n_levels)))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(n_levels,0,-1):
            self.ups.append(UpConv1d(seq_len//pow(2,i)))
            self.ups.append(MambaEncoderFlip(seq_len//pow(2,i-1)))
        
    def forward(self,x):
        residuals = []
        for i in range(0,2*self.level,2):
            residuals.append(x)
            x = rearrange(x,'b c l -> b l c')
            x = self.downs[i](x)
            x = rearrange(x,'b l c -> b c l')
            x = self.downs[i+1](x)
        x = self.mid_ssms(x)
        for i in range(0,2*self.level,2):
            x = rearrange(x,'b c l -> b l c')
            x = self.ups[i](x)
            x = rearrange(x,'b l c -> b c l')
            x = self.ups[i+1](x)
            x = x + residuals[int(-0.5*i)+self.level-1]
        return x
    
class UResidualMambaChannel(nn.Module):
    def __init__(self,num_ch:int,mid_ssm_num_c:int,n_levelc:int=4):
        super().__init__()
        assert num_ch%pow(2,n_levelc)==0
        self.level = n_levelc
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(n_levelc):
            self.downs.append(DownConv1d(num_ch//pow(2,i)))
            self.downs.append(MambaEncoderForward(num_ch//pow(2,i+1)))
        '''
        middle part
        '''
        mid_ssms = []
        for _ in range(mid_ssm_num_c):
            mid_ssms.append(MambaEncoderForward(num_ch//pow(2,n_levelc)))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(n_levelc,0,-1):
            self.ups.append(UpConv1d(num_ch//pow(2,i)))
            self.ups.append(MambaEncoderForward(num_ch//pow(2,i-1)))

    def forward(self,x):
        residuals = []
        for i in range(0,2*self.level,2):
            residuals.append(x)
            x = self.downs[i](x)
            x = rearrange(x,'b c l -> b l c')
            x = self.downs[i+1](x)
            x = rearrange(x,'b l c -> b c l')
        x = rearrange(x,'b c l -> b l c')
        x = self.mid_ssms(x)
        x = rearrange(x,'b l c -> b c l')
        for i in range(0,2*self.level,2):
            x = self.ups[i](x)
            x = rearrange(x,'b c l -> b l c')
            x = self.ups[i+1](x)
            x = rearrange(x,'b l c -> b c l')
            x = x + residuals[int(-0.5*i)+self.level-1]
        return x

class UResidualMamba(nn.Module):
    def __init__(self,seq_len,num_ch,mid_ssm_num_s,mid_ssm_num_c,n_levels=2,n_levelc=4):
        super().__init__()
        self.URMC = UResidualMambaChannel(num_ch=num_ch,mid_ssm_num_c=mid_ssm_num_c,n_levelc=n_levelc)
        self.URMS = UResidualMambaSeq(seq_len=seq_len,mid_ssm_num_s=mid_ssm_num_s,n_levels=n_levels)
    def forward(self,x):
        out = self.URMS(x)
        out = self.URMC(out)
        out = out + x
        return out
    
class SequentialSSM(nn.Module):
    def __init__(self,num_ch,seq_len,num_ssm):
        super().__init__()
        self.num_ssm = num_ssm
        self.ssms = nn.ModuleList()
        for _ in range(num_ssm):
            self.ssms.append(MambaEncoderFlip(seq_len))
            self.ssms.append(MambaEncoderForward(num_ch))
    def forward(self,x):
        out = x
        for i in range(0,2*self.num_ssm,2):
            x = self.ssms[i](x)
            x = rearrange(x,'b c l -> b l c')
            x = self.ssms[i+1](x)
            x = rearrange(x,'b l c -> b c l')            
        return out + x
    
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,res_channels,diffusion_embedding_dim,cond_ssm_num,input_ssm_num,mid_ssm_num_c,mid_ssm_num_s,num_ch,seq_len,n_levels=2,n_levelc=4):
        super().__init__()
        self.URM1 = UResidualMamba(num_ch=num_ch,seq_len=seq_len,mid_ssm_num_c=mid_ssm_num_c,mid_ssm_num_s=mid_ssm_num_s,n_levelc=n_levelc,n_levels=n_levels)
        self.URM2 = UResidualMamba(seq_len=seq_len,num_ch=2*num_ch,mid_ssm_num_c=mid_ssm_num_c,mid_ssm_num_s=mid_ssm_num_s,n_levels=n_levels,n_levelc=n_levelc)
        self.cond_ssm = SequentialSSM(num_ch=2*num_ch,seq_len=seq_len,num_ssm=cond_ssm_num)
        self.num_ch = num_ch
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.seq_len = seq_len
        '''
        input ssm
        '''
        self.input_ssm = SequentialSSM(num_ch=num_ch,seq_len=seq_len,num_ssm=input_ssm_num)
        '''
        diffusion projection
        '''
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim,res_channels)
        self.input_proj = Conv1d_with_init(in_channels,num_ch,1)
        self.mid_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.out_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.res_conv = Conv1d_with_init(res_channels,in_channels,1)
        self.skip_conv = Conv1d_with_init(res_channels,res_channels,1)
        '''
        cond_proj
        '''
        self.cond_conv = Conv1d_with_init(2*in_channels,2*res_channels,1)
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
        h = self.input_ssm(h)
        '''
        diffusion embedding
        '''
        diffemb = self.diffusion_proj(diffusion_step_embed)
        diffemb = diffemb.view([B,self.res_channels,1])
        h = h + diffemb
        h = self.URM1(h)
        h = self.mid_proj(h)
        '''
        cond embedding
        '''
        assert cond is not None
        cond = self.cond_conv(cond)
        cond = self.cond_ssm(cond)
        h += cond
        h = self.URM2(h)
        out = torch.tanh(h[:,:self.res_channels,:])*torch.sigmoid(h[:,self.res_channels:,:])
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)
        return (x+res)*math.sqrt(0.5), skip

class BiImputer(nn.Module):
    def __init__(
        self,
        layers,
        seq_len,
        in_channels,
        res_channels,
        diffusion_embedding_dim,
        num_steps,
        cond_ssm_num,
        input_ssm_num,
        mid_ssm_num_c,
        mid_ssm_num_s,
        num_ch,
        n_levelc,
        n_levels,
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
                    mid_ssm_num_c=mid_ssm_num_c,
                    mid_ssm_num_s=mid_ssm_num_s,
                    num_ch=num_ch,
                    seq_len=seq_len,
                    n_levels=n_levels,
                    n_levelc=n_levelc
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