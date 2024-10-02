from mamba_ssm.modules.mamba2 import Mamba2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import numpy as np
from einops import rearrange
import math
import gpytorch

def swish(x):
    return x * torch.sigmoid(x)

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

class PostNormMamba2(nn.Module):
    def __init__(self,d_model:int,expand:int,headdim:int):
        super().__init__()
        self.mamba2 = Mamba2(d_model=d_model,expand=expand,headdim=headdim)
    def forward(self,x):
        self.norm = RMSNorm(x.shape[2]).cuda()
        out = self.mamba2(x)
        out = self.norm(out)
        return out + x
    
class MambaEncoderFlip(nn.Module):
    def __init__(self,in_dim:int,expand:int,headdim:int):
        super().__init__()
        self.input_proj = Conv1d_with_init(in_dim,2*in_dim,1)
        self.weight_proj = Conv1d_with_init(in_dim,2*in_dim,1)
        self.fwd_proj = Conv1d_with_init(2*in_dim,2*in_dim,1)
        self.fwd_ssm = PostNormMamba2(2*in_dim,expand,headdim)
        self.bwd_proj = Conv1d_with_init(2*in_dim,2*in_dim,1)
        self.bwd_ssm = PostNormMamba2(2*in_dim,expand,headdim)
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
    def __init__(self,input_dim:int,expand:int,headdim:int):
        super().__init__()
        self.input_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.weight_proj = Conv1d_with_init(input_dim,2*input_dim,1)
        self.fwd_proj = Conv1d_with_init(2*input_dim,2*input_dim,1)
        self.fwd_ssm = PostNormMamba2(2*input_dim,expand,headdim)
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
    def __init__(self,seq_len:int,mid_ssm_num_s:int,expand_s:int,headdim_s:int,n_levels:int=2):
        super().__init__()
        assert seq_len%pow(2,n_levels)==0
        self.level = n_levels
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(self.level):
            self.downs.append(DownConv1d(seq_len//pow(2,i)))
            self.downs.append(MambaEncoderFlip(seq_len//pow(2,i+1),expand_s,headdim_s))
        '''
        middle parts
        '''
        mid_ssms = []
        for _ in range(mid_ssm_num_s):
            mid_ssms.append(MambaEncoderFlip(seq_len//pow(2,n_levels),expand_s,headdim_s))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(n_levels,0,-1):
            self.ups.append(UpConv1d(seq_len//pow(2,i)))
            self.ups.append(MambaEncoderFlip(seq_len//pow(2,i-1),expand_s,headdim_s))
        
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
    def __init__(self,num_ch:int,mid_ssm_num_c:int,expand_c:int,headdim_c:int,n_levelc:int=4):
        super().__init__()
        assert num_ch%pow(2,n_levelc)==0
        self.level = n_levelc
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(n_levelc):
            self.downs.append(DownConv1d(num_ch//pow(2,i)))
            self.downs.append(MambaEncoderForward(num_ch//pow(2,i+1),expand_c,headdim_c))
        '''
        middle part
        '''
        mid_ssms = []
        for _ in range(mid_ssm_num_c):
            mid_ssms.append(MambaEncoderForward(num_ch//pow(2,n_levelc),expand_c,headdim_c))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(n_levelc,0,-1):
            self.ups.append(UpConv1d(num_ch//pow(2,i)))
            self.ups.append(MambaEncoderForward(num_ch//pow(2,i-1),expand_c,headdim_c))

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
    def __init__(self,seq_len,num_ch,mid_ssm_num_s,mid_ssm_num_c,expand_s,expand_c,headdim_s,headdim_c,n_levels=2,n_levelc=4):
        super().__init__()
        self.URMC = UResidualMambaChannel(num_ch=num_ch,mid_ssm_num_c=mid_ssm_num_c,n_levelc=n_levelc,expand_c=expand_c,headdim_c=headdim_c)
        self.URMS = UResidualMambaSeq(seq_len=seq_len,mid_ssm_num_s=mid_ssm_num_s,n_levels=n_levels,expand_s=expand_s,headdim_s=headdim_s)
    def forward(self,x):
        out = self.URMS(x)
        out = self.URMC(out)
        out = out + x
        return out
    
class SequentialSSM(nn.Module):
    def __init__(self,num_ch,seq_len,num_ssm,expand_c,expand_s,headdim_c,headdim_s):
        super().__init__()
        self.num_ssm = num_ssm
        self.ssms = nn.ModuleList()
        for _ in range(num_ssm):
            self.ssms.append(MambaEncoderFlip(seq_len,expand=expand_s,headdim=headdim_s))
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
    def __init__(self,in_channels,res_channels,diffusion_embedding_dim,seq_dim,cond_ssm_num,input_ssm_num,mid_ssm_num_c,mid_ssm_num_s,num_ch,seq_len,expand_c,expand_s,headdim_c,headdim_s,n_levels=2,n_levelc=4):
        super().__init__()
        self.URM1 = UResidualMamba(num_ch=num_ch,seq_len=seq_dim,mid_ssm_num_c=mid_ssm_num_c,mid_ssm_num_s=mid_ssm_num_s,n_levelc=n_levelc,n_levels=n_levels,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.URM2 = UResidualMamba(seq_len=seq_dim,num_ch=2*num_ch,mid_ssm_num_c=mid_ssm_num_c,mid_ssm_num_s=mid_ssm_num_s,n_levels=n_levels,n_levelc=n_levelc,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        self.cond_ssm = SequentialSSM(num_ch=2*num_ch,seq_len=seq_dim,num_ssm=cond_ssm_num,expand_s=expand_s,expand_c=expand_c,headdim_c=headdim_c,headdim_s=headdim_s)
        self.num_ch = num_ch
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.seq_len = seq_len
        '''
        seq proj
        '''
        self.seq_proj = Conv1d_with_init(seq_len,seq_dim,1)
        '''
        input ssm
        '''
        self.input_ssm = SequentialSSM(num_ch=num_ch,seq_len=seq_dim,num_ssm=input_ssm_num,expand_c=expand_c,expand_s=expand_s,headdim_c=headdim_c,headdim_s=headdim_s)
        ''' 
        diffusion projection
        '''
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim,res_channels)
        self.input_proj = Conv1d_with_init(in_channels,num_ch,1)
        self.mid_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.out_proj = Conv1d_with_init(res_channels,2*res_channels,1)
        self.res_conv = Conv1d_with_init(res_channels,in_channels,1)
        self.res_proj_len = Conv1d_with_init(seq_dim,seq_len,1)
        self.skip_conv = Conv1d_with_init(res_channels,res_channels,1)
        '''
        cond_proj
        '''
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
        h = self.URM1(h)
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
        h = self.URM2(h)
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self,train_x,train_y,likelihood):
        super(ExactGPModel,self).__init__(train_x,train_y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self,x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean,covar)

class BiM2GPImputer(nn.Module):
    def __init__(
        self,
        layers,
        seq_len,
        seq_dim,
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
        expand_c,
        expand_s,
        headdim_c,
        headdim_s,
        samples,
        gp_iter,
        gp_lr
    ):
        super().__init__()
        self.samples = samples
        self.gp_iter = gp_iter
        self.gp_lr = gp_lr
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps= num_steps,
            embedding_dim= diffusion_embedding_dim
        )
        self.input_proj = Conv1d_with_init(in_channels,res_channels,1)
        self.out_proj1 = Conv1d_with_init(self.res_channels,2*self.in_channels,1)
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
                    seq_dim=seq_dim,
                    n_levels=n_levels,
                    n_levelc=n_levelc,
                    expand_c=expand_c,
                    expand_s=expand_s,
                    headdim_c=headdim_c,
                    headdim_s=headdim_s,
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
        mean, std_d = torch.chunk(x,2,1)
        # extract sampled terms
        std_p = std_d
        observations = noise * mask
        observations = observations.flatten()
        nonzero_indices = torch.nonzero(observations).flatten()
        x_pred = torch.where(observations==0)[0]
        sampled_indices = torch.randperm(nonzero_indices.shape[0])
        x_obs = nonzero_indices[sampled_indices[:self.samples]]
        y_obs = observations[x_obs]
        x_obs = x_obs.squeeze(0).cuda()
        y_obs = y_obs.squeeze(0).cuda()
        
        noisy = torch.zeros(x_obs.shape)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noisy,learn_additional_noise=True)
        # likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_obs,y_obs,likelihood).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
        gp_optimizer = torch.optim.Adam(model.parameters(),lr=self.gp_lr)
        model.train()
        likelihood.train()
        for i in range(self.gp_iter):
            gp_optimizer.zero_grad()
            output = model(x_obs)
            loss = -mll(output,y_obs)
            loss.backward()
            gp_optimizer.step()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(x_pred))
            std_g = observed_pred.stddev
        std_d = std_d.flatten()
        std_d = std_d[x_pred]
        return mean, std_d, std_g

class BiM2GPImputer_inference(nn.Module):
    def __init__(
        self,
        layers,
        seq_len,
        seq_dim,
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
        expand_c,
        expand_s,
        headdim_c,
        headdim_s,
        samples,
        gp_iter,
        gp_lr
    ):
        super().__init__()
        self.samples = samples
        self.gp_iter = gp_iter
        self.gp_lr = gp_lr
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps= num_steps,
            embedding_dim= diffusion_embedding_dim
        )
        self.input_proj = Conv1d_with_init(in_channels,res_channels,1)
        self.out_proj1 = Conv1d_with_init(self.res_channels,2*self.in_channels,1)
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
                    seq_dim=seq_dim,
                    n_levels=n_levels,
                    n_levelc=n_levelc,
                    expand_c=expand_c,
                    expand_s=expand_s,
                    headdim_c=headdim_c,
                    headdim_s=headdim_s,
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
        mean, std_d = torch.chunk(x,2,1)
        return mean, std_d

