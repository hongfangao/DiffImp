import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
# from ModelUtils.mamba import Mamba, MambaConfig
from ModelUtils.mamba import MambaConfig
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, ConstantKernel, ExpSineSquared, Exponentiation
from sklearn.gaussian_process import GaussianProcessRegressor
import math
import torch
import gpytorch

class MambaLayers(nn.Module):
    def __init__(self,cfg:MambaConfig):
        super().__init__()
        mambas = []
        for _ in range(cfg.n_layers):
            mambas.append(Mamba(d_model=cfg.d_model,dt_init='constant'))
        self.mlayers = nn.Sequential(*mambas)
    def forward(self,x):
        return self.mlayers(x)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
    # layer = nn.utils.parametrizations.weight_norm(layer)
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
        table = torch.cat([torch.sin(table),torch.cos(table)],dim=1) # [num_steps, 2*dim]
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
    
class UResidualMambaSeq(nn.Module):
    '''
    Args:
    d_model: d_model arg in SSM
    n_layer: n_layer in mambaconfig, number of residuals
    mid_ssm_num: nums of ssms in the lowest resolution
    level: level of downsample modules
    '''
    def __init__(self,seq_len:int,n_layer:int,mid_ssm_num:int,level:int=2):
        super().__init__()
        assert seq_len%pow(2,level) == 0
        self.level = level
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(level):
            self.downs.append(DownConv1d(seq_len//pow(2,i)))
            cur_config = MambaConfig(d_model=seq_len//pow(2,i+1),n_layers=n_layer)
            self.downs.append(MambaLayers(cur_config))
            # self.downs.append(Mamba(cur_config))
        '''
        middle parts        
        '''
        mid_config = MambaConfig(seq_len//pow(2,level),n_layer)
        mid_ssms = []
        for _ in range(mid_ssm_num):
            # mid_ssms.append(Mamba(mid_config))
            mid_ssms.append(MambaLayers(mid_config))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(level,0,-1):
            self.ups.append(UpConv1d(seq_len//pow(2,i)))
            cur_config = MambaConfig(seq_len//pow(2,i-1),n_layer)
            self.ups.append(MambaLayers(cur_config))
            # self.ups.append(Mamba(cur_config))
    
    def forward(self,x):
        residuals = []
        for i in range(0,2*self.level,2):
            residuals.append(x)
            x = x.permute(0,2,1)
            x = self.downs[i](x)
            x = x.permute(0,2,1)
            x = self.downs[i+1](x)
        x = self.mid_ssms(x)
        for i in range(0,2*self.level,2):
            x = x.permute(0,2,1)
            x = self.ups[i](x)
            x = x.permute(0,2,1)
            x = self.ups[i+1](x)
            x = x + residuals[int(-0.5*i)+self.level-1]
        return x
    
class UResidualMambaChannel(nn.Module):
    def __init__(self,d_model:int,n_layer:int,mid_ssm_num:int,level:int=2):
        super().__init__()
        assert d_model%pow(2,level)==0
        self.level = level
        '''
        downsample
        '''
        self.downs = nn.ModuleList()
        for i in range(level):
            self.downs.append(DownConv1d(d_model//pow(2,i)))
            cur_config = MambaConfig(d_model//pow(2,i+1),n_layer)
            # self.downs.append(Mamba(cur_config))
            self.downs.append(MambaLayers(cur_config))
        '''
        middle part
        '''
        mid_config = MambaConfig(d_model//pow(2,level),n_layer)
        mid_ssms = []
        for _ in range(mid_ssm_num):
            # mid_ssms.append(Mamba(mid_config))
            mid_ssms.append(MambaLayers(mid_config))
        self.mid_ssms = nn.Sequential(*mid_ssms)
        '''
        upsample
        '''
        self.ups = nn.ModuleList()
        for i in range(level,0,-1):
            self.ups.append(UpConv1d(d_model//pow(2,i)))
            cur_config = MambaConfig(d_model//pow(2,i-1),n_layer)
            # self.ups.append(Mamba(cur_config))
            self.ups.append(MambaLayers(cur_config))
    def forward(self,x):
        residuals = []
        for i in range(0, 2*self.level, 2):
            residuals.append(x)
            x = self.downs[i](x)
            x = x.permute(0,2,1)
            x = self.downs[i+1](x)
            x = x.permute(0,2,1)
        x = x.permute(0,2,1)
        x = self.mid_ssms(x)
        x = x.permute(0,2,1)
        for i in range(0,2*self.level,2):
            x = self.ups[i](x)
            x = x.permute(0,2,1)
            x = self.ups[i+1](x)
            x = x.permute(0,2,1)
            x = x + residuals[int(-0.5*i)+self.level-1]
        return x
    
class UResidualMamba(nn.Module):
    def __init__(self,d_model:int,seq_len:int,n_layer:int,mid_ssm_num:int,level:int):
        super().__init__()
        self.URMC = UResidualMambaChannel(d_model,n_layer,mid_ssm_num,level)
        self.URMS = UResidualMambaSeq(seq_len,n_layer,mid_ssm_num,level)

    def forward(self,x):
        out = self.URMS(x)
        out = self.URMC(out)
        out = out + x
        return out

class SequentialSSM(nn.Module):
    def __init__(self,d_model,seq_len,n_layers,num_ssm):
        super().__init__()
        self.num_ssm = num_ssm
        self.ssms = nn.ModuleList()
        for _ in range(num_ssm):
            # self.ssms.append(Mamba(MambaConfig(seq_len,n_layers)))
            # self.ssms.append(Mamba(MambaConfig(d_model,n_layers)))
            self.ssms.append(MambaLayers(MambaConfig(seq_len,n_layers)))
            self.ssms.append(MambaLayers(MambaConfig(d_model,n_layers)))
    def forward(self,x):
        out = x
        for i in range(0,2*self.num_ssm,2):
            x = self.ssms[i](x)
            x = x.permute(0,2,1)
            x = self.ssms[i+1](x)
            x = x.permute(0,2,1)
        return out + x 

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,res_channels,diffusion_embedding_dim,cond_ssm_num,input_ssm_num,mid_ssm_num,d_model,n_layer,seq_len,level):
        super().__init__()
        self.URM1 = UResidualMamba(d_model,seq_len,n_layer,mid_ssm_num,level)
        self.URM2 = UResidualMamba(2*d_model,seq_len,n_layer,mid_ssm_num,level)
        self.cond_ssm = SequentialSSM(2*d_model,seq_len,n_layer,cond_ssm_num)
        self.d_model = d_model
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.seq_len = seq_len
        '''
        ssm for input
        '''
        self.input_ssm = SequentialSSM(d_model,seq_len,n_layer,input_ssm_num)
        '''
        diffusion projection
        '''
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim,res_channels)
        self.input_proj = Conv1d_with_init(in_channels,d_model,1)
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
        B,C,L = x.shape
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
        out = torch.tanh(h[:,:self.res_channels,:])* torch.sigmoid(h[:,self.res_channels:,:])
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)
        return (x+res)*math.sqrt(0.5), skip

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel,self).__init__(train_x,train_y,likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self,x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean,covar)

class SSMGP2Imputer(nn.Module):
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
        mid_ssm_num,
        d_model,
        n_layer,
        level,
        samples
    ):
        super().__init__()
        self.samples = samples
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps = num_steps,
            embedding_dim = diffusion_embedding_dim
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
                    mid_ssm_num=mid_ssm_num,
                    d_model=d_model,
                    n_layer=n_layer,
                    seq_len=seq_len,
                    level=level
                )
                for _ in range(layers)
            ]            
        )
    def forward(self, input_data):
        noise, condition, mask, diffusion_step = input_data
        condition = condition * mask
        condition = torch.cat([condition, mask.float()],dim=1)
        B, C, L = noise.shape
        diffusion_step_embed = cal_diffusion_step_embedding(diffusion_step,self.diffusion_embedding_dim)
        h = noise
        skip = 0
        for n in range(self.layers):
            h, skip_n = self.residual_layers[n]((h,condition,diffusion_step_embed))
            skip += skip_n
        x = skip/math.sqrt(self.layers)
        x = self.out_proj1(x)
        mean, std_d = torch.chunk(x,2,1)
        observations = noise * mask
        observations = observations.flatten()
        nonzero_indices = torch.nonzero(observations)
        x_pred = torch.where(observations==0)[0]
        '''
        x_observed
        '''
        sampled_indices = torch.randperm(nonzero_indices.shape[0])
        x_obs = nonzero_indices[sampled_indices[:self.samples]]
        y_obs = observations[x_obs]
        x_obs = x_obs.squeeze(1)
        y_obs = y_obs.squeeze(1)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_obs,y_obs,likelihood).cuda()
        model.train()
        likelihood.train()
        gp_optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
        training_iter = 50
        for _ in range(training_iter):
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
    

