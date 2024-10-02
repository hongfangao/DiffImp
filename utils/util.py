import os
import numpy as np
import torch
import random
import logging

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        logging.info("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6))


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    logging.info('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            if torch.isnan(epsilon_theta).any():
                pass
                t = t + 1
            else:
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta 
    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask

def training_loss_gp(net, loss_diff, loss_gp, X, diffusion_hyperparams, coeff,only_generate_missing=1):
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B,C,L = audio.shape
    diffusion_steps = torch.randint(T,size=(B,1,1)).cuda()
    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z *(1-mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta, std_d, std_g = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta 
    if only_generate_missing == 1:
        return loss_diff(epsilon_theta[loss_mask],z[loss_mask]) + coeff * loss_gp(std_d,std_g)
    elif only_generate_missing == 0:
        return loss_diff(epsilon_theta,z) + coeff * loss_gp(std_d,std_g)
    

def sampling_bigpssm(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    logging.info('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            mean, std_d = net((x, cond, mask, diffusion_steps))
            
            epsilon_theta = mean
            # update x_{t-1} to \mu_\theta(x_t)
            if torch.isnan(epsilon_theta).any():
                pass
                t = t + 1
            else:
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x, std_d

def calc_quantile_CRPS_custom(target, forecast, eval_points, mask, mean_scaler=0, scaler=1):
    """
    Custom CRPS calculation for individual time steps with missing data mask.
    
    Parameters:
    - target: True target values.
    - forecast: Generated forecast samples.
    - eval_points: Mask for valid points (0 or 1).
    - mask: Mask indicating missing values.
    - mean_scaler: Mean value used for scaling back the data.
    - scaler: Scaling factor applied to the data.

    Returns:
    - CRPS: Continuous Ranked Probability Score.
    """
    # 还原尺度
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    # 只计算缺失部分
    mask = mask.to(torch.bool)
    
    quantiles = np.arange(0.05, 1.0, 0.05)  # 19个分位数
    CRPS = 0
    
    # 计算每个分位数的预测值
    for q in quantiles:
        q_pred = torch.quantile(forecast, q, dim=1)  # 在生成的样本维度上计算分位数
        logging.info("target shape:{}".format(target.shape))
        logging.info("q_pred shape:{}".format(q_pred.shape))
        logging.info("eval_points shape:{}".format(eval_points.shape))
        logging.info("mask shape:{}".format(mask.shape))
        # 计算分位数损失，只对缺失数据区域进行计算
        q_loss = quantile_loss(target[mask], q_pred[mask], q, eval_points[mask])
        denom = calc_denominator(target[mask], eval_points[mask])

        # 累加分位数损失
        CRPS += q_loss / denom

    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum_custom(target, forecast, eval_points, mask, mean_scaler=0, scaler=1):
    """
    Custom CRPS_SUM calculation which aggregates the time dimension before calculating CRPS.
    
    Parameters:
    - target: True target values.
    - forecast: Generated forecast samples.
    - eval_points: Mask for valid points (0 or 1).
    - mask: Mask indicating missing values.
    - mean_scaler: Mean value used for scaling back the data.
    - scaler: Scaling factor applied to the data.

    Returns:
    - CRPS_SUM: Aggregated CRPS score across time dimension.
    """
    # 还原尺度
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    # 只计算缺失部分
    mask = mask.to(torch.bool)

    # 对时间维度进行聚合操作
    target_sum = target.sum(dim=-1)
    forecast_sum = forecast.sum(dim=-1)
    eval_points_sum = eval_points.mean(dim=-1)

    quantiles = np.arange(0.05, 1.0, 0.05)  # 19个分位数
    CRPS_SUM = 0

    # 计算每个分位数的预测值
    for q in quantiles:
        q_pred = torch.quantile(forecast_sum, q, dim=1)  # 聚合后的预测值分位数

        # 计算分位数损失，只对缺失数据区域进行计算
        q_loss = quantile_loss(target_sum[mask], q_pred[mask], q, eval_points_sum[mask])
        denom = calc_denominator(target_sum[mask], eval_points_sum[mask])

        # 累加分位数损失
        CRPS_SUM += q_loss / denom

    return CRPS_SUM.item() / len(quantiles)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
