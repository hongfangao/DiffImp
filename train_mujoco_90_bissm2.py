import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import logging
import datetime

from torch.utils.data import Dataset, DataLoader

from utils.util import find_max_epoch, print_size, training_loss_gp, calc_diffusion_hyperparams,training_loss
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm

# from imputers.BISSMGPImputer import BiM2GPImputer
from imputers.BISSM2Imputer import BiSSM2Imputer

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    filename='train_'+ current_time + '.log',
    filemode='w'
)

logger = logging.getLogger(__name__)

logging.info("training bissm")
logging.info("the dataset is mujoco")

def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logging.info("output directory:{}".format(output_directory))

    # Ensure diffusion_hyperparams are on the GPU
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    if use_model == 1:
        net = BiSSM2Imputer(**model_config).cuda()
    else:
        logging.info('Model chosen not available.')

    logging.info(net)
    print_size(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        ckpt_iter = 450000
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logging.info('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            logging.info('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        logging.info('No valid checkpoint model found, start training from initialization.')

    ### Custom data loading and reshaping ###
    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.split(training_data, 160, 0)
    training_data = np.array(training_data)
    training_data = torch.from_numpy(training_data).float().cuda()
    logging.info('Data loaded')

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k).cuda()
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k).cuda()
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k).cuda()

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1).cuda()

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            # loss = training_loss_gp(net, nn.MSELoss(), nn.MSELoss(), X, diffusion_hyperparams, coeff=1,
            #                             only_generate_missing=only_generate_missing)
            loss = training_loss(net,nn.MSELoss(),X,diffusion_hyperparams,only_generate_missing=only_generate_missing)
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                logging.info("iteration: {} \tloss: {}".format(n_iter, loss.item()))

                # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(output_directory, checkpoint_name))
                logging.info('model at iteration %s is saved' % n_iter)

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_bissm2_mujoco_90.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    logging.info(config)

    global ssm_config
    ssm_config = config['biSSM2_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 1:
        model_config = config['biSSM2_config']

    train(**train_config)