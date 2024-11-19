import os
import argparse
import json
import numpy as np
import torch

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from imputers.BISSM2Imputer import BiSSM2Imputer #Take BiSSM2Imputer for example

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch.distributions as dist
from statistics import mean
import logging
import datetime

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    filename='inference_'+ current_time + '.log',
    filemode='w'
)

logger = logging.getLogger(__name__)

def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

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

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    if use_model == 1:
        net = BiSSM2Imputer(**model_config).cuda()
    else:
        logging.info('Model chosen not available.')
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = 450000
        #ckpt_iter = 250000
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'],strict=False)
        logging.info('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    ### Custom data loading and reshaping ###
    testing_data = np.load(trainset_config['test_data_path'])
    testing_data = np.split(testing_data, 4, 0)
    testing_data = np.array(testing_data)
    testing_data = torch.from_numpy(testing_data).float().cuda()
    logging.info('Data loaded')

    all_rmse = []
    all_mse = []
    all_mae = []
    all_rmae = []
    all_mre = []
    inference_times = []
    
    for i, batch in enumerate(testing_data):
        generated_samples = []
        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()

        batch = batch.permute(0, 2, 1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                    diffusion_hyperparams,
                                    cond=batch,
                                    mask=mask,
                                    only_generate_missing=only_generate_missing)

        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000 
        inference_times.append(inference_time)
        logging.info('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                             ckpt_iter,
                                                                                             inference_time))
        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy() 

        outfile = f'imputation{i+20}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i+20}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i+20}.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, mask)

        logging.info('saved generated samples at iteration %s' % ckpt_iter)
        
        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        logging.info('MSE: {}'.format(mse))
        all_mse.append(mse)

        rmse = np.sqrt(mse)
        logging.info('RMSE: {}'.format(rmse))
        all_rmse.append(rmse)

        mae = mean_absolute_error(batch[~mask.astype(bool)], generated_audio[~mask.astype(bool)])
        logging.info('MAE: {}'.format(mae))
        all_mae.append(mae)

        rmae = np.sqrt(mae)
        logging.info('RMAE: {}'.format(rmae))
        all_rmae.append(rmae)

        mre_numerator = np.sum(np.abs(batch[~mask.astype(bool)] - generated_audio[~mask.astype(bool)]))
        mre_denominator = np.sum(np.abs(batch[~mask.astype(bool)]))
        mre = mre_numerator / mre_denominator
        logging.info('MRE: {}'.format(mre))
        all_mre.append(mre)

    logging.info('Total MSE:{}'.format(mean(all_mse)))
    logging.info('Total RMSE:{}'.format(mean(all_rmse)))
    logging.info('Total MAE:{}'.format(mean(all_mae)))
    logging.info('Total RMAE:{}'.format(mean(all_rmae)))
    logging.info('Total MRE:{}'.format(mean(all_mre)))

    inference_times_file = os.path.join(output_directory, 'inference_times_mujoco_bissm2_90_6.json')
    with open(inference_times_file, 'w') as f:
        json.dump(inference_times, f)
    logging.info(f'Saved inference times to {inference_times_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_bissm2_mujoco_90.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    logging.info(config)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
