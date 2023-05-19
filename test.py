import os
import yaml
import torch
from pprint import pprint
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from loss import CustomLoss, ZeroReferenceLoss
from utils import save_model, load_model, set_seed
from utils import get_base_name, get_timecode
from utils import get_dataset, get_model, eval, load_model


def test(config):
	# check device
	if config['device']:
		device = torch.device(config['device'])
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device [{device}]")

	# load dataset
	train_set, val_set, test_set = get_dataset(config)

	# model initiation & load weights
	model = get_model(config)
	state, _ = load_model(config['eval_model_path'])
	model.load_state_dict(state)
	model.to(device)

	set_seed(config['seed'])

	# evaluation and save visualization result
	out_dir = os.path.join(config['visualization_save_path'], f"test-{config['name']}-{get_timecode()}")
	psnr, ssim, mae, lpips = eval(model, test_set, device, out_dir=out_dir, profiling=config['profiling'])
	print(f"Testing model: {config['eval_model_path']} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f} | LPIPS = {lpips:.3f}")


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Zongnan Bao Capstone Project')
	parser.add_argument('--config', type=str, required=True, help='path to config file')

	# convert argparsed args to dictionary 
	args = vars(parser.parse_args())

	# read & converts yaml document
	with open(args['config'], 'r') as f:
		try:
			config = yaml.safe_load(f)
		except yaml.YAMLError as error:
			print(error)

	# merge args with config files (args have higher priority)
	config.update(args)
	pprint(config)

	test(config=config)
