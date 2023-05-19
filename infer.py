import os
import numpy as np
import yaml
import time
import torch
from pprint import pprint
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from skimage.util import img_as_float32
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms

from loss import CustomLoss, ZeroReferenceLoss
from utils import save_model, load_model, set_seed
from utils import get_base_name, get_timecode
from utils import get_dataset, get_model, eval, save_tensor, display_tensors


def test(config):
	# check device
	if config['device']:
		device = torch.device(config['device'])
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device [{device}]")

	# load image & pre-process
	image = img_as_float32(io.imread(config['image_path']))
	toTensor = transforms.Compose([
			np.ascontiguousarray,
			transforms.ToTensor()
	])
	image = toTensor(image).unsqueeze(dim=0).to(device)

	# model initiation & load weights
	model = get_model(config)
	state, _ = load_model(config['eval_model_path'])
	model.load_state_dict(state)
	model.to(device)

	set_seed(config['seed'])
	
	if config['profiling']:
		with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_inference"):
				start = time.time()
				out = model(image)
				total_time = time.time() - start
		print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
		print(f"[Model Infer Time: {total_time:.3f} s ({total_time*1000:.1f} ms)]")
	else:
		out = model(image)
	
	# display result
	if config['show']:
		display_tensors(out[0])

	# save result
	out_dir = os.path.join(config['visualization_save_path'], "inference")
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, f"out-{get_base_name(config['image_path'])}")
	save_tensor(out[0], out_path)
	print(f"Testing model: {config['eval_model_path']} \nUsing input image: {config['image_path']}\nSaved results to: {out_path}")


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Zongnan Bao Capstone Project')
	parser.add_argument('--config', type=str, required=True, help='path to config file')
	parser.add_argument('--image_path', type=str, required=True, help='path to test image')
	parser.add_argument('--show', action="store_true", help='display image after inference if enabled')

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
