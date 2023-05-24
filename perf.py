import os
import yaml
import torch
import time
import numpy as np
from pprint import pprint
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from loss import CustomLoss, ZeroReferenceLoss
from models.neuralops.networks import Renderer
from models.bilateral_neuralops.networks import BilateralRenderer
from models.bilateral_neuralops.networks import SimpleBilateralRenderer, AdaptiveBilateralRenderer
from models.bilateral_neuralops.networks import TorchSimpleBilateralRenderer
from utils import save_model, load_model, set_seed, save_tensor, hstack_tensors
from utils import get_base_name, get_timecode
from utils import get_dataset, get_model, eval


def perf(config):
	"""
	load model and infer on images with different resolution 
	"""
	# check device
	if config['device']:
		device = torch.device(config['device'])
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device [{device}]")

	# model initiation
	# if config['model']['name'] == 'neuralops':
	# 	model = Renderer(
	# 		in_nc=config['model']['in_nc'],
	# 		out_nc=config['model']['out_nc'],
	# 		base_nf=config['model']['base_nf']
	# 	)
	# 	summary(model, [config['input_size'], config['input_size'], config['input_size'], (1,), (1,), (1,)])
	# 	input = [torch.rand(config['input_size'], device=device) for _ in range(3)] + [0.5, 0.5, 0.5]
	# elif config['model']['name'] == 'bilateral_neuralops':
	# 	model = BilateralRenderer(
	# 		in_nc=config['model']['in_nc'],
	# 		out_nc=config['model']['out_nc']
	# 	)
	# 	summary(model, [config['input_size'], config['input_size'], config['input_size'], (1,), (1,), (1,)])
	# 	input = [torch.rand(config['input_size'], device=device) for _ in range(3)] + [0.5, 0.5, 0.5]
	# elif config['model']['name'] == 'simple_bilateral_neuralops':
	# 	model = SimpleBilateralRenderer(
	# 		in_nc=config['model']['in_nc'],
	# 		out_nc=config['model']['out_nc']
	# 	)
	# 	summary(model, [config['input_size'], config['input_size'], config['input_size'], (1,), (1,), (1,)])
	# 	input = [torch.rand(config['input_size'], device=device) for _ in range(3)] + [0.5, 0.5, 0.5]
	
	# elif config['model']['name'] == 'torch_simple_bilateral_neuralops':
	# 	model = TorchSimpleBilateralRenderer(
	# 		in_nc=config['model']['in_nc'],
	# 		out_nc=config['model']['out_nc']
	# 	)
	# 	summary(model, [config['input_size'], config['input_size'], config['input_size'], (1,), (1,), (1,)])
	# 	input = [torch.rand(config['input_size'], device=device) for _ in range(3)] + [0.5, 0.5, 0.5]
	
	# elif config['model']['name'] == 'adaptive_bilateral_neuralops':
	# 	model = AdaptiveBilateralRenderer(
	# 		n_in=config['model']['n_in'],
	# 		n_out=config['model']['n_out'],
	# 		lowres=config['model']['lowres'],
	# 		luma_bins=config['model']['luma_bins'],
	# 		spatial_bins=config['model']['spatial_bins'],
	# 		channel_multiplier=config['model']['channel_multiplier'],
	# 		guide_pts=config['model']['guide_pts'],
	# 		norm=config['model']['batch_norm'],
	# 		iteratively_upsample=config['model']['iteratively_upsample']
	# 	)
	# 	summary(model, [config['input_size'], config['input_size'], config['input_size'], (1,), (1,), (1,)])
	# 	input = [torch.rand(config['input_size'], device=device) for _ in range(3)] + [0.5, 0.5, 0.5]

	# else:
	model = get_model(config=config)
	input = [torch.rand(config['input_size'], device=device),]

	model.to(device)

	# set seed
	set_seed(config['seed'])

	# inference once
	with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
		with record_function("model_inference"):
			out = model(*input)

	# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))
	print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

	# inference batch && TODO: SHOULD BE WORKING FOR INIT MODEL AS WELL
	dummy_input = torch.rand(config['input_size'], device=device)

	batch_size = 100
	total_time = 0
	with torch.no_grad():
		for _ in tqdm(range(batch_size)):
			starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
			starter.record()
			_ = model(dummy_input)
			ender.record()
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender) # in ms
			total_time += curr_time

	print(f"[Runtime Perf] {total_time/batch_size:.2f} ms per inference ({batch_size/(total_time/1000):.2f} FPS) - Input Image Size: {config['input_size'][1:]}")

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Zongnan Bao Capstone Project')
	parser.add_argument('--config', type=str, required=True, help='path to config file')
	parser.add_argument('--input_size', nargs='+', type=int,required=True, help='input size for perf eval, such as --input_size 500 300 stands for (1,3,500,300)')
	parser.add_argument('--device', type=str, default="", help="'cpu' or 'cuda' to overwrite settings in config")

	# convert argparsed args to dictionary 
	args = vars(parser.parse_args())
	prioritized_device = args['device']
	args['input_size'] = (1, 3, int(args['input_size'][0]), int(args['input_size'][1]))

	# read & converts yaml document
	with open(args['config'], 'r') as f:
		try:
			config = yaml.safe_load(f)
		except yaml.YAMLError as error:
			print(error)

	# merge args with config files (args have higher priority)
	config.update(args)
	# manually overwrite device
	if prioritized_device:
		config['device'] = prioritized_device

	pprint(config)

	perf(config=config)
