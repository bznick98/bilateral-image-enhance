import numpy as np
import cv2
import os
import glob
import torch
import random
import time
import skimage

from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def resize(img, size=512, strict=False):
	short = min(img.shape[:2])
	scale = size/short
	if not strict:
		img = cv2.resize(img, (round(
			img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
	else:
		img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
	return img


def crop(img, size=512):
	try:
		y, x = random.randint(
			0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
	except Exception as e:
		y, x = 0, 0
	return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
	img = cv2.imread(filename, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if size:
		img = resize(img, size=size)
	if use_crop:
		img = crop(img, size)
	return img

def get_latest_ckpt(path):
	try:
		list_of_files = glob.glob(os.path.join(path,'*')) 
		latest_file = max(list_of_files, key=os.path.getctime)
		return latest_file
	except ValueError:
		return None

def save_model(state, path, config=None):
	state['config'] = config
	torch.save(state, path)
	return state

def load_model(path):
	"""
	return (state, config)
	"""
	state = torch.load(path)
	config = state['config']
	del state['config']
	return state, config


def display_tensors(*args):
	"""
	display multiple torch tensors using hstack
	"""
	ToPIL = transforms.ToPILImage()
	imgs = []
	for arg in args:
		imgs.append(ToPIL(arg))

	imgs = tuple(imgs)
	return Image.fromarray(np.hstack(imgs)).show()

def hstack_tensors(*args):
	"""
	hstack multiple torch CxHxW tensors into CxHxnW
	"""
	imgs = []
	for arg in args:
		imgs.append(arg.permute(1, 2, 0))

	imgs = tuple(imgs)
	imgs = torch.hstack(imgs)
	return imgs.permute(2, 0, 1)

def save_tensor(tensor, dir, heatmap=False):
	if len(tensor.shape) == 3:
		tensor = (tensor.cpu().detach().numpy()).transpose(1, 2, 0)
	else:
		raise ValueError("Tensors to be saved as image must be 3-dim.")
	tensor = skimage.exposure.rescale_intensity(tensor, out_range=(0.0, 255.0)).astype(np.uint8)
	tensor = tensor[...,::-1]
	# convert to black and white if enabled
	if heatmap:
		tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2GRAY)
		tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
	cv2.imwrite(dir, tensor)

def set_seed(seed=46):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

def check_and_rotate(img1, img2):
	"""
	rotate the first image (PIL Image) if two images have unmatched orientations
		e.g. img1 is in landscape orientation, img2 is in portrait orientation
		rotate img1 to have same orientation to img2
	"""
	if img1.size != img2.size:
		return img1.rotate(-90, expand=True)
	return img1

def get_base_name(path):
	return os.path.basename(os.path.normpath(path))
	
def get_timecode():
	  return datetime.now().strftime("%y%m%d_%H%M")

def get_model(config):
	"""
	based on config file, return the corresponding model
	"""
	name = config['model']['name'].lower()

	if name == "hdrnet":
		from models.hdrnet.DeepBilateralNetCurves import DeepBilateralNetCurves
		model = DeepBilateralNetCurves(
			lowres=config['low_res'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			iteratively_upsample=config['model']['iteratively_upsample']
		)
	elif name == "zero_dce":
		from models.zero_dce.zero_dce import ZERO_DCE_EXT
		model = ZERO_DCE_EXT(
			scale_factor=config['model']['scale_factor']
		)

	elif name == "neuralops":
		from models.neuralops.networks import NeurOP
		model = NeurOP(
			in_nc=config['model']['in_nc'],
			out_nc=config['model']['out_nc'],
			base_nf=config['model']['base_nf'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path']
		)
	else:
		raise NotImplementedError("Please add model initiation in utils.py:get_model()")
	
	# if specified pretrain path, load from that
	if config['pretrain_path']:
		try:
			print(f"{name} loading from: {config['pretrain_path']}")
			# state_dict, pretrain_config = load_model(config['pretrain_path'])
			state_dict = torch.load(config['pretrain_path'])
			model.load_state_dict(state_dict)
		except Exception as e:
			print(f"[Error Msg] Failed to load pretrain model from {config['pretrain_path']}, initialized from scratch. | {e}")

	return model

def get_dataset(config):
	"""
	given configuration, return train_set, val_set, test_set
	"""
	if config['dataset'] == "SICE":
		from datasets.SICE import SICE_Dataset
		train_dir = os.path.join(config['data_dir'], "Dataset_Part1/")
		test_dir = os.path.join(config['data_dir'], "Dataset_Part2/")
		train_set = SICE_Dataset(train_dir, augment=config['augment'], low_res=config['low_res'])
		train_set, val_set = random_split(train_set, [0.8015, 0.1985])
		test_set = SICE_Dataset(test_dir, under_expose_only=True, resize=(900, 1200), low_res=config['low_res'])

	elif config['dataset'] == "LOL":
		from datasets.LOL import LOL_Dataset
		train_dir = os.path.join(config['data_dir'], "our485/")
		test_dir = os.path.join(config['data_dir'], "eval15/")
		train_set = LOL_Dataset(train_dir, augment=config['augment'], resize=config['resize'], low_res=config['low_res'])
		val_set = None
		test_set = LOL_Dataset(test_dir, low_res=config['low_res'])

	elif config['dataset'] == "VELOL":
		from datasets.VE_LOL import VELOL_Dataset
		train_dir = os.path.join(config['data_dir'], "Train/")
		test_dir = os.path.join(config['data_dir'], "Test/")
		train_set = VELOL_Dataset(train_dir, augment=config['augment'], resize=config['resize'], low_res=config['low_res'])
		val_set = None
		test_set = VELOL_Dataset(test_dir, low_res=config['low_res'])

	elif config['dataset'] == "FiveK_Lite":
		from datasets.FiveK_Lite import FiveK_Lite_Dataset
		data_dir = config['data_dir']
		train_set = FiveK_Lite_Dataset(data_dir, train=True, augment=config['augment'], resize=config['resize'], low_res=config['low_res'])
		val_set = None
		test_set = FiveK_Lite_Dataset(data_dir, train=False, low_res=config['low_res'])

	else:
		raise NotImplementedError(f"{config['dataset']} not implemented! Please implement the dataset class in datasets/")

	return train_set, val_set, test_set

def eval(model, dataset, device, out_dir=None):
	"""
	given a model, evaluate the model performance (PSNR, SSIM) on the dataset
	return:
		psnr: average PSNR score across the dataset
		ssim: average SSIM score across the dataset
	"""
	# create dir for saving images
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir)

	set_seed(1143)
	dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

	psnr_list = []
	ssim_list = []
	mae_list = []
	lpips_list = []

	# visualize batch number if on this list
	visualize_idx = list(range(10))

	# metrics
	psnr = PeakSignalNoiseRatio()
	ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
	mae = MeanAbsoluteError() # different from MAE in Zero-DCE++ paper
	lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

	# time measure
	total_time = 0

	model.eval()
	with torch.no_grad():
		for i, batch in enumerate(tqdm(dataloader)):
			# push data to device
			for ii, b in enumerate(batch):
				batch[ii] = b.to(device)

			input = batch[:-1]
			target = batch[-1]

			start = time.time()
			# enhanced, coeff = model(img_low, img_full)
			enhanced, vals = model(*input)
			total_time += time.time() - start
			
			psnr_list.append(psnr(enhanced.cpu(), target.cpu()))
			ssim_list.append(ssim(enhanced.cpu(), target.cpu()))
			mae_list.append(mae(enhanced.cpu(), target.cpu()))
			lpips_list.append(lpips(enhanced.cpu(), target.cpu()))
			
			if out_dir and i in visualize_idx:
				# save input, enhanced and ref image to disk
				# save_tensor(img_full[0], os.path.join(out_dir_i, "input.jpg"))
				save_tensor(enhanced[0], os.path.join(out_dir, f"output_{i}.jpg"))
				# save_tensor(ref[0], os.path.join(out_dir_i, "ref.jpg"))
				# save_tensor(coeff[0], os.path.join(out_dir_i, "map.jpg"), heatmap=True)


	# report time
	print(f"Total time used = {total_time:.3f} s inferencing {len(dataset)} images on [{device}] ({len(dataset) / total_time:.3f} FPS)")

	return np.mean(psnr_list), np.mean(ssim_list), np.mean(mae_list), np.mean(lpips_list)