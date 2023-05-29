import numpy as np
import cv2
import os
import glob
import torch
import random
import time
import skimage
import math

from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary
from skimage.color import rgb2lab, deltaE_cie76
from skimage.metrics import structural_similarity


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
	if config:
		state['config'] = config
	torch.save(state, path)
	return state

def load_model(path):
	"""
	return (state, config)
	"""
	state = torch.load(path)
	try:
		config = state['config']
		del state['config']
	except:
		config = {}
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
	os.environ['PYTHONHASHSEED']=str(seed)

def check_and_rotate(img1, img2):
	"""
	rotate the first image ([0-1] ndarray) if two images have unmatched orientations
		e.g. img1 is in landscape orientation, img2 is in portrait orientation
		rotate img1 to have same orientation to img2
	"""
	if img1.shape != img2.shape:
		return skimage.transform.rotate(img1, -90, resize=True)
	return img1

def get_base_name(path):
	return os.path.basename(os.path.normpath(path))
	
def get_timecode():
	  return datetime.now().strftime("%y%m%d_%H%M")

def get_model(config, print_summary=True):
	"""
	based on config file, return the corresponding model
	"""
	name = config['model']['name'].lower()

	if name == "hdrnet":
		from models.hdrnet.DeepBilateralNetCurves import DeepBilateralNetCurves
		model = DeepBilateralNetCurves(
			lowres=config['model']['low_res'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			iteratively_upsample=config['model']['iteratively_upsample']
		)
		if print_summary: summary(model, [(1, 3, 500, 300), (1, 3, 256, 256)])

	elif name == "zero_dce":
		from models.zero_dce.zero_dce import ZERO_DCE_EXT
		model = ZERO_DCE_EXT(
			scale_factor=config['model']['scale_factor']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "neuralops":
		from models.neuralops.networks import NeurOP
		model = NeurOP(
			in_nc=config['model']['in_nc'],
			out_nc=config['model']['out_nc'],
			base_nf=config['model']['base_nf'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "bilateral_neuralops":
		from models.bilateral_neuralops.networks import BilateralNeurOP
		model = BilateralNeurOP(
			in_nc=config['model']['in_nc'],
			out_nc=config['model']['out_nc'],
			base_nf=config['model']['base_nf'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "simple_bilateral_neuralops":
		from models.bilateral_neuralops.networks import SimpleBilateralNeurOP
		model = SimpleBilateralNeurOP(
			in_nc=config['model']['in_nc'],
			out_nc=config['model']['out_nc'],
			base_nf=config['model']['base_nf'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "adaptive_bilateral_neuralops":
		from models.bilateral_neuralops.networks import AdaptiveBilateralNeurOP
		model = AdaptiveBilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "color_bilateral_neuralops":
		from models.bilateral_neuralops.networks import ColorBilateralNeurOP
		model = ColorBilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))
	
	elif name == "torch_bilateral_neuralops":
		from models.bilateral_neuralops.networks import TorchBilateralNeurOP
		model = TorchBilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "sm_color_bilateral_neuralops":
		from models.bilateral_neuralops.networks import SMColorBilateralNeurOP
		model = SMColorBilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "sm_bilateral_neuralops":
		from models.bilateral_neuralops.networks import SMBilateralNeurOP
		model = SMBilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "smv2_bilateral_neuralops":
		from models.bilateral_neuralops.networks import SMV2BilateralNeurOP
		model = SMV2BilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	elif name == "colorv2_bilateral_neuralops":
		from models.bilateral_neuralops.networks import ColorV2BilateralNeurOP
		model = ColorV2BilateralNeurOP(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			encode_nf=config['model']['encode_nf'],
			load_path=config['model']['load_path'],
			return_vals=config['model']['return_vals'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample'],
			order=config['model']['order']
		)
		if print_summary: summary(model, (1, 3, 500, 300))

	else:
		raise NotImplementedError("Please add model initiation in utils.py:get_model()")
	
	# if specified pretrained model path, load from that
	if config['resume_from']:
		try:
			print(f"{name} loading from: {config['resume_from']}")
			# state_dict, pretrain_config = load_model(config['resume_from'])
			state_dict = torch.load(config['resume_from'])
			model.load_state_dict(state_dict)
		except Exception as e:
			print(f"[Error Msg] Failed to load pretrain model from {config['resume_from']}, initialized from scratch. | {e}")

	return model

def get_dataset(config):
	"""
	given configuration, return train_set, val_set, test_set
	"""
	set_seed(config['seed'])

	dataset_name = config['dataset'].upper()
	if dataset_name == "SICE":
		from datasets.SICE import SICE_Dataset
		train_dir = os.path.join(config['data_dir'], "Dataset_Part1/")
		test_dir = os.path.join(config['data_dir'], "Dataset_Part2/")
		train_set = SICE_Dataset(train_dir, augment=config['augment'], low_res=config['low_res'])
		train_set, val_set = random_split(train_set, [0.8015, 0.1985])
		test_set = SICE_Dataset(test_dir, under_expose_only=True, resize=(900, 1200), low_res=config['model']['low_res'])

	elif dataset_name == "LOL":
		from datasets.LOL import LOL_Dataset
		train_dir = os.path.join(config['data_dir'], "our485/")
		test_dir = os.path.join(config['data_dir'], "eval15/")
		train_set = LOL_Dataset(train_dir, augment=config['augment'], resize=config['resize'], low_res=config['model']['low_res'])
		val_set = None
		test_set = LOL_Dataset(test_dir, low_res=config['low_res'])

	elif dataset_name == "VELOL":
		from datasets.VE_LOL import VELOL_Dataset
		train_dir = os.path.join(config['data_dir'], "Train/")
		test_dir = os.path.join(config['data_dir'], "Test/")
		train_set = VELOL_Dataset(train_dir, augment=config['augment'], resize=config['resize'], low_res=config['low_res'])
		val_set = None
		test_set = VELOL_Dataset(test_dir, low_res=config['low_res'])

	elif dataset_name == "FIVEK":
		from datasets.FiveK import FiveK_Dataset
		data_dir = config['data_dir']
		train_set = FiveK_Dataset(data_dir, train=True, augment=config['augment'], resize=config['resize'], low_res=config['low_res'])
		val_set = None
		test_set = FiveK_Dataset(data_dir, train=False, low_res=config['low_res'])

	elif dataset_name == "NEURALOPS_INIT":
		from datasets.FiveK import InitDataset
		data_dir = config['data_dir']
		train_set = InitDataset(data_dir, config['augment'])
		generator = torch.Generator().manual_seed(config['seed'])
		_, test_set = random_split(train_set, [0.9, 0.1], generator=generator)
		# test_set = train_set
		val_set = None

	elif dataset_name == "NEURAL4OPS_INIT":
		from datasets.FiveK import Init4OpsDataset
		data_dir = config['data_dir']
		train_set = Init4OpsDataset(data_dir, config['augment'])
		generator = torch.Generator().manual_seed(config['seed'])
		_, test_set = random_split(train_set, [0.9, 0.1], generator=generator)
		# test_set = train_set
		val_set = None

	else:
		raise NotImplementedError(f"{config['dataset']} not implemented! Please implement the dataset class in datasets/")

	return train_set, val_set, test_set

def metric_preprocess(img1, img2, mono=False):
	"""
	@input:
		img1, img2: CHW/HWC/HW, torch.Tensor or np.ndarray
	@output:
		img1, img2: HWC/HW, np.ndarray
	"""
	assert img1.shape == img2.shape
	if mono:
		n_channel = 1
	else:
		n_channel = 3
	# convert to numpy ndarray if input is tensor
	if torch.is_tensor(img1): img1 = img1.numpy()
	if torch.is_tensor(img2): img2 = img2.numpy()
	# convert CHW to HWC
	if len(img1.shape) == 3 and img1.shape[-1] != n_channel:
		img1 = img1.transpose(1, 2, 0)
	if len(img2.shape) == 3 and img2.shape[-1] != n_channel:
		img2 = img2.transpose(1, 2, 0)

	return img1, img2

# Metrics
def __calc_dE(img1, img2):
	img1, img2 = metric_preprocess(img1, img2)
	return np.array(deltaE_cie76(rgb2lab(img1),rgb2lab(img2))).mean()

def calc_dE(img1, img2):
	assert img1.shape == img2.shape
	if len(img1.shape) == 4:
		b,_,_,_ = img1.shape
		scores = []
		for i in range(b):
			scores.append(__calc_dE(img1[i], img2[i]))

		return np.mean(scores)

	return __calc_dE(img1, img2)

def __calc_ssim(img1, img2):
	img1, img2 = metric_preprocess(img1, img2)
	return structural_similarity(img1, img2, data_range=1.0, channel_axis=-1,multichannel=True)

def calc_ssim(img1, img2):
	assert img1.shape == img2.shape
	if len(img1.shape) == 4:
		b,_,_,_ = img1.shape
		scores = []
		for i in range(b):
			scores.append(__calc_ssim(img1[i], img2[i]))

		return np.mean(scores)

	return __calc_ssim(img1, img2)

def __calc_psnr(img1, img2):
	img1, img2 = metric_preprocess(img1, img2)
	mse = np.mean((img1 - img2)**2)
	if mse == 0:
		return float('inf')
	return -20 * math.log10(math.sqrt(mse))

def calc_psnr(img1, img2):
	assert img1.shape == img2.shape
	if len(img1.shape) == 4:
		b,_,_,_ = img1.shape
		scores = []
		for i in range(b):
			scores.append(__calc_psnr(img1[i], img2[i]))

		return np.mean(scores)

	return __calc_psnr(img1, img2)

def calc_lpips(img1, img2):
	# input has to be BCHW torch.Tensors
	lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
	return lpips(img1, img2)


# End of Metrics (TODO: move them into a separate file)

def eval(model, dataset, device, out_dir=None, profiling=False):
	"""
	given a model, evaluate the model performance (PSNR, SSIM) on the dataset
	return:
		psnr: average PSNR score across the dataset
		ssim: average SSIM score across the dataset
	"""
	# create dir for saving images
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir)

	dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

	psnr_list = []
	ssim_list = []
	mae_list = []
	lpips_list = []

	# visualize batch number if on this list
	visualize_idx = list(range(10)) + [25, 55, 115, 145, 175, 250, 275, 325, 345, 375, 455, 475]

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
			# if profiling, add required wrapper
			if profiling and i == 0:
				with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
					with record_function("model_inference"):
						enhanced = model(*input)
				print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
			else:
				enhanced = model(*input)

			# clamp enhanced
			enhanced = torch.clamp(enhanced, 0, 1)

			total_time += time.time() - start
			
			psnr_list.append(calc_psnr(enhanced.cpu().detach(), target.cpu().detach()))
			ssim_list.append(calc_ssim(enhanced.cpu().detach(), target.cpu().detach()))
			mae_list.append(calc_dE(enhanced.cpu().detach(), target.cpu().detach()))
			lpips_list.append(calc_lpips(enhanced.cpu().detach(), target.cpu().detach()))
			
			if out_dir and i in visualize_idx:
				# save input, enhanced and ref image to disk
				# save_tensor(img_full[0], os.path.join(out_dir_i, "input.jpg"))
				# save_tensor(enhanced[0], os.path.join(out_dir, f"output_{i}.jpg"))
				save_tensor(hstack_tensors(input[0][0], enhanced[0], target[0], torch.mean(torch.abs(target[0] - enhanced[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir, f"input_output_ref_{i}.jpg"))
				# save_tensor(ref[0], os.path.join(out_dir_i, "ref.jpg"))
				# save_tensor(coeff[0], os.path.join(out_dir_i, "map.jpg"), heatmap=True)


	# report time
	print(f"Total time used = {total_time:.3f} s inferencing {len(dataset)} images on [{device}] ({len(dataset) / total_time:.3f} FPS)")

	return np.mean(psnr_list), np.mean(ssim_list), np.mean(mae_list), np.mean(lpips_list)


if __name__ == "__main__":
	state = torch.load("./result/checkpoints/NeuralOps_FiveKLite_Official_Pretrain/neurop_fivek_lite.pth")
	print(state.keys())
	keys_to_delete = ['image_encoder.conv1.weight', 'image_encoder.conv1.bias', 'image_encoder.conv2.weight', 'image_encoder.conv2.bias', 'ex_predictor.fc3.weight', 'ex_predictor.fc3.bias', 'bc_predictor.fc3.weight', 'bc_predictor.fc3.bias', 'vb_predictor.fc3.weight', 'vb_predictor.fc3.bias']
	for key in keys_to_delete:
		del state[key]
	keys_to_rename = {
		'ex_renderer.encoder.weight': 'ex_block.encoder.weight',
		'ex_renderer.encoder.bias': 'ex_block.encoder.bias',
		'ex_renderer.mid_conv.weight': 'ex_block.mid_conv.weight',
		'ex_renderer.mid_conv.bias': 'ex_block.mid_conv.bias',
		'ex_renderer.decoder.weight': 'ex_block.decoder.weight',
		'ex_renderer.decoder.bias': 'ex_block.decoder.bias',
		'bc_renderer.encoder.weight': 'bc_block.encoder.weight',
		'bc_renderer.encoder.bias': 'bc_block.encoder.bias',
		'bc_renderer.mid_conv.weight': 'bc_block.mid_conv.weight',
		'bc_renderer.mid_conv.bias': 'bc_block.mid_conv.bias',
		'bc_renderer.decoder.weight': 'bc_block.decoder.weight',
		'bc_renderer.decoder.bias': 'bc_block.decoder.bias',
		'vb_renderer.encoder.weight': 'vb_block.encoder.weight',
		'vb_renderer.encoder.bias': 'vb_block.encoder.bias',
		'vb_renderer.mid_conv.weight': 'vb_block.mid_conv.weight',
		'vb_renderer.mid_conv.bias': 'vb_block.mid_conv.bias',
		'vb_renderer.decoder.weight': 'vb_block.decoder.weight',
		'vb_renderer.decoder.bias': 'vb_block.decoder.bias'
	}
	for old, new in keys_to_rename.items():
		tmp = state[old]
		del state[old]
		state[new] = tmp
	torch.save(state, "./result/checkpoints/NeuralOps_FiveKLite_Official_Pretrain/neurops_fivek_init.pth")
	# img1 = skimage.io.imread("/home/ppnk-wsl/capstone/Dataset/FiveK_Dark/testB/0065.jpg")
	# img2 = skimage.io.imread("/home/ppnk-wsl/capstone/Dataset/FiveK_Dark/testB/0108.jpg")
	# img1 = skimage.util.img_as_float(img1)
	# img2 = skimage.util.img_as_float(img2)

	# # skimage.io.imsave("loaded_1.jpg", img1)
	# # skimage.io.imsave("loaded_2.jpg", img2)

	# img1 = check_and_rotate(img1, img2)

	# # skimage.io.imsave("checked_1.jpg", img1)
	# # skimage.io.imsave("checked_2.jpg", img2)