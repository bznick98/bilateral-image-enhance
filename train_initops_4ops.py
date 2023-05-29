import os
import yaml
import torch
import numpy as np
from pprint import pprint
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from loss import CustomLoss, ZeroReferenceLoss
from models.bilateral_neuralops.networks import ColorBilateralRenderer, ColorV2BilateralRenderer
from utils import save_model, load_model, set_seed, save_tensor, hstack_tensors
from utils import get_dataset, get_model, eval
from utils import calc_dE, calc_lpips, calc_psnr, calc_ssim


def train(config):
	os.makedirs(os.path.join(config['checkpoint_save_path'], config['name']), exist_ok=True)

	# check device
	if config['device']:
		device = torch.device(config['device'])
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device [{device}]")

	# tensorboard visualization
	tb = SummaryWriter(comment=config['name'])

	# load dataset
	train_set, val_set, test_set = get_dataset(config)
	train_loader = DataLoader(train_set, batch_size=config['bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
	test_loader = DataLoader(test_set, batch_size=config['bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

	if config['model']['name'] == "color_bilateral_neuralops":
		model = ColorBilateralRenderer(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample']
		)
	elif config['model']['name'] == "colorv2_bilateral_neuralops":
		model = ColorV2BilateralRenderer(
			n_in=config['model']['n_in'],
			n_out=config['model']['n_out'],
			lowres=config['model']['lowres'],
			luma_bins=config['model']['luma_bins'],
			spatial_bins=config['model']['spatial_bins'],
			channel_multiplier=config['model']['channel_multiplier'],
			guide_pts=config['model']['guide_pts'],
			norm=config['model']['batch_norm'],
			iteratively_upsample=config['model']['iteratively_upsample']
		)
	else:
		raise NotImplementedError()
	
	summary(model, [(1,3,500,300), (1,3,500,300), (1,3,500,300), (1,3,500,300), (1,), (1,), (1,), (1,)])
	model.to(device)


	# loss function
	# loss = torch.nn.SmoothL1Loss()#torch.nn.SmoothL1Loss()#L2LOSS()#torch.nn.SmoothL1Loss()#
	# loss = CustomLoss(
	# 		patch_size=config['loss']['patch_size'], E=config['loss']['E'],
	# 		W_TV=config['loss']['W_TV'], W_col=config['loss']['W_col'], W_spa=config['loss']['W_spa'], W_exp=config['loss']['W_exp'], 
	# 		W_L2=config['loss']['W_L2'], W_L1=config['loss']['W_L1'], W_cos=config['loss']['W_cos']
	# 	)
	
	loss = torch.nn.L1Loss()

	# optimizer
	optimizer = Adam(model.parameters(),
					config['lr'],
					betas=(config['beta1'], config['beta2']),
					weight_decay=config['weight_decay'])
	# scheduler = lr_scheduler.MultiStepLR(optimizer,
	# 										milestones=[config['epochs'] // 2 + i * 10 for i in range(10)],
	# 										gamma=0.5)

	# set seed
	set_seed(config['seed'])

	for e in range(config['epochs']):
		model.train()
		# scheduler.step()

		psnr_list = []
		ssim_list = []
		mae_list = []
		lpips_list = []

		epoch_loss = []

		# training
		with tqdm(train_loader, unit="batch") as tepoch:
			for i, batch in enumerate(tepoch):
				A_ex = batch['A_ex'].to(device) 
				A_bc = batch['A_bc'].to(device) 
				A_wb = batch['A_wb'].to(device)
				A_vb = batch['A_vb'].to(device)
				val_ex = batch['val_ex'].to(device) 
				val_bc = batch['val_bc'].to(device) 
				val_wb = batch['val_wb'].to(device) 
				val_vb = batch['val_vb'].to(device)
				B_ex = batch['B_ex'].to(device) 
				B_bc = batch['B_bc'].to(device) 
				B_wb = batch['B_wb'].to(device) 
				B_vb = batch['B_vb'].to(device)
			
				optimizer.zero_grad()

				# run inference
				rec_ex, rec_bc, rec_wb, rec_vb, map_ex, map_bc, map_wb, map_vb = model(
					A_ex, A_bc, A_wb, A_vb, val_ex, val_bc, val_wb, val_vb
				)
				
				# compute loss
				loss_unary = loss(rec_ex, A_ex) + loss(rec_bc, A_bc) + loss(rec_wb, A_wb) + loss(rec_vb, A_vb)
				loss_pair = loss(map_ex, B_ex) + loss(map_bc, B_bc) + loss(map_wb, B_wb) + loss(map_vb, B_vb)
				batch_loss = loss_unary + loss_pair
				batch_loss.backward()
				
				optimizer.step()

				tepoch.set_postfix(epoch=f"{e+1}/{config['epochs']}", loss=f"{batch_loss.cpu().detach().numpy():.3f}")
				epoch_loss.append(batch_loss.cpu().detach().numpy())

		# evaluation
		model.eval()
		visualization_indices = list(range(10))
		with torch.no_grad():
			for i, batch in enumerate(test_loader):
				if i not in visualization_indices: continue

				A_ex = batch['A_ex'].to(device) 
				A_bc = batch['A_bc'].to(device) 
				A_wb = batch['A_wb'].to(device)
				A_vb = batch['A_vb'].to(device)
				val_ex = batch['val_ex'].to(device) 
				val_bc = batch['val_bc'].to(device) 
				val_wb = batch['val_wb'].to(device) 
				val_vb = batch['val_vb'].to(device)
				B_ex = batch['B_ex'].to(device) 
				B_bc = batch['B_bc'].to(device) 
				B_wb = batch['B_wb'].to(device) 
				B_vb = batch['B_vb'].to(device)

				# run inference
				rec_ex, rec_bc, rec_wb, rec_vb, map_ex, map_bc, map_wb, map_vb = model(
					A_ex, A_bc, A_wb, A_vb, val_ex, val_bc, val_wb, val_vb
				)

				rec_ex, rec_bc, rec_vb = torch.clamp(rec_ex, 0, 1), torch.clamp(rec_bc, 0, 1), torch.clamp(rec_vb, 0, 1)
				map_ex, map_bc, map_vb = torch.clamp(map_ex, 0, 1), torch.clamp(map_bc, 0, 1), torch.clamp(map_vb, 0, 1)

				psnr_list.append((calc_psnr(map_ex.cpu(), B_ex.cpu()) + calc_psnr(rec_ex.cpu(), A_ex.cpu()))/2)
				ssim_list.append((calc_ssim(map_ex.cpu(), B_ex.cpu()) + calc_ssim(rec_ex.cpu(), A_ex.cpu()))/2)
				mae_list.append((calc_dE(map_ex.cpu(), B_ex.cpu()) + calc_dE(rec_ex.cpu(), A_ex.cpu()))/2)
				lpips_list.append((calc_lpips(map_ex.cpu(), B_ex.cpu()) + calc_lpips(rec_ex.cpu(), A_ex.cpu()))/2)

				if e % config['visualize_interval'] == 0:
					out_dir = os.path.join(config['visualization_save_path'], config['name'], f"{e}-Epoch-Viz")
					out_dir_i = os.path.join(out_dir, str(i))
					if out_dir_i and not os.path.exists(out_dir_i):
						os.makedirs(out_dir_i)
					# save input, enhanced and ref image to disk
					save_tensor(hstack_tensors(rec_ex[0], A_ex[0], torch.mean(torch.abs(A_ex[0] - rec_ex[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "rec_A_ex.jpg"))
					save_tensor(hstack_tensors(rec_bc[0], A_bc[0], torch.mean(torch.abs(A_bc[0] - rec_bc[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "rec_A_bc.jpg"))
					save_tensor(hstack_tensors(rec_wb[0], A_wb[0], torch.mean(torch.abs(A_wb[0] - rec_wb[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "rec_A_wb.jpg"))
					save_tensor(hstack_tensors(rec_vb[0], A_vb[0], torch.mean(torch.abs(A_vb[0] - rec_vb[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "rec_A_vb.jpg"))

					save_tensor(hstack_tensors(map_ex[0], B_ex[0], torch.mean(torch.abs(B_ex[0] - map_ex[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "map_B_ex.jpg"))
					save_tensor(hstack_tensors(map_bc[0], B_bc[0], torch.mean(torch.abs(B_bc[0] - map_bc[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "map_B_bc.jpg"))
					save_tensor(hstack_tensors(map_wb[0], B_wb[0], torch.mean(torch.abs(B_wb[0] - map_wb[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "map_B_wb.jpg"))
					save_tensor(hstack_tensors(map_vb[0], B_vb[0], torch.mean(torch.abs(B_vb[0] - map_vb[0]), dim=0).repeat(3,1,1)), os.path.join(out_dir_i, "map_B_vb.jpg"))

			# save model every epoch
			save_filename = "ckpt_" + str(e) + ".pth"
			save_path = os.path.join(config['checkpoint_save_path'], config['name'], save_filename)
			save_model(model.state_dict(), save_path)

			print(f"Init Model: {e+1}/{config['epochs']} in {save_path} \nAverage PSNR = {np.mean(psnr_list):.3f} | SSIM = {np.mean(ssim_list):.3f} | MAE = {np.mean(mae_list):.3f} | LPIPS = {np.mean(lpips_list):.3f}")

			# # validate (save test image every visualize_interval)
			# if e % config['visualize_interval'] == 0:
			# 	out_dir = os.path.join(config['visualization_save_path'], config['name'], f"{e}-Epoch-Viz")
			# else:
			# 	out_dir = None
			# psnr, ssim, mae, lpips = eval(model, test_set, device, out_dir=out_dir)
			# print(f"Testing model: {e+1}/{config['epochs']} in {config['checkpoint_save_path']} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f} | LPIPS = {lpips:.3f}")

			# add info on tensorboard
			tb.add_scalar("Loss", np.mean(epoch_loss), e)
			tb.add_scalar("PSNR", np.mean(psnr_list), e)
			tb.add_scalar("SSIM", np.mean(ssim_list), e)
			tb.add_scalar("MAE", np.mean(mae_list), e)
			tb.add_scalar("LPIPS", np.mean(lpips_list), e)

	tb.close()

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

	train(config=config)
