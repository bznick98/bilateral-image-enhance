import os
import yaml
import torch
from pprint import pprint
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from loss import CustomLoss, ZeroReferenceLoss
from utils import save_model, load_model
from utils import get_base_name, get_timecode
from utils import get_dataset, get_model, eval


def train(config):
	os.makedirs(os.path.join(config['checkpoint_save_path'], config['name']), exist_ok=True)

	# check device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")

	# tensorboard visualization
	tb = SummaryWriter(comment=config['name'])

	# load dataset
	train_set, val_set, test_set = get_dataset(config)
	train_loader = DataLoader(train_set, batch_size=config['bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)

	# model initiation
	model = get_model(config)
	model.to(device)

	# loss function
	# loss = torch.nn.SmoothL1Loss()#torch.nn.SmoothL1Loss()#L2LOSS()#torch.nn.SmoothL1Loss()#
	loss = CustomLoss(
			patch_size=config['loss']['patch_size'], E=config['loss']['E'],
			W_TV=config['loss']['W_TV'], W_col=config['loss']['W_col'], W_spa=config['loss']['W_spa'], W_exp=config['loss']['W_exp'], 
			W_L2=config['loss']['W_L2'], W_L1=config['loss']['W_L1'], W_cos=config['loss']['W_cos']
		)

	# optimizer
	optimizer = Adam(model.parameters(), config['lr'], amsgrad=True)
	# scheduler = lr_scheduler.MultiStepLR(optimizer,
	# 										milestones=[config['epochs'] // 2 + i * 10 for i in range(10)],
	# 										gamma=0.5)

	# training
	for e in range(config['epochs']):
		model.train()
		# scheduler.step()

		with tqdm(train_loader, unit="batch") as tepoch:
			for i, (lowres, img, target) in enumerate(tepoch):
				optimizer.zero_grad()

				lowres = lowres.to(device)
				img = img.to(device)
				target = target.to(device)

				# out, coeff = model(lowres, img)
				out = model(lowres, img)
				
				batch_loss = loss(out, target)
				# batch_loss = loss(target, out, img, coeff)
				batch_loss.backward()
				
				optimizer.step()

				tepoch.set_postfix(epoch=f"{e+1}/{config['epochs']}", loss=f"{batch_loss.cpu().detach().numpy():.3f}")

		# save model every epoch
		save_filename = "ckpt_" + str(e) + ".pth"
		save_path = os.path.join(config['checkpoint_save_path'], config['name'], save_filename)
		save_model(model.state_dict(), save_path, config)

		# validate (save test image every visualize_interval)
		if e % config['visualize_interval'] == 0:
			out_dir = os.path.join(config['visualization_save_path'], config['name'], f"{e}-Epoch-Viz")
		else:
			out_dir = None
		psnr, ssim, mae, lpips = eval(model, test_set, device, out_dir=out_dir)
		print(f"Testing model: {e+1}/{config['epochs']} in {config['checkpoint_save_path']} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f} | LPIPS = {lpips:.3f}")

		# add info on tensorboard
		tb.add_scalar("Loss", batch_loss.cpu().detach().numpy(), e)
		tb.add_scalar("PSNR", psnr, e)
		tb.add_scalar("SSIM", ssim, e)
		tb.add_scalar("MAE", mae, e)
		tb.add_scalar("LPIPS", lpips, e)

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
