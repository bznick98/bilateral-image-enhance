import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import check_and_rotate
import datasets.transforms as tr_custom


class BaseDataset(Dataset):
	def __init__(self, image_dir, resize=None, augment=False, low_res=None):
		"""
		@input:
			- image_dir: directory to image dataset, used as base dir for extract_image_pairs()
			- resize: default=None, if specified, loaded image (and the target) will be resized to (C,resize[0],resize[1])
			- augment: default=False, if enabled, loaded image (and the target) will be augmented
			- low_res: default=None, if specified, a low-resolution input image will be generated in addition to loaded image and target (HDRNet requires a low-res input)
		"""
		# each child dataset needs to implement self.extract_image_pairs()
		self.data_list = self.extract_image_pairs(image_dir)
		
		self.resize = resize
		self.augment = augment
		self.low_res = low_res

		# basic image pre-processing (resize + convert to torch.Tensor)
		self.preprocess = [
			transforms.Resize(resize, Image.BICUBIC) if resize else lambda x:x,
			transforms.ToTensor()
		]
		self.preprocess = transforms.Compose(self.preprocess)

		if low_res:
			self.to_lowres = transforms.Compose([
				transforms.Resize(low_res, Image.BICUBIC),
				transforms.ToTensor()
			])

		print(f"Dataset loaded from: {image_dir}")
		print(f"# of loaded samples = {len(self.data_list)} | resize = {resize} | augment = {augment} | low_res_branch = {low_res}")
		
	def __getitem__(self, index):
		image_path, target_path = self.data_list[index]
		image = Image.open(image_path)
		target = Image.open(target_path)

		# check and rotate image if orientation unmatched
		image = check_and_rotate(image, target)

		# augmentation
		if self.augment:
			image, target = tr_custom.random_horizontal_flip([image, target])
			image, target = tr_custom.random_rotation([image, target], angle=15)
			# image, target = tr_custom.random_crop([image, target], scale=(0.8, 1.0),
			# 									  aspect_ratio=(image.size[1]/image.size[0]))
		
		# add additional low-res branch (for models like HDRNet)
		if self.low_res:
			image_lowres = self.to_lowres(image)
			image = self.preprocess(image)
			target = self.preprocess(target)
			return image_lowres, image, target
		
		# resize and to torch.Tensor
		image = self.preprocess(image)
		target = self.preprocess(target)
		
		return image, target

	def __len__(self):
		return len(self.data_list)

	def extract_image_pairs(self, dataset_dir):
		"""
		extract images paired with corresponding reference images 
		return:
			list of tuple paths, 
				e.g. [(image1, target1), (image2, target2), ...]
		"""
		raise NotImplementedError("Implement this method in your specific dataset class.")
	

if __name__ == "__main__":
	pass
