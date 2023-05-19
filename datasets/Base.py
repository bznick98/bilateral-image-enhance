import numpy as np
import random
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import check_and_rotate
import datasets.transforms as tr_custom
from skimage import io
from skimage.util import img_as_float32
from skimage.transform import resize


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

		# converts ndarray HxWxC to torch.Tensor CxHxW
		self.toTensor = transforms.Compose([
			np.ascontiguousarray,
			transforms.ToTensor()
		])

		print(f"Dataset loaded from: {image_dir}")
		print(f"# of loaded samples = {len(self.data_list)} | resize = {resize} | augment = {augment} | low_res_branch = {low_res}")
		
	def __getitem__(self, index):
		image_path, target_path = self.data_list[index]
		# read & converts image to range [0, 1]
		image = img_as_float32(io.imread(image_path))
		target = img_as_float32(io.imread(target_path))

		# check and rotate image if orientation unmatched
		image = check_and_rotate(image, target)

		# augmentation
		if self.augment:
			# NeuralOps Augment
			image, target = self.aug_process(image, target)

			# deep_bilateral_network Augment
			# image, target = tr_custom.random_horizontal_flip([image, target])
			# image, target = tr_custom.random_rotation([image, target], angle=15)
			# image, target = tr_custom.random_crop([image, target], scale=(0.8, 1.0),
			# 									  aspect_ratio=(image.size[1]/image.size[0]))
		
		# add additional low-res branch (for models like HDRNet)
		if self.low_res:
			image_lowres = resize(image, self.low_res)
			image_lowres = self.toTensor(image_lowres)
			image = self.toTensor(image)
			target = self.toTensor(target)

			return image_lowres, image, target
		
		# to torch.Tensor
		image = self.toTensor(image)
		target = self.toTensor(target)
		
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
	
	def aug_process(self, img_GT, img_LQ, img_M=None):
		"""
		from NeuralOps paper: https://github.com/amberwangyili/neurop/blob/main/codes_pytorch/utils.py
		"""
		h, w = img_GT.shape[:2]
		crop_size = 20
		new_h = random.randint(h - crop_size, h - 1)
		new_w = random.randint(w - crop_size, w - 1)

		y = random.randint(0, h - new_h - 1)
		x = random.randint(0, w - new_w - 1)

		img_GT = img_GT[y:y+new_h, x:x+new_w,:]
		img_LQ = img_LQ[y:y+new_h, x:x+new_w,:]
		if img_M is not None:
			img_M = img_M[y:y+new_h, x:x+new_w]

		is_flip = random.randint(0,3)
		if is_flip == 0:
			img_GT = np.flip(img_GT, axis=0)
			img_LQ = np.flip(img_LQ, axis=0)
			if img_M is not None:
				img_M = np.flip(img_M,axis=0)
		elif is_flip == 2:
			img_GT = np.flip(img_GT, axis=1)
			img_LQ = np.flip(img_LQ, axis=1)
			if img_M is not None:
				img_M = np.flip(img_M, axis=1)
		is_rot = random.randint(0,3)
		if is_rot !=0:
			if img_M is not None:
				img_M = np.rot90(img_M, is_rot)
			img_GT = np.rot90(img_GT, is_rot)
			img_LQ = np.rot90(img_LQ, is_rot)
		if img_M is not None:
			return img_GT, img_LQ, img_M
		else:
			return img_GT, img_LQ
		
		
if __name__ == "__main__":
	pass
