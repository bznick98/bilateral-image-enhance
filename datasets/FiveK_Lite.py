import numpy as np
import random
import sys
import glob
from PIL import Image
from natsort import natsorted
sys.path.append("..")
from datasets.Base import BaseDataset


class FiveK_Lite_Dataset(BaseDataset):
	"""
	MIT-Adobe-5K Lite dataset file structure:
		dataset_dir/	# base_dir
			trainA/			# train images
				...         
			trainB/         # train targets
				...         
			testA/			# test images
				...
			testB/          # test targets
				...
	"""
	def __init__(self, image_dir, resize=None, augment=False, low_res=None, train=True):
		self.train = train
		super().__init__(image_dir, resize, augment, low_res)
		
	def extract_image_pairs(self, dataset_dir):
		"""
		extract images paired with corresponding reference images 
		return:
			list of tuple paths, 
				e.g. [(image1, target1), (image2, target2), ...]
		"""
		if self.train:
			image_list = glob.glob(dataset_dir + "trainA/**.jpg", recursive=True)
			target_list = glob.glob(dataset_dir + "trainB/**.jpg", recursive=True)
		else:
			image_list = glob.glob(dataset_dir + "testA/**.jpg", recursive=True)
			target_list = glob.glob(dataset_dir + "testB/**.jpg", recursive=True)

		image_list = natsorted(image_list)
		target_list = natsorted(target_list)

		return list(zip(image_list, target_list))
	
	def __getitem__(self, index):
		image_path, target_path = self.data_list[index]
		image = Image.open(image_path)
		target = Image.open(target_path)

		# augmentation
		if self.augment:
			image, target = self.aug_process(np.array(image), np.array(target))
			image = Image.fromarray(image)
			target = Image.fromarray(target)
			
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
	# Example of how to use this dataset
	img_dir = "/home/ppnk-wsl/capstone/Dataset/FiveK_Lite/"

	train_set = FiveK_Lite_Dataset(img_dir, train=True)
	test_set = FiveK_Lite_Dataset(img_dir, train=False)
