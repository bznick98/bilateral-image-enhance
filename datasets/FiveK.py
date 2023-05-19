import numpy as np
import random
import torch
import sys, os
import glob
from PIL import Image
from natsort import natsorted
sys.path.append("..")
from datasets.Base import BaseDataset
from torch.utils.data import Dataset
from skimage import io
from collections import defaultdict


class FiveK_Dataset(BaseDataset):
	"""
	MIT-Adobe-5K Lite/Dark dataset file structure:
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
			image_list = glob.glob(dataset_dir + "trainA/**.**", recursive=True)
			target_list = glob.glob(dataset_dir + "trainB/**.jpg", recursive=True)
		else:
			image_list = glob.glob(dataset_dir + "testA/**.**", recursive=True)
			target_list = glob.glob(dataset_dir + "testB/**.jpg", recursive=True)

		image_list = natsorted(image_list)
		target_list = natsorted(target_list)

		return list(zip(image_list, target_list))
	

class InitDataset(Dataset):
	def __init__(self, image_dir, augment=False):
		super(InitDataset,self).__init__()
		self.augment = augment

		filepath_EX = self.get_file_paths(os.path.join(image_dir,'EX'),'png')
		filepath_BC = self.get_file_paths(os.path.join(image_dir,'BC'),'png')
		filepath_VB = self.get_file_paths(os.path.join(image_dir,'VB'),'png')

		self.file_ex = defaultdict(list)
		self.file_bc = defaultdict(list)
		self.file_vb = defaultdict(list)

		for f_ex,f_bc,f_vb in zip(filepath_EX,filepath_BC,filepath_VB):
			idx_ex = f_ex.split('/')[-1].split('-')[0]
			idx_bc = f_bc.split('/')[-1].split('-')[0]
			idx_vb = f_vb.split('/')[-1].split('-')[0]
			self.file_ex[idx_ex].append(f_ex)
			self.file_bc[idx_bc].append(f_bc)
			self.file_vb[idx_vb].append(f_vb)

		self.file_keys = list(self.file_ex.keys())
	def __len__(self):
		return len(self.file_keys)
	def __getitem__(self, index):       
		key = self.file_keys[index]    
		A_ex, B_ex = np.random.choice(self.file_ex[key],2,replace=False)
		A_bc, B_bc = np.random.choice(self.file_bc[key],2,replace=False)
		A_vb, B_vb = np.random.choice(self.file_vb[key],2,replace=False)
		
		val_ex = torch.tensor((int(self.get_file_name(B_ex).split('-')[-1]) - int(self.get_file_name(A_ex).split('-')[-1]))/20).float()
		val_bc = torch.tensor((int(self.get_file_name(B_bc).split('-')[-1]) - int(self.get_file_name(A_bc).split('-')[-1]))/20).float()
		val_vb = torch.tensor((int(self.get_file_name(B_vb).split('-')[-1]) - int(self.get_file_name(A_vb).split('-')[-1]))/20).float()

		img_A_ex = np.array(io.imread(A_ex))/255
		img_B_ex = np.array(io.imread(B_ex))/255

		img_A_bc = np.array(io.imread(A_bc))/255
		img_B_bc = np.array(io.imread(B_bc))/255

		img_A_vb = np.array(io.imread(A_vb))/255
		img_B_vb = np.array(io.imread(B_vb))/255

		if self.augment:
			img_A_ex, img_B_ex = self.aug_process(img_A_ex, img_B_ex)
			img_A_bc, img_B_bc = self.aug_process(img_A_bc, img_B_bc)
			img_A_vb, img_B_vb = self.aug_process(img_A_vb, img_B_vb)

		img_A_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_ex, (2, 0, 1)))).float()
		img_B_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_ex, (2, 0, 1)))).float()

		img_A_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_bc, (2, 0, 1)))).float()
		img_B_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_bc, (2, 0, 1)))).float()

		img_A_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_vb, (2, 0, 1)))).float()
		img_B_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_vb, (2, 0, 1)))).float()

		return {'A_ex': img_A_ex, 'B_ex': img_B_ex, 'val_ex':val_ex, 
				'A_bc': img_A_bc, 'B_bc': img_B_bc, 'val_bc':val_bc, 
				'A_vb': img_A_vb, 'B_vb': img_B_vb, 'val_vb':val_vb 
			   } 

	def get_file_paths(self, folder,suffix):
		file_paths = []
		for root, dirs, filenames in os.walk(folder):
			filenames = sorted(filenames)
			for filename in filenames:
				input_path = os.path.abspath(root)
				file_path = os.path.join(input_path, filename)
				if filename.split('.')[-1] == suffix:
					file_paths.append(file_path)
			break  
		return file_paths

	def get_file_name(self, fp):
		return fp.split('/')[-1].split('.')[0]

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


class Init4OpsDataset(Dataset):
	def __init__(self, image_dir, augment=False):
		super(Init4OpsDataset,self).__init__()
		self.augment = augment

		filepath_EX = self.get_file_paths(os.path.join(image_dir,'EX'),'png')
		filepath_BC = self.get_file_paths(os.path.join(image_dir,'BC'),'png')
		filepath_WB = self.get_file_paths(os.path.join(image_dir,'WB'),'png')
		filepath_VB = self.get_file_paths(os.path.join(image_dir,'VB'),'png')

		self.file_ex = defaultdict(list)
		self.file_bc = defaultdict(list)
		self.file_wb = defaultdict(list)
		self.file_vb = defaultdict(list)

		for f_ex,f_bc,f_wb,f_vb in zip(filepath_EX,filepath_BC,filepath_WB,filepath_VB):
			idx_ex = f_ex.split('/')[-1].split('-')[0]
			idx_bc = f_bc.split('/')[-1].split('-')[0]
			idx_wb = f_wb.split('/')[-1].split('-')[0]
			idx_vb = f_vb.split('/')[-1].split('-')[0]
			self.file_ex[idx_ex].append(f_ex)
			self.file_bc[idx_bc].append(f_bc)
			self.file_wb[idx_wb].append(f_wb)
			self.file_vb[idx_vb].append(f_vb)

		self.file_keys = list(self.file_ex.keys())
	def __len__(self):
		return len(self.file_keys)
	def __getitem__(self, index):       
		key = self.file_keys[index]    
		A_ex, B_ex = np.random.choice(self.file_ex[key],2,replace=False)
		A_bc, B_bc = np.random.choice(self.file_bc[key],2,replace=False)
		A_wb, B_wb = np.random.choice(self.file_wb[key],2,replace=False)
		A_vb, B_vb = np.random.choice(self.file_vb[key],2,replace=False)
		
		val_ex = torch.tensor((int(self.get_file_name(B_ex).split('-')[-1]) - int(self.get_file_name(A_ex).split('-')[-1]))/20).float()
		val_bc = torch.tensor((int(self.get_file_name(B_bc).split('-')[-1]) - int(self.get_file_name(A_bc).split('-')[-1]))/20).float()
		val_wb = torch.tensor((int(self.get_file_name(B_wb).split('-')[-1]) - int(self.get_file_name(A_wb).split('-')[-1]))/20).float()
		val_vb = torch.tensor((int(self.get_file_name(B_vb).split('-')[-1]) - int(self.get_file_name(A_vb).split('-')[-1]))/20).float()

		img_A_ex = np.array(io.imread(A_ex))/255
		img_B_ex = np.array(io.imread(B_ex))/255

		img_A_bc = np.array(io.imread(A_bc))/255
		img_B_bc = np.array(io.imread(B_bc))/255

		img_A_wb = np.array(io.imread(A_wb))/255
		img_B_wb = np.array(io.imread(B_wb))/255

		img_A_vb = np.array(io.imread(A_vb))/255
		img_B_vb = np.array(io.imread(B_vb))/255

		if self.augment:
			img_A_ex, img_B_ex = self.aug_process(img_A_ex, img_B_ex)
			img_A_bc, img_B_bc = self.aug_process(img_A_bc, img_B_bc)
			img_A_wb, img_B_wb = self.aug_process(img_A_wb, img_B_wb)
			img_A_vb, img_B_vb = self.aug_process(img_A_vb, img_B_vb)

		img_A_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_ex, (2, 0, 1)))).float()
		img_B_ex = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_ex, (2, 0, 1)))).float()

		img_A_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_bc, (2, 0, 1)))).float()
		img_B_bc = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_bc, (2, 0, 1)))).float()

		img_A_wb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_wb, (2, 0, 1)))).float()
		img_B_wb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_wb, (2, 0, 1)))).float()

		img_A_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_A_vb, (2, 0, 1)))).float()
		img_B_vb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_B_vb, (2, 0, 1)))).float()

		return {'A_ex': img_A_ex, 'B_ex': img_B_ex, 'val_ex':val_ex, 
				'A_bc': img_A_bc, 'B_bc': img_B_bc, 'val_bc':val_bc, 
				'A_wb': img_A_wb, 'B_wb': img_B_wb, 'val_wb':val_wb, 
				'A_vb': img_A_vb, 'B_vb': img_B_vb, 'val_vb':val_vb 
			   } 

	def get_file_paths(self, folder,suffix):
		file_paths = []
		for root, dirs, filenames in os.walk(folder):
			filenames = sorted(filenames)
			for filename in filenames:
				input_path = os.path.abspath(root)
				file_path = os.path.join(input_path, filename)
				if filename.split('.')[-1] == suffix:
					file_paths.append(file_path)
			break  
		return file_paths

	def get_file_name(self, fp):
		return fp.split('/')[-1].split('.')[0]

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
	img_dir = "/home/ppnk-wsl/capstone/Dataset/FiveK_Dark/"

	# from utils import set_seed, display_tensors
	# set_seed(0)
	# d1 = FiveK_Dataset(img_dir, augment=False, train=True)
	# set_seed(0)
	# d2 = FinetuneDataset({'dataroot':img_dir, 'name':'fivek_dark'}, phase="train")

	# i = 0

	# i1, t1 = d1[i][0], d1[i][1]
	# i2, t2 = d2[i]['LQ'], d2[i]['GT']

	# print(i1.shape, i2.shape)
	# print(torch.sum((i1-i2)**2), torch.sum((t1-t2)**2))

	# display_tensors(i1, t1)
	# display_tensors(i2, t2)

	# from utils import set_seed, display_tensors
	# set_seed(1000)
	# my_train = FiveK_Dataset(img_dir, train=True, augment=True)
	# orig_train = FinetuneDataset(opt={'name':'fivek_lite', 'dataroot':img_dir}, phase='train')

	# i = 14
	# my_LQ, my_GT = my_train[i]
	# orig_LQ, orig_GT = orig_train[i]['LQ'], orig_train[i]['GT']
	# print(my_GT.shape)
	# print(orig_GT.shape)
	# print(torch.mean(torch.absolute(my_LQ - orig_LQ)))
