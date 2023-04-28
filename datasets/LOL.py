import sys
import glob
from natsort import natsorted
sys.path.append("..")
from datasets.Base import BaseDataset


class LOL_Dataset(BaseDataset):
	"""
	LOL dataset file structure:
		dataset_dir/	# base_dir
			our485/			# train set
				low/			# images
				high/			# targets
			eval15/			# test set
				low/			# images
				high/			# targets
	"""
	def __init__(self, image_dir, resize=None, augment=False, low_res=None):
		super().__init__(image_dir, resize, augment, low_res)
		
	def extract_image_pairs(self, dataset_dir):
		"""
		extract images paired with corresponding reference images 
		return:
			list of tuple paths, 
				e.g. [(image1, target1), (image2, target2), ...]
		"""
		image_list = glob.glob(dataset_dir + "low/**.png", recursive=True)
		image_list = natsorted(image_list)
		target_list = glob.glob(dataset_dir + "high/**.png", recursive=True)
		target_list = natsorted(target_list)

		return list(zip(image_list, target_list))
	

if __name__ == "__main__":
	# Example of how to use this dataset
	train_dir = "/home/ppnk-wsl/capstone/Dataset/LOL/our485/"
	test_dir = "/home/ppnk-wsl/capstone/Dataset/LOL/eval15/"

	train_set = LOL_Dataset(train_dir)
	test_set = LOL_Dataset(test_dir)
