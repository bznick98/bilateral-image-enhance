import sys
import glob
from natsort import natsorted
sys.path.append("..")
from datasets.Base import BaseDataset


class VELOL_Dataset(BaseDataset):
	"""
	VELOL dataset file structure:
		dataset_dir/	# base_dir
			Train/			# train set
				Low/			# images
				Normal/			# targets
			Test/			# test set
				Low/			# images
				Normal/			# targets
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
		image_list = glob.glob(dataset_dir + "Low/**.png", recursive=True)
		image_list = natsorted(image_list)
		target_list = glob.glob(dataset_dir + "Normal/**.png", recursive=True)
		target_list = natsorted(target_list)

		return list(zip(image_list, target_list))
	

if __name__ == "__main__":
	# Example of how to use this dataset
	train_dir = "/home/ppnk-wsl/capstone/Dataset/VE-LOL-L-Cap-Full/Train/"
	test_dir = "/home/ppnk-wsl/capstone/Dataset/VE-LOL-L-Cap-Full/Test/"

	train_set = VELOL_Dataset(train_dir)
	test_set = VELOL_Dataset(test_dir)
