import sys
import glob
from natsort import natsorted
sys.path.append("..")
from datasets.Base import BaseDataset


class SICE_Dataset(BaseDataset):
	"""
	SICE dataset file structure:
		dataset_dir/	# base_dir
			Dataset_Part1/			# train set
				1/				# image 1
				2/				# image 2
				...
				label/
					1.jpg		# target 1
					...
			Dataset_Part2/			# test set
				1/				# image 1
				2/				# image 2
				...
				label/
					1.jpg		# target 1
					...
	"""
	def __init__(self, image_dir, resize=None, augment=False, low_res=None, under_expose_only=False):
		# SICE dataset has multiple exposure level for 1 input image, if under_expose_only is enabled,
		# we will grab only lower-half of the exposure levels (under-exposed)
		self.under_expose_only = under_expose_only
		super().__init__(image_dir, resize, augment, low_res)

	def extract_image_pairs(self, dataset_dir):
		"""
		extract images paired with corresponding reference images (under Label/)
		- under_expose_only: if enabled, only select under-exposed images for each scene,
			e.g. if there are 7 images, select only the first 3 images;
				if there are 9 images, select only the first 4 images.
		return:
			list of tuple paths,
				e.g. [(some-dir/1/1.jpg, some-dir/label/1.jpg), (some-dir/1/2.jpg, some-dir/label/1.jpg), ...]
		"""
		label_list = glob.glob(dataset_dir + "Label/**.**", recursive=True)
		label_list = natsorted(label_list)
		data_list = []
		for i, label_path in enumerate(label_list):
			jpgs = glob.glob(dataset_dir + str(i+1) + "/**.**")
			jpgs = natsorted(jpgs)
			if self.under_expose_only:
				# remove over-exposed images
				jpgs = jpgs[:len(jpgs)//2]
			# add (image, reference) pair to data list
			for jpg in jpgs:
				data_list.append((jpg, label_path))
		
		return data_list
	

if __name__ == "__main__":
	# Example of how to use this dataset
	train_dir = "/home/ppnk-wsl/capstone/Dataset/SICE/Dataset_Part1/"
	test_dir = "/home/ppnk-wsl/capstone/Dataset/SICE/Dataset_Part2/"

	train_set = SICE_Dataset(train_dir)
	test_set = SICE_Dataset(test_dir, under_expose_only=True)


