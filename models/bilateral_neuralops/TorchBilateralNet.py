import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.bilateral_neuralops.layers import conv

class TorchSimpleBilateralNetPointwise(nn.Module):
	def __init__(self, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm=False, n_in=3, n_out=3,
				 iteratively_upsample=False):
		super().__init__()
		self.low_res = lowres
		self.luma_bins = luma_bins
		self.spatial_bins = spatial_bins
		self.channel_multiplier = channel_multiplier
		self.feature_multiplier = self.luma_bins * self.channel_multiplier
		self.guide_pts = guide_pts
		self.norm = norm
		self.n_in, self.n_out = n_in, n_out
		self.iteratively_upsample = iteratively_upsample
				
		self.guide = GuideNN()
		self.slice = Slice()
		self.apply_coeffs = ApplyCoeffs()

		self.coefficient_params = self.make_coefficient_params(lowres)
		self.guide_params = self.make_guide_params()

	def forward(self, image, val):
		# resize input to low-res
		image_lowres = F.interpolate(input=image, size=((self.low_res[0], self.low_res[1])), mode='bilinear', align_corners=False)
		coefficients = self.forward_coefficients(image_lowres, val)
		if self.iteratively_upsample:
			coefficients = self.iterative_upsample(coefficients, image.shape[2:4])
		# guidemap = self.forward_guidemap(image)
		guidemap = self.guide(image)
		################################
		slice_coeffs = self.slice(coefficients, guidemap)
		output = self.apply_coeffs(slice_coeffs, image)
		#################################
		return output
	
	def forward_coefficients(self, image_lowres, val):
		splat_features = self.coefficient_params.splat_1(image_lowres)
		### add constant value as in neuralops
		splat_features = splat_features + val
		###
		splat_features = self.coefficient_params.splat_2(splat_features)
		splat_features = self.coefficient_params.splat_3(splat_features)
		coefficients = self.coefficient_params.prediction(splat_features)
		coefficients = torch.stack(torch.split(coefficients, self.n_out*(self.n_in+1), dim=1), dim=2)
		return coefficients

	def make_coefficient_params(self, lowres):
		coefficient_params = nn.Module()
		coefficient_params.splat_1 = conv(self.n_in, 16, 1, stride=2, relu=False)
		coefficient_params.splat_2 = conv(16, 32, 1, stride=2)
		coefficient_params.splat_3 = conv(32, 32, 1, stride=2)

		prediction = conv(32, self.luma_bins * (self.n_in+1) * self.n_out, 1, norm=False, relu=False)

		coefficient_params.prediction = prediction
		return coefficient_params

	def forward_guidemap(self, image_fullres):
		guidemap = self.guide_params.conv1(image_fullres)
		guidemap = self.guide_params.conv2(guidemap)
		return guidemap

	def make_guide_params(self):
		conv1 = conv(self.n_in, self.guide_pts, 1, norm=False)
		conv2 = nn.Sequential(nn.Conv2d(self.guide_pts, 1, 1),
							  nn.Tanh())
		guide_params = nn.Module()
		guide_params.conv1 = conv1
		guide_params.conv2 = conv2
		return guide_params

	def iterative_upsample(self, coefficients, fullres):
		res = self.spatial_bins
		coefficients = coefficients.view(coefficients.shape[0], -1, res, res)
		while res * 2 < min(fullres):
			res *= 2
			coefficients = F.upsample(coefficients, size=[res, res], mode='bilinear')
		coefficients = coefficients.view(coefficients.shape[0], -1, self.luma_bins, res, res)
		return coefficients

class Slice(nn.Module):
	def __init__(self):
		super(Slice, self).__init__()

	def forward(self, bilateral_grid, guidemap):
		device = bilateral_grid.get_device()

		N, _, H, W = guidemap.shape
		hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
		if device >= 0:
			hg = hg.to(device)
			wg = wg.to(device)
		hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
		wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
		guidemap = guidemap.permute(0,2,3,1).contiguous()
		guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
		coeff = F.grid_sample(bilateral_grid, guidemap_guide)
		return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
	def __init__(self):
		super(ApplyCoeffs, self).__init__()
		self.degree = 3

	def forward(self, coeff, full_res_input):

		'''
			Affine:
			r = a11*r + a12*g + a13*b + a14
			g = a21*r + a22*g + a23*b + a24
			...
		'''

		R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
		G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
		B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

		return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
	def __init__(self, params=None):
		super(GuideNN, self).__init__()
		self.params = params
		self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=False)
		self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

	def forward(self, x):
		return self.conv2(self.conv1(x))
	

class ConvBlock(nn.Module):
	def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
				 batch_norm=False):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
		self.activation = activation() if activation else None
		self.bn = nn.BatchNorm2d(outc) if batch_norm else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn:
			x = self.bn(x)
		if self.activation:
			x = self.activation(x)
		return x