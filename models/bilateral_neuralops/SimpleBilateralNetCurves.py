import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.bilateral_neuralops.layers import conv, fc, BilateralSliceFunction


class SimpleBilateralNetCurves(nn.Module):
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

		self.coefficient_params = self.make_coefficient_params(lowres)
		self.guide_params = self.make_guide_params()

	def forward(self, image, val):
		# resize input to low-res
		image_lowres = F.interpolate(input=image, size=((self.low_res[0], self.low_res[1])), mode='bilinear', align_corners=False)
		coefficients = self.forward_coefficients(image_lowres, val)
		if self.iteratively_upsample:
			coefficients = self.iterative_upsample(coefficients, image.shape[2:4])
		guidemap = self.forward_guidemap(image)
		output = BilateralSliceFunction.apply(coefficients, guidemap, image, True)
		if not self.training:
			# clipping TODO: could use a softer approach instead of hard clipping
			output = F.hardtanh(output, min_val=0, max_val=1)
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
		guidemap = self.guide_params.ccm(image_fullres)     # bs x C x H x W
		guidemap = guidemap.unsqueeze(dim=4)                # bs x C x H x W x 1
		# bs x C x H x W = ( 1 x C x 1 x 1 x 4 * F.relu( bs x C x H x W x 1 - C x 1 x 1 x 4 ).sum(dim=4)
		guidemap = (self.guide_params.slopes * F.relu(guidemap - self.guide_params.shifts)).sum(dim=4)
		guidemap = self.guide_params.projection(guidemap)   # bs x 1 x H x W
		guidemap = F.hardtanh(guidemap, min_val=0, max_val=1)
		guidemap = guidemap.squeeze(dim=1)                  # bs x H x W
		return guidemap

	def make_guide_params(self):
		ccm = conv(self.n_in, self.n_in, 1, norm=False, relu=False,
				   weights_init=(np.identity(self.n_in, dtype=np.float32) +
								 np.random.randn(1).astype(np.float32) * 1e-4)
				   .reshape((self.n_in, self.n_in, 1, 1)),
				   bias_init=torch.zeros(self.n_in))

		shifts = np.linspace(0, 1, self.guide_pts, endpoint=False, dtype=np.float32)
		shifts = shifts[np.newaxis, np.newaxis, np.newaxis, :]
		shifts = np.tile(shifts, (self.n_in, 1, 1, 1))
		shifts = nn.Parameter(data=torch.from_numpy(shifts))

		slopes = np.zeros([1, self.n_in, 1, 1, self.guide_pts], dtype=np.float32)
		slopes[:, :, :, :, 0] = 1.0
		slopes = nn.Parameter(data=torch.from_numpy(slopes))

		projection = conv(self.n_in, 1, 1, norm=False, relu=False,
						  weights_init=torch.ones(1, self.n_in, 1, 1) / self.n_in,
						  bias_init=torch.zeros(1))

		guide_params = nn.Module()
		guide_params.ccm = ccm
		guide_params.shifts = shifts
		guide_params.slopes = slopes
		guide_params.projection = projection
		return guide_params

	def iterative_upsample(self, coefficients, fullres):
		res = self.spatial_bins
		coefficients = coefficients.view(coefficients.shape[0], -1, res, res)
		while res * 2 < min(fullres):
			res *= 2
			coefficients = F.upsample(coefficients, size=[res, res], mode='bilinear')
		coefficients = coefficients.view(coefficients.shape[0], -1, self.luma_bins, res, res)
		return coefficients


