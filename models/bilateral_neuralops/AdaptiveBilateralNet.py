import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.bilateral_neuralops.layers import conv, fc, BilateralSliceFunction


class AdaptiveBilateralNetPointwise(nn.Module):
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

		self.make_coefficient_params(lowres)
		self.make_guide_params()

	def forward(self, image, val):
		# resize input to low-res
		image_lowres = F.interpolate(input=image, size=((self.low_res[0], self.low_res[1])), mode='bilinear', align_corners=False)
		coefficients = self.forward_coefficients(image_lowres, val)
		if self.iteratively_upsample:
			coefficients = self.iterative_upsample(coefficients, image.shape[2:4])
		guidemap = self.forward_guidemap(image, val)
		output = BilateralSliceFunction.apply(coefficients, guidemap, image, True)
		# breakpoint()
		# if not self.training:
		# 	# clipping TODO: could use a softer approach instead of hard clipping
		# 	output = F.hardtanh(output, min_val=0, max_val=1)
		return output

	def forward_coefficients(self, image_lowres, val):
		# splat features: N x n_splat x spatial_bins x spatial_bins
		splat_features = self.splat(image_lowres)
		splat_features = splat_features + val

		# local branch
		local_features = self.local(splat_features)

		# condition branch
		condition_features = self.cond_net(splat_features)
		condition_features = condition_features.view(image_lowres.shape[0], -1)
		# condition_features = condition_features + val
		condition_features = self.cond_fc(condition_features)
		condition_features = condition_features.view(image_lowres.shape[0], self.n_splat, 1, 1)
		fused_features = F.relu(condition_features + local_features)
		
		coefficients = self.pred_grid(fused_features)
		coefficients = torch.stack(torch.split(coefficients, self.n_out*(self.n_in+1), dim=1), dim=2)
		return coefficients

	def make_coefficient_params(self, lowres):
		# HDRNet splat params
		splat = []
		in_channels = self.n_in
		num_downsamples = int(np.log2(min(lowres) / self.spatial_bins))
		extra_convs = max(0, int(np.log2(self.spatial_bins) - np.log2(16)))
		extra_convs = np.linspace(0, num_downsamples - 1, extra_convs, dtype=int).tolist()
		for i in range(num_downsamples):
			out_channels = (2 ** i) * self.feature_multiplier
			splat.append(conv(in_channels, out_channels, 3, stride=2, norm=False if i == 0 else self.norm))
			if i in extra_convs:
				splat.append(conv(out_channels, out_channels, 3, norm=self.norm))
			in_channels = out_channels

		self.n_splat = in_channels
		splat.append(conv(self.n_splat, self.n_splat, 1, relu=False))
		self.splat = nn.Sequential(*splat)

		# local branch
		self.local = nn.Sequential(conv(self.n_splat, 2*self.n_splat, 1),
			     				conv(2*self.n_splat, 2*self.n_splat, 1),
							  conv(2*self.n_splat, self.n_splat, 1))

		# # splat features
		# self.splat = nn.Sequential(
		# 	conv(self.n_in, self.n_splat, 1, stride=2),
		# 	conv(self.n_splat, self.n_splat, 1, stride=1)
		# )

		# condition networks (global condition)
		self.cond_net = nn.Sequential(
			conv(self.n_splat, 4, 1, stride=2),
			nn.AdaptiveAvgPool2d(4) 	# pool to (N,4,4,4)
		)

		self.cond_fc = nn.Sequential(
			fc(64, 64),
			fc(64, self.n_splat),
		)

		# from fused of condition + splat (32-channel) -> (3*4*luma_bins-channel)
		self.pred_grid = conv(self.n_splat, self.luma_bins * (self.n_in+1) * self.n_out, 1, norm=False, relu=False)

	def forward_guidemap(self, image_fullres, val):
		guidemap = self.ccm(image_fullres)     # bs x C x H x W
		guidemap = guidemap.unsqueeze(dim=4)                # bs x C x H x W x 1
		# bs x C x H x W = ( 1 x C x 1 x 1 x 4 * F.relu( bs x C x H x W x 1 - C x 1 x 1 x 4 ).sum(dim=4)
		guidemap = (self.slopes * F.relu(guidemap - self.shifts)).sum(dim=4)
		guidemap = self.projection(guidemap)   	# bs x 1 x H x W
		guidemap = guidemap.sum(dim=1)  			# bs x 1 x H x W, instead of project, use sum as indicated in the paper
		guidemap = F.hardtanh(guidemap, min_val=0, max_val=1)
		# guidemap = guidemap.squeeze(dim=1)                  # bs x H x W
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

		self.ccm = ccm
		self.shifts = shifts
		self.slopes = slopes
		self.projection = projection

	# def forward_guidemap(self, image_fullres, val):
	# 	guidemap = self.guide_conv1(image_fullres)
	# 	guidemap = guidemap + val
	# 	guidemap = self.guide_conv2(guidemap)
	# 	guidemap = self.guide_conv3(guidemap)
	# 	guidemap = guidemap.squeeze(dim=1)
	# 	return guidemap

	# def make_guide_params(self):
	# 	self.guide_conv1 = conv(self.n_in, self.guide_pts, 1, norm=False, relu=False)
	# 	self.guide_conv2 = conv(self.guide_pts, self.guide_pts, 1, norm=False, relu=True)
	# 	self.guide_conv3 = nn.Sequential(
	# 		nn.Conv2d(self.guide_pts, 1, 1),
	# 		nn.Sigmoid()
	# 	)

	def iterative_upsample(self, coefficients, fullres):
		res = self.spatial_bins
		coefficients = coefficients.view(coefficients.shape[0], -1, res, res)
		while res * 2 < min(fullres):
			res *= 2
			coefficients = F.upsample(coefficients, size=[res, res], mode='bilinear')
		coefficients = coefficients.view(coefficients.shape[0], -1, self.luma_bins, res, res)
		return coefficients
