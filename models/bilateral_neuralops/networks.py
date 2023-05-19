import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import numpy as np

from models.bilateral_neuralops.DeepBilateralNetCurves import DeepBilateralNetCurves
from models.bilateral_neuralops.SimpleBilateralNetCurves import SimpleBilateralNetCurves
from models.bilateral_neuralops.SimpleBilateralNetPoint import SimpleBilateralNetPointwise
from models.bilateral_neuralops.AdaptiveBilateralNet import AdaptiveBilateralNetPointwise
from models.bilateral_neuralops.TorchBilateralNet import TorchSimpleBilateralNetPointwise


#####################################
#             Operator              #
##################################### 

class Operator(nn.Module):
	def __init__(self, in_nc=3, out_nc=3,base_nf=64):
		super(Operator,self).__init__()
		self.base_nf = base_nf
		self.out_nc = out_nc
		self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
		self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
		self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
		self.act = nn.LeakyReLU(inplace=True)

	def forward(self, x, val):
		x_code = self.encoder(x)
		y_code = x_code + val
		y_code = self.act(self.mid_conv(y_code))
		y = self.decoder(y_code)
		return y

############################################################################################################
class BilateralOperator(nn.Module):
	"""
	This operator learns a bilateral grid from down-sampled input image, 
	when forward, it slices the learned bilateral grid using full-res input image
	to get a full-res output image. 
	"""
	def __init__(self, in_nc=3, out_nc=3,base_nf=64):
		super(BilateralOperator,self).__init__()
		# self.base_nf = base_nf
		# self.out_nc = out_nc
		# self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
		# self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
		# self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
		# self.act = nn.LeakyReLU(inplace=True)

		self.bilateral_model = DeepBilateralNetCurves(
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=16,
			channel_multiplier=1,
			guide_pts=4,
			norm=False,
			n_in=in_nc,
			n_out=out_nc,
			iteratively_upsample=False
		)

	def forward(self, x, val):
		# x_code = self.encoder(x)
		# y_code = x_code + val
		# y_code = self.act(self.mid_conv(y_code))
		# y = self.decoder(y_code)
		y = self.bilateral_model(x, val)
		return y
############################################################################################################

############################################################################################################
class SimpleBilateralOperator(nn.Module):
	"""
	This operator learns a bilateral grid from down-sampled input image, 
	when forward, it slices the learned bilateral grid using full-res input image
	to get a full-res output image. 
	"""
	def __init__(self, in_nc=3, out_nc=3,base_nf=64):
		super(SimpleBilateralOperator,self).__init__()
		# self.base_nf = base_nf
		# self.out_nc = out_nc
		# self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
		# self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
		# self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
		# self.act = nn.LeakyReLU(inplace=True)

		self.bilateral_model = SimpleBilateralNetPointwise(
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=64,
			channel_multiplier=1,
			guide_pts=8,
			norm=False,
			n_in=in_nc,
			n_out=out_nc,
			iteratively_upsample=False
		)

	def forward(self, x, val):
		# x_code = self.encoder(x)
		# y_code = x_code + val
		# y_code = self.act(self.mid_conv(y_code))
		# y = self.decoder(y_code)
		y = self.bilateral_model(x, val)
		return y
############################################################################################################

############################################################################################################
class AdaptiveBilateralOperator(nn.Module):
	"""
	This operator learns a bilateral grid from down-sampled input image, 
	when forward, it slices the learned bilateral grid using full-res input image
	to get a full-res output image. 
	"""
	def __init__(self,
			in_nc=3,
			out_nc=3,
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=64,
			channel_multiplier=1,
			guide_pts=8,
			norm=False,
			iteratively_upsample=False
		):
		super(AdaptiveBilateralOperator,self).__init__()

		self.bilateral_model = AdaptiveBilateralNetPointwise(
			lowres=lowres,
			luma_bins=luma_bins,
			spatial_bins=spatial_bins,
			channel_multiplier=channel_multiplier,
			guide_pts=guide_pts,
			norm=norm,
			n_in=in_nc,
			n_out=out_nc,
			iteratively_upsample=iteratively_upsample
		)

	def forward(self, x, val):
		y = self.bilateral_model(x, val)
		return y
############################################################################################################

############################################################################################################
class TorchSimpleBilateralOperator(nn.Module):
	"""
	This operator learns a bilateral grid from down-sampled input image, 
	when forward, it slices the learned bilateral grid using full-res input image
	to get a full-res output image. 
	"""
	def __init__(self, in_nc=3, out_nc=3,base_nf=64):
		super(TorchSimpleBilateralOperator,self).__init__()
		# self.base_nf = base_nf
		# self.out_nc = out_nc
		# self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
		# self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
		# self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
		# self.act = nn.LeakyReLU(inplace=True)

		self.bilateral_model = TorchSimpleBilateralNetPointwise(
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=64,
			channel_multiplier=1,
			guide_pts=8,
			norm=False,
			n_in=in_nc,
			n_out=out_nc,
			iteratively_upsample=False
		)

	def forward(self, x, val):
		# x_code = self.encoder(x)
		# y_code = x_code + val
		# y_code = self.act(self.mid_conv(y_code))
		# y = self.decoder(y_code)
		y = self.bilateral_model(x, val)
		return y
############################################################################################################

#####################################
#             Renderer              #
##################################### 

class Renderer(nn.Module):
	def __init__(self, in_nc=3, out_nc=3,base_nf=64):
		super(Renderer,self).__init__()
		self.in_nc = in_nc
		self.base_nf = base_nf
		self.out_nc = out_nc
		self.ex_block = Operator(in_nc,out_nc,base_nf)
		self.bc_block = Operator(in_nc,out_nc,base_nf)
		self.vb_block = Operator(in_nc,out_nc,base_nf)

	def forward(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb
	
############################################################################################################
class BilateralRenderer(nn.Module):
	def __init__(self, in_nc=3, out_nc=3):
		super(BilateralRenderer,self).__init__()
		self.in_nc = in_nc
		self.out_nc = out_nc
		self.ex_block = BilateralOperator(in_nc,out_nc)
		self.bc_block = BilateralOperator(in_nc,out_nc)
		self.vb_block = BilateralOperator(in_nc,out_nc)

	def forward(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb
############################################################################################################

############################################################################################################
class SimpleBilateralRenderer(nn.Module):
	def __init__(self, in_nc=3, out_nc=3):
		super(SimpleBilateralRenderer,self).__init__()
		self.in_nc = in_nc
		self.out_nc = out_nc
		self.ex_block = SimpleBilateralOperator(in_nc,out_nc)
		self.bc_block = SimpleBilateralOperator(in_nc,out_nc)
		self.vb_block = SimpleBilateralOperator(in_nc,out_nc)

	def forward(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb
############################################################################################################

############################################################################################################
class AdaptiveBilateralRenderer(nn.Module):
	def __init__(self,
			n_in=3,
			n_out=3,
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=64,
			channel_multiplier=1,
			guide_pts=8,
			norm=False,
			iteratively_upsample=False
	):
		super(AdaptiveBilateralRenderer, self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.ex_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)
		self.bc_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)
		self.vb_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)

	def forward(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb
############################################################################################################

############################################################################################################
class ColorBilateralRenderer(nn.Module):
	def __init__(self,
			n_in=3,
			n_out=3,
			lowres=[256, 256],
			luma_bins=8,
			spatial_bins=64,
			channel_multiplier=1,
			guide_pts=8,
			norm=False,
			iteratively_upsample=False
	):
		super(ColorBilateralRenderer, self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.ex_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)
		self.bc_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)
		self.wb_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)
		self.vb_block = AdaptiveBilateralOperator(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample)

	def forward(self, x_ex, x_bc, x_wb, x_vb, v_ex, v_bc, v_wb, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_wb = self.wb_block(x_wb,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_wb = self.wb_block(x_wb,v_wb)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_wb, rec_vb, map_ex, map_bc, map_wb, map_vb
############################################################################################################

############################################################################################################
class TorchSimpleBilateralRenderer(nn.Module):
	def __init__(self, in_nc=3, out_nc=3):
		super(TorchSimpleBilateralRenderer,self).__init__()
		self.in_nc = in_nc
		self.out_nc = out_nc
		self.ex_block = TorchSimpleBilateralOperator(in_nc,out_nc)
		self.bc_block = TorchSimpleBilateralOperator(in_nc,out_nc)
		self.vb_block = TorchSimpleBilateralOperator(in_nc,out_nc)

	def forward(self, x_ex, x_bc, x_vb, v_ex, v_bc, v_vb):
		rec_ex = self.ex_block(x_ex,0)
		rec_bc = self.bc_block(x_bc,0)
		rec_vb = self.vb_block(x_vb,0)

		map_ex = self.ex_block(x_ex,v_ex)
		map_bc = self.bc_block(x_bc,v_bc)
		map_vb = self.vb_block(x_vb,v_vb)

		return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb
############################################################################################################

class Encoder(nn.Module):
	def __init__(self, in_nc=3, encode_nf=32):
		super(Encoder, self).__init__()
		stride = 2
		pad = 0
		self.pad = nn.ZeroPad2d(1)
		self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
		self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
		self.act = nn.ReLU(inplace=True)
		self.max = nn.AdaptiveMaxPool2d((1,1))

	def forward(self, x):
		b, _,_,_ = x.size()
		conv1_out = self.act(self.conv1(self.pad(x)))
		conv2_out = self.act(self.conv2(self.pad(conv1_out)))
		std, mean = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
		maxs = self.max(conv2_out).squeeze(2).squeeze(2)
		out = torch.cat([std, mean, maxs], dim=1)
		return out


class Predictor(nn.Module):
	"""
	outputs a 1-dim scalar
	"""
	def __init__(self,fea_dim):
		super(Predictor,self).__init__()
		self.fc3 = nn.Linear(fea_dim,1)
		self.tanh = nn.Tanh()
	def forward(self,img_fea):
		val = self.tanh(self.fc3(img_fea))
		return val    

class AdaptivePredictor(nn.Module):
	"""
	outputs a out_dim dimensional vector
	"""
	def __init__(self, fea_dim, out_dim):
		super(AdaptivePredictor,self).__init__()
		self.fc3 = nn.Linear(fea_dim, out_dim)
		self.tanh = nn.Tanh()
	def forward(self, img_fea):
		val = self.tanh(self.fc3(img_fea))
		return val    

#####################################
#         Neural Op Models          #
##################################### 

############################################################################################################
class BilateralNeurOP(nn.Module):
	def __init__(self,
				 # neural op params
				 in_nc = 3,
				 out_nc = 3,
				 base_nf = 64,
				 encode_nf = 32,
				 load_path = None,
				 return_vals = False,
				 # bilatera grid params
				 luma_bins = 8,
				 spatial_bins = 16,
				 channel_multiplier = 1,
				 guide_pts = 4,
				 norm = False,
				 iterative_upsample = False
		):
		super(BilateralNeurOP,self).__init__()

		self.fea_dim = encode_nf * 3
		self.image_encoder = Encoder(in_nc,encode_nf)
		renderer = BilateralRenderer(in_nc,out_nc) # TODO: pass bilateral params here
		if load_path is not None: 
			renderer.load_state_dict(torch.load(load_path))
			
		self.bc_renderer = renderer.bc_block
		self.bc_predictor =  Predictor(self.fea_dim)
		
		self.ex_renderer = renderer.ex_block
		self.ex_predictor =  Predictor(self.fea_dim)
		
		self.vb_renderer = renderer.vb_block
		self.vb_predictor =  Predictor(self.fea_dim)

		self.renderers = [self.bc_renderer, self.ex_renderer, self.vb_renderer]
		self.predict_heads = [self.bc_predictor ,self.ex_predictor, self.vb_predictor]

		# if enabled, forward will return output, vals
		self.return_vals = return_vals
			
	def render(self, x, vals):
		b,_,h,w = img.shape
		imgs = []
		for nop, scalar in zip(self.renderers,vals):
			img = nop(img,scalar)
			output_img = torch.clamp(img, 0, 1.0)
			imgs.append(output_img)
		return imgs
	
	def forward(self, img):
		b,_,h,w = img.shape
		vals = []
		for nop, predict_head in zip(self.renderers,self.predict_heads):
			img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img,scalar)
		if self.return_vals:
			return img, vals
		else:
			return img
############################################################################################################


############################################################################################################
class SimpleBilateralNeurOP(nn.Module):
	def __init__(self,
				 # neural op params
				 in_nc = 3,
				 out_nc = 3,
				 base_nf = 64,
				 encode_nf = 32,
				 load_path = None,
				 return_vals = False,
				 # bilatera grid params
				 luma_bins = 8,
				 spatial_bins = 16,
				 channel_multiplier = 1,
				 guide_pts = 4,
				 norm = False,
				 iterative_upsample = False
		):
		super(SimpleBilateralNeurOP,self).__init__()

		self.fea_dim = encode_nf * 3
		self.image_encoder = Encoder(in_nc,encode_nf)
		renderer = SimpleBilateralRenderer(in_nc, out_nc) # TODO: pass bilateral params here
		if load_path is not None: 
			renderer.load_state_dict(torch.load(load_path))
			
		self.bc_renderer = renderer.bc_block
		self.bc_predictor =  Predictor(self.fea_dim)
		
		self.ex_renderer = renderer.ex_block
		self.ex_predictor =  Predictor(self.fea_dim)
		
		self.vb_renderer = renderer.vb_block
		self.vb_predictor =  Predictor(self.fea_dim)

		self.renderers = [self.bc_renderer, self.ex_renderer, self.vb_renderer]
		self.predict_heads = [self.bc_predictor ,self.ex_predictor, self.vb_predictor]

		# if enabled, forward will return output, vals
		self.return_vals = return_vals
			
	def render(self, x, vals):
		b,_,h,w = img.shape
		imgs = []
		for nop, scalar in zip(self.renderers,vals):
			img = nop(img,scalar)
			output_img = torch.clamp(img, 0, 1.0)
			imgs.append(output_img)
		return imgs
	
	def forward(self, img):
		b,_,h,w = img.shape
		vals = []
		for nop, predict_head in zip(self.renderers,self.predict_heads):
			img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img,scalar)
		if self.return_vals:
			return img, vals
		else:
			return img
############################################################################################################

############################################################################################################
class AdaptiveBilateralNeurOP(nn.Module):
	def __init__(self,
				 # neural op params
				 n_in = 3,
				 n_out = 3,
				 encode_nf = 32,
				 load_path = None,
				 return_vals = False,
				 # bilatera grid params
				 lowres=[256, 256],
				 luma_bins = 8,
				 spatial_bins = 64,
				 channel_multiplier = 1,
				 guide_pts = 8,
				 norm = False,
				 iteratively_upsample = False
		):
		super(AdaptiveBilateralNeurOP,self).__init__()

		self.fea_dim = encode_nf * 3
		self.image_encoder = Encoder(n_in,encode_nf)
		renderer = AdaptiveBilateralRenderer(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample) # TODO: pass bilateral params here
		if load_path is not None: 
			renderer.load_state_dict(torch.load(load_path))
			
		self.bc_renderer = renderer.bc_block
		self.bc_predictor =  Predictor(self.fea_dim)
		
		self.ex_renderer = renderer.ex_block
		self.ex_predictor =  Predictor(self.fea_dim)
		
		self.vb_renderer = renderer.vb_block
		self.vb_predictor =  Predictor(self.fea_dim)

		self.renderers = [self.bc_renderer, self.ex_renderer, self.vb_renderer]
		self.predict_heads = [self.bc_predictor ,self.ex_predictor, self.vb_predictor]

		# if enabled, forward will return output, vals
		self.return_vals = return_vals
			
	def render(self, x, vals):
		b,_,h,w = img.shape
		imgs = []
		for nop, scalar in zip(self.renderers, vals):
			img = nop(img, scalar)
			output_img = torch.clamp(img, 0, 1.0)
			imgs.append(output_img)
		return imgs
	
	def forward(self, img):
		b,_,h,w = img.shape
		vals = []
		for nop, predict_head in zip(self.renderers,self.predict_heads):
			img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img,scalar)
		if self.return_vals:
			return img, vals
		else:
			return img
############################################################################################################

############################################################################################################
class ColorBilateralNeurOP(nn.Module):
	def __init__(self,
				 # neural op params
				 n_in = 3,
				 n_out = 3,
				 encode_nf = 32,
				 load_path = None,
				 return_vals = False,
				 # bilatera grid params
				 lowres=[256, 256],
				 luma_bins = 8,
				 spatial_bins = 64,
				 channel_multiplier = 1,
				 guide_pts = 8,
				 norm = False,
				 iteratively_upsample = False,
				 order=None
		):
		super(ColorBilateralNeurOP,self).__init__()

		self.fea_dim = encode_nf * 3
		self.image_encoder = Encoder(n_in,encode_nf)
		renderer = ColorBilateralRenderer(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample) # TODO: pass bilateral params here
		if load_path is not None: 
			renderer.load_state_dict(torch.load(load_path))
			
		self.bc_renderer = renderer.bc_block
		self.bc_predictor =  Predictor(self.fea_dim)
		
		self.ex_renderer = renderer.ex_block
		self.ex_predictor =  Predictor(self.fea_dim)

		self.wb_renderer = renderer.wb_block
		self.wb_predictor =  Predictor(self.fea_dim)
		
		self.vb_renderer = renderer.vb_block
		self.vb_predictor =  Predictor(self.fea_dim)

		if order == "ebwv":
			print(f"[Neural Ops Order] {order}")
			# exp5 order (exposure - black clipping - white balance - vibrance)
			self.renderers = [self.ex_renderer, self.bc_renderer, self.wb_renderer, self.vb_renderer]
			self.predict_heads = [self.ex_predictor, self.bc_predictor, self.wb_predictor, self.vb_predictor]
		else:
			# default order (black clipping - exposure - white balance - vibrance)
			self.renderers = [self.bc_renderer, self.ex_renderer, self.wb_renderer, self.vb_renderer]
			self.predict_heads = [self.bc_predictor ,self.ex_predictor, self.wb_predictor, self.vb_predictor]

		# if enabled, forward will return output, vals
		self.return_vals = return_vals
			
	def render(self, x, vals):
		b,_,h,w = img.shape
		imgs = []
		for nop, scalar in zip(self.renderers, vals):
			img = nop(img, scalar)
			output_img = torch.clamp(img, 0, 1.0)
			imgs.append(output_img)
		return imgs
	
	def forward(self, img):
		b,_,h,w = img.shape
		vals = []
		for nop, predict_head in zip(self.renderers,self.predict_heads):
			img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img,scalar)
		if self.return_vals:
			return img, vals
		else:
			return img
############################################################################################################

############################################################################################################
class MultiLossBilateralNeurOP(nn.Module):
	def __init__(self,
				 # neural op params
				 n_in = 3,
				 n_out = 3,
				 encode_nf = 32,
				 load_path = None,
				 return_vals = False,
				 # bilatera grid params
				 lowres=[256, 256],
				 luma_bins = 8,
				 spatial_bins = 64,
				 channel_multiplier = 1,
				 guide_pts = 8,
				 norm = False,
				 iteratively_upsample = False,
				 order=None
		):
		super(ColorBilateralNeurOP,self).__init__()

		self.fea_dim = encode_nf * 3
		self.image_encoder = Encoder(n_in,encode_nf)
		renderer = ColorBilateralRenderer(n_in, n_out, lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts, norm, iteratively_upsample) # TODO: pass bilateral params here
		if load_path is not None: 
			renderer.load_state_dict(torch.load(load_path))
			
		self.bc_renderer = renderer.bc_block
		self.bc_predictor =  Predictor(self.fea_dim)
		
		self.ex_renderer = renderer.ex_block
		self.ex_predictor =  Predictor(self.fea_dim)

		self.wb_renderer = renderer.wb_block
		self.wb_predictor =  Predictor(self.fea_dim)
		
		self.vb_renderer = renderer.vb_block
		self.vb_predictor =  Predictor(self.fea_dim)

		if order == "ebwv":
			print(f"[Neural Ops Order] {order}")
			# exp5 order (exposure - black clipping - white balance - vibrance)
			self.renderers = [self.ex_renderer, self.bc_renderer, self.wb_renderer, self.vb_renderer]
			self.predict_heads = [self.ex_predictor, self.bc_predictor, self.wb_predictor, self.vb_predictor]
		else:
			# default order (black clipping - exposure - white balance - vibrance)
			self.renderers = [self.bc_renderer, self.ex_renderer, self.wb_renderer, self.vb_renderer]
			self.predict_heads = [self.bc_predictor ,self.ex_predictor, self.wb_predictor, self.vb_predictor]

		# if enabled, forward will return output, vals
		self.return_vals = return_vals
			
	def render(self, x, vals):
		b,_,h,w = img.shape
		imgs = []
		for nop, scalar in zip(self.renderers, vals):
			img = nop(img, scalar)
			output_img = torch.clamp(img, 0, 1.0)
			imgs.append(output_img)
		return imgs
	
	def forward(self, img):
		b,_,h,w = img.shape
		vals = []
		imgs = []
		for nop, predict_head in zip(self.renderers,self.predict_heads):
			img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
			feat = self.image_encoder(img_resized)
			scalar = predict_head(feat)
			vals.append(scalar)
			img = nop(img,scalar)
			imgs.append(img)

		if self.return_vals:
			return imgs, vals
		else:
			return imgs
############################################################################################################