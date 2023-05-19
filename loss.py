import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


### Custom Loss Compatible with Config File
class CustomLoss(nn.Module):
    def __init__(self, 
        patch_size=16,
        W_TV=0, W_col=0, W_spa=0, W_exp=0, E=0.6,
        W_L2=0, W_L1=0, W_cos=0
    ):
        super().__init__()
        # self.zero_ref_loss = ZeroReferenceLoss(
        #     patch_size,
        #     W_TV, W_col,
        #     W_spa, W_exp,
        #     E)
        self.TV_loss = TVLoss()
        self.L2_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss() # nn.SmoothL1Loss()
        self.cos_loss = CosineLoss()
        self.W_L2 = W_L2
        self.W_L1 = W_L1
        self.W_cos = W_cos
        self.W_TV = W_TV
    
    def forward(self, y, target, x=None, illum_map=None):
        """
        calculate weighted sum of each loss components
        if x (input image) or illum_map is None, zero_ref_loss will not be computed
        """
        if x is None or illum_map is None:
            zero_ref_loss = 0
        else:
            zero_ref_loss = self.zero_ref_loss(x, y, illum_map)

        l1_loss = self.L1_loss(y, target)
        l2_loss = self.L2_loss(y, target)
        cos_loss = self.cos_loss(y, target)
        TV_loss = self.TV_loss(y)

        # weighted sum of loss components
        return l2_loss * self.W_L2 +\
               l1_loss * self.W_L1 +\
               cos_loss * self.W_cos +\
               TV_loss * self.W_TV

### NeuralOps Loss
class NeuralOpsLoss(nn.Module):
    def __init__(self, W_TV=0.1, W_cos=0.1):
        super(NeuralOpsLoss, self).__init__()
        self.W_TV = W_TV
        self.W_cos = W_cos
        self.TV_Loss = TVLoss()
        self.Cos_Loss = CosineLoss()
        self.L1_Loss = nn.L1Loss()
    
    def forward(self, y, target):
        return self.L1_Loss(y, target) + self.TV_Loss(y) * self.W_TV + self.Cos_Loss(y, target) * self.W_cos
    
class ExposureLoss(nn.Module):
    def __init__(self):
        super(ExposureLoss, self).__init__()
        self.L1_Loss = nn.L1Loss()
        self.grayscale = transforms.Grayscale()

    def forward(self, y, target):
        # convert y & target to monochrome
        y = self.grayscale(y)
        target = self.grayscale(target)
        
        return self.L1_Loss(y, target)

### NeuralOps Loss Components
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x, y):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view(-1, c)
        y = y.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        return loss

### Zero Reference Loss from Zero-DCE++
class ZeroReferenceLoss(nn.Module):
    def __init__(self, patch_size=16, W_TV=20, W_col=0.5, W_spa=1.0, W_exp=1.0, E=0.6):
        super(ZeroReferenceLoss, self).__init__()
        self.E = E
        self.W_TV = W_TV
        self.W_col = W_col
        self.W_spa = W_spa
        self.W_exp = W_exp

        self.color_consistency_loss = ColorConsistencyLoss()
        self.spatial_consistency_loss = SpatialConsistencyLoss()
        self.exposure_control_loss = ExposureControlLoss(patch_size)
        self.illumination_smoothness_loss = IlluminationSmoothnessLoss()

    def forward(self, input, enhanced, A):
        # loss_TV = 1600 * self.illumination_smoothness_loss(A)
        # loss_spa = torch.mean(self.spatial_consistency_loss(enhanced, input))
        # loss_col = 5 * torch.mean(self.color_consistency_loss(enhanced))
        # loss_exp = 10 * torch.mean(self.exposure_control_loss(enhanced, self.E))

        loss_TV =  self.illumination_smoothness_loss(A)
        loss_spa = torch.mean(self.spatial_consistency_loss(enhanced, input))
        loss_col = torch.mean(self.color_consistency_loss(enhanced))
        loss_exp = torch.mean(self.exposure_control_loss(enhanced, self.E))

        return self.W_TV * loss_TV + self.W_spa * loss_spa + self.W_col * loss_col + self.W_exp * loss_exp

### ZeroReferenceLoss Components
class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

class SpatialConsistencyLoss(nn.Module):

    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
    
class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size):
        super(ExposureControlLoss, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        return d
        
class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(IlluminationSmoothnessLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    

