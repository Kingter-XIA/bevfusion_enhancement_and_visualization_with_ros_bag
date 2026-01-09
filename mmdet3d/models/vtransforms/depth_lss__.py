from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform
import time
__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
        
    def set_logging_flag(self, logging):
        self.logging = logging
        self.logging_data={}

    # Original API - renamed to keep it for later use
    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    # add logging to orig
    @force_fp32()
    def get_cam_feats_org_withlog(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)

        #logging
        if(self.logging == True):
            self.logging_data['lidar_depth_64'] = d.cpu().numpy()

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)                           
        
        #logging 
        if(self.logging == True):
            self.logging_data['img_depth_118'] = depth.cpu().numpy()
            self.logging_data['img_feat_80'] = x[:, self.D : (self.D + self.C)].cpu().numpy()
            
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
                                                            #depth = [6,118,32,88]
                                                            #depth.un(1) = [6,1, 118,32,88]
                                                            #x[] = [6,80,32,88]
                                                            #x[].un(2)=[6,80,1,32,88]
                                                            #x =[6,80,118,32,88]
        x = x.view(B, N, self.C, self.D, fH, fW)            #x =[1,6,80,118,32,88]
        x = x.permute(0, 1, 3, 4, 5, 2)                     #x =[1,6,118,32,88,80]
        return x

    #modifying org API to measure performance by replacing d with 1, softmax(1), rand, etc. 
    @force_fp32()
    def get_cam_feats_replace_d(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        #original code 
        if(False):
            depth = x[:, : self.D].softmax(dim=1)

        #all softmax(1.)
        if(True):
            depth = x[:, : self.D]
            depth[...] =1.0
            depth = depth.softmax(dim=1)
            #depth=depth.to('cuda')

            #depth = depth.softmax(dim=1)
            
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x


    """
    改写的get_cam_feats()函数, 相比原函数，我们不做外乘，了，只是将d=dtransform(d)，跟x拼接进行后进行x=depthnet(x)， 
    然后返回其中的图像特征[6,80,118,32,88]
    """
    @force_fp32()
    def get_cam_feats_2(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])                     #[6,1,256,704]
        x = x.view(B * N, C, fH, fW)                        #[6,256,32,88]

        d = self.dtransform(d)                              #[6,64,32,88]

        #logging
        if(self.logging == True):
            self.logging_data['lidar_depth_64'] = d.cpu().numpy()

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)                                #[6,198,32,88]

        #logging 
        if(self.logging == True):
            self.logging_data['img_depth_118'] = x[:, : self.D].softmax(dim=1).cpu().numpy()
            self.logging_data['img_feat_80'] = x[:, self.D : (self.D + self.C)].cpu().numpy()

        """
        depth = x[:, : self.D].softmax(dim=1)                           
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        """
        x= x[:, self.D : (self.D + self.C)].unsqueeze(2)    #[6,80,1,32,88]

        return x

    """
    改写的get_cam_feats()函数, 相比原函数，我们不做外乘，了，只是将d=dtransform(d)，跟x拼接进行
    x=depthnet(x)， 然后分别返回其中的图像特征[6,80,32,88]和深度信息[6,118,32,88]
    """
    @force_fp32()
    def get_cam_feats_3(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])                     #[6,1,256,704]
        x = x.view(B * N, C, fH, fW)                        #[6,256,32,88]

        d = self.dtransform(d)                              #[6,64,32,88]
        
        #logging
        if(self.logging == True):
            self.logging_data['lidar_depth_64'] = d.cpu().numpy()

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)                                #[6,198,32,88]

        #logging 
        if(self.logging == True):
            self.logging_data['img_depth_118'] = x[:, : self.D].softmax(dim=1).cpu().numpy()
            self.logging_data['img_feat_80'] = x[:, self.D : (self.D + self.C)].cpu().numpy()

        """
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        """
        d= x[:, : self.D]                                   #[6,118,32,88]
        x= x[:, self.D : (self.D + self.C)]                 #[6,80,32,88]
        return x,d 

    """
    改写的get_cam_feats()函数, 相比原函数，我们不做depthnet -- 将输入x=[6,256,32,88] 压缩到[6,80,32,88] 返回，
    这个x会跟lidar真值进行叉乘.
    x=[1,6,256,32,88]
    返回 x=[1,6,1,32,88,80]
    """
    @force_fp32()
    def get_cam_feats_4(self, x):
        x = x.view(6, 256, 32, 88)                          #[6,256,32,88]
        
        # Fixed projection matrix: [80, 256]
        projection_matrix = torch.randn(80, 256).to('cuda')

        # Project features: [6, 256, 32, 88] -> [6, 80, 32, 88]
        y = torch.einsum('bcxy,dc->bdxy', x, projection_matrix)
        print(f'-------------------------------------> type]{type(y)}')

        return y
        # Reshape to group channels (256 -> 80 groups of size 3.2)
        x = x[:,16:256,:,:]
        x = x.view(6, 80, 3, 32, 88)

        # Average within groups: [6, 80, 3, 32, 88] -> [6, 80, 32, 88]
        x = x.mean(dim=2)
        x = x.unsqueeze(2)
        print(f'-------------x ={x.shape}')
        return x

    @force_fp32()
    def get_cam_feats_5(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])                     #[6,1,256,704]
        x = x.view(B * N, C, fH, fW)                        #[6,256,32,88]

        d = self.dtransform(d)                              #[6,64,32,88]
        d = torch.full((6,64,32,88),-0.05).to('cuda')                 
                                                            # with no replace:  mini -->0.5755
                                                            # with all zero:    mini -->0.5691 
                                                            # with all one:     mini -->0.5179   
                                                            # with all 0.5      mini -->0.5673  
                                                            # with all -1       mini -->0.4668
                                                            # with all 0.1      mini -->0.5712
                                                            # with randn:       mini -->0.5672
                                                            # with rand         mini -->0.5672
                                                            # with all 0.2      mini -->0.5704
                                                            # with all 0.15     mini -->0.5708
                                                            # with all 0.05     mini -->0.5681
                                                            # with all -0.05    mini -->0.5705
        #logging
        if(self.logging == True):
            self.logging_data['lidar_depth_64'] = d.cpu().numpy()

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)                                #[6,198,32,88]

        #logging 
        if(self.logging == True):
            self.logging_data['img_depth_118'] = x[:, : self.D].softmax(dim=1).cpu().numpy()
            self.logging_data['img_feat_80'] = x[:, self.D : (self.D + self.C)].cpu().numpy()

        """
        depth = x[:, : self.D].softmax(dim=1)                           
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        """
        x= x[:, self.D : (self.D + self.C)].unsqueeze(2)    #[6,80,1,32,88]

        return x

    def forward(self, *args, **kwargs):
        #torch.cuda.synchronize()
        #t0 = time.time()    #t0 = time.perf_counter()         
        x = super().forward(*args, **kwargs)
        torch.cuda.synchronize()
        #t1 = time.time()    #t0 = time.perf_counter()         
        # print(f'total time ={(t1-t0)*1000}')
        print(f'x------------------={x.shape}')
        x = self.downsample(x)
        
        #logging 
        if(self.logging == True):
            self.logging_data['bev_img'] = x.cpu().numpy()
        
        return x