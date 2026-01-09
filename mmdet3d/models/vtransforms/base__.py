from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool
import time 
import os
import numpy as np 

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.frustum_d1 = self.create_frustum_d1()                          #创建[1,32,88]的结构，用于推理时保存深度信息
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    """
    参考create_frustum(), 改写后新增加这个函数，在程序初始化的时候创建self.frustum_d1数据结构。
    self.frustum=create_frustum() 创建的是[118,32,88,3]的tensor, 每个点x,y,z。 
     x=(0, 8.0, ..., 703) 是图像的宽度，像素单位
     y=(0, 8.2, ..., 205) 是图像的高度，像素单位
     z=(1, 1.5, ..., 59.5)是图像的深度，米单位。
    self.get_geometry()则负责把.self.frustum转换到lidar坐标系。  
   

    本函数创建self.frustum_d1,他说一个维度是[1,32,88,3]的
     x=(0, 8.0, ..., 703) 是图像的宽度，像素单位
     y=(0, 8.2, ..., 205) 是图像的高度，像素单位
     z=(0)                初始为0，是图像的深度，米单位。
    self.get_geometry_d1()被调用时，提供Lildar在32x88个格子上的真实的深度信息，然后会把它转换到lidar坐标系。

    在init函数中，调用self.frustum_d1 = self.create_frustum_d1()，来保存创建的结构
    """    
    @force_fp32()
    def create_frustum_d1(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size
        D = 1

        xs = (
            torch.linspace(4.0, iW - 1, fW, dtype=torch.float)                  #格子的中心点4.0取代左边界
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(4.0, iH - 1, fH, dtype=torch.float)                  #格子的中心点4.0取代上边界
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        ds = torch.zeros((1,fH,fW), dtype=torch.float)

        frustum = torch.stack((xs, ys, ds), -1).to('cuda')
        #return nn.Parameter(frustum, requires_grad=False)                  
        return frustum
    
    """
    试验一个新的函数。 原来的frustum_d1是固定的x,y坐标，新的函数将用points的平均坐标来取代固定的x,y 来看看精度是不是更好
    输入不是深度而是包含点云的xyz坐标 depth -->pt_xyz. 是点云投影到6个image空间的xyz坐标。因此本函数只是将点云(在图像空间）
    的xyz转换到Lidar空间

    本函数貌似效果不好。

    """
    @force_fp32()
    def get_geometry_d1_xyz(self, 
                        pt_xyz,                                 #input pt xyz = [1,6,1,32,88,3]  
                        camera2lidar_rots,
                        camera2lidar_trans,
                        intrins,
                        post_rots,
                        post_trans,
                        **kwargs,
        ):
        B, N, _ = camera2lidar_trans.shape                      # 取batch(=1), N(=6)

        points = pt_xyz - post_trans.view(B,N,1,1,1,3)          # [1,6,1,32,88,3]  图像crop的逆操作
        
        points = (torch.inverse(post_rots).view(B,N, 1, 1, 1,3, 3).matmul(points.unsqueeze(-1)))
        
        points = torch.cat(                                         # UV = UV*z
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))  # [1,6,3,3]                            
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
                                                                    # 图像坐标系 -> Camera坐标系 -> Lildar坐标系。 更改的应该只是UV,深度信息在这个转换中不变 
        

        if "extra_rots" in kwargs:                                  #Nuscenes数据集没有对Lildar点云进行额外的处理（缩放、平移等），因此这里是空操作
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )

        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
       
        return points
    
    
    """
    这个函数是get_geometry()的对应版本。相比于get_geometry_d1()，本函数多传入了一个depth=[1,6,1,32,88]的数据结构。

    self.frustum_d1从[1,32,88,3] --> [1,6,1,32,88,3]. 实际的深度信息被赋值，然后再转换回Lildar坐标系。
    """
    @force_fp32()
    def get_geometry_d1(self, 
                        depth,                                      #input depth = [1,6,1,32,88]  
                        camera2lidar_rots,
                        camera2lidar_trans,
                        intrins,
                        post_rots,
                        post_trans,
                        **kwargs,
        ):
        
        B, N, _ = camera2lidar_trans.shape                          # 取batch(=1), N(=6)

        depth[depth<1.0 ]=1.0                                       # depth[]中所有为0的数替换为1.0米，跟原来的frustum()最小深度一致。        

        points = self.frustum_d1 - post_trans.view(B,N,1,1,1,3)     # [1,6,1,32,88,3]  图像crop的逆操作
        points = (torch.inverse(post_rots).view(B,N, 1, 1, 1,3, 3).matmul(points.unsqueeze(-1)))
                                                                    #[1,6,1,32,88,3,1] 图像缩放resize的逆操作
        
        points[...,2,0] = depth                                     # depth复制到points中
        points = torch.cat(                                         # UV = UV*z
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))  # [1,6,3,3]                            
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
                                                                    # 图像坐标系 -> Camera坐标系 -> Lildar坐标系。 更改的应该只是UV,深度信息在这个转换中不变 
        

        if "extra_rots" in kwargs:                                  #Nuscenes数据集没有对Lildar点云进行额外的处理（缩放、平移等），因此这里是空操作
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )

        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
       
        return points
    
    """
    这个新增加的函数depth_32_88(), 输入d=[1,6,1,256,704],是点云投影到6个image的深度(在Camera坐标系中的)。输出
    是[1,6,1,32,88]，它是按照高、宽每8个像素，聚合在每个格子中的点云的平均深度。
    注意这里有两个版本的depth_32_88(). 这个已经不用，改名为depth_32_88_NO_USE(). 这个是手写的版本，速度比另外一
    个版本慢很多(1100ms)。另一个版本是GPT给出的性能优化的版本，快了很多很多！！！(需要0.2ms)
    """
    @force_fp32()
    def depth_32_88_NO_USE(self, d):
        
        B, N, DD, DH, DW = d.shape
        H = 32; W = 88
        d_ = d.cpu()
        
        dd = torch.zeros(B, N, 1, H, W).to(d.device)        #创建[1,6,1,32,88]tensor, 并且放入d所在的GPU device

        #d=[1,6,1,256,704]
        for b in range(B):
            for n in range(N):
                d__ = d[b][n][0]
                for h in range (0,DH,8):
                    for w in range(0,DW, 8):
                        blk = d__[h:h+8, w:w+8]
                        sum = blk.sum()
                        cnt = blk.count_nonzero()
                        dd[b,n,0,h//8,w//8]= sum/cnt
        
        nan_indices = torch.nonzero(torch.isnan(dd))
        
        nan_mask = torch.isnan(dd)
        #dd = self.replace_nan_with_mean(dd, nan_mask)

        return dd
    
    """
    chatGPT给出的新版本，速度是手写版本的5000倍！
    这个新增加的函数depth_32_88(), 输入d=[1,6,1,256,704],是点云投影到6个image的深度(在Camera坐标系中的)。输出
    是[1,6,1,32,88]，它是按照高、宽每8个像素，聚合在每个格子中的点云的平均深度。
    """
    def depth_32_88(self, d):
        
        B, N, DD, DH, DW = d.shape
        H = 32; W = 88
        
        d = d.squeeze(2)
        
        # Create a mask for non-zero values
        mask = (d > 0).float()
        
        # Use average pooling to sum the valid values within each 8x8 cell
        sum_depth = torch.nn.functional.avg_pool2d(d, kernel_size=8, stride=8, divisor_override=1)
        
        # Use average pooling to count the non-zero values within each 8x8 cell
        count_valid = torch.nn.functional.avg_pool2d(mask, kernel_size=8, stride=8, divisor_override=1)
        
        # Avoid division by zero
        average_depth = sum_depth / (count_valid + 1e-6)
        
        #replace 0 with inf - 这是后添加，把没有点云的cell 原来为0替换为inf --结果0.5758-->0.5763
        average_depth = torch.where(average_depth >0.10, average_depth, torch.tensor(float('inf'), device='cuda'))

        # Reshape to ensure desired output shape [1, 6, 1, 32, 88]
        average_depth=average_depth.unsqueeze(2)

        return average_depth

    """
    这是对每个图像特征像素中的点云做最小polling. 代码没有优化，硬编码为跟256x704相关的维度
    输入 d=[1,6,1,256,704]
    """
    def depth_32_88_min_z(self, d):
        
        d = d.view(6,256,704)
        
        unfolded = d.unfold(1,8,8).unfold(2,8,8)            #[6,32,88,8,8]
        unfolded = unfolded.contiguous().view(6,32,88,64)
        
        #对<0.1的数据，置为inf。
        unfolded_mask = torch.where(unfolded >0.10, unfolded, torch.tensor(float('inf'), device='cuda'))
        
        #取最小值
        min_vals,_= unfolded_mask.min(dim=-1)               #[6,32,88]
        
        #将inf转为nan --- mini = 0.5757
        # inf保持不变 --- mini = 0.5759
        # inf替为64.0 --- mini = 0.5761
        # inf替换为0  --- mini = 0.5753
        #B = torch.where(min_vals == float('inf'), torch.tensor(float('nan'), device='cuda'), min_vals)
        B = min_vals

        B = B.view(1,6,1,32,88)
        return B

    def _sparse_to_dense(self, sparse_depth, valid_coords, valid_values):
        """
        将稀疏深度图转为稠密深度图
        参数：
            sparse_depth: 稀疏深度图 [1, H, W]
            valid_coords: 有效点坐标 [N, 2] (y,x)
            valid_values: 有效点深度值 [N]
        返回：
            稠密深度图 [1, H, W]
        """
        # 方法1：最近邻插值 (保持原有实现风格)
        dense_depth = sparse_depth.clone()
        
        # 创建网格坐标
        h, w = sparse_depth.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=sparse_depth.device),
            torch.arange(w, device=sparse_depth.device)
        )
        # 计算每个网格点到有效点的距离
        dist = (grid_y.unsqueeze(-1) - valid_coords[:,0])**2 + \
            (grid_x.unsqueeze(-1) - valid_coords[:,1])**2
        
        # 找到最近的有效点索引
        _, min_idx = torch.min(dist, dim=-1)
        
        # 赋值最近的有效点深度
        dense_depth[0, grid_y, grid_x] = valid_values[min_idx]
        
        return dense_depth

        # 方法2：可选的双线性插值 (如需更平滑结果)
        # 可根据实际需求切换不同插值方法

    """
    这是对每个图像特征像素中的点云做最小polling. 代码没有优化，硬编码为跟256x704相关的维度。 
    代码来源于BevDepth， 因为Lift Attend Splat也用来同样的最小池化。
    d=[1,6,1,256,704]
    """

    def depth_32_88_min_pooling(self, lidar_depth):
        valid = (lidar_depth >0.1) & (torch.isfinite(lidar_depth))
        lidar_depth = torch.where(valid, lidar_depth, torch.full_like(lidar_depth, float('inf')))
        lidar_depth = lidar_depth.view(6,32,8,88,8,1)
    
        lidar_depth = lidar_depth.permute(0, 1, 3, 5, 2, 4).contiguous()        # [6,32,88,1,8,8]
        lidar_depth = lidar_depth.view(-1, 64)                                  # [16896,64]

        lidar_depth = torch.min(lidar_depth, dim=-1).values                   # [16896]
        lidar_depth = lidar_depth.view(1,6, 1, 32, 88)                          # [6,1,32,88]



        B, N, _, H, W = lidar_depth.shape
    
        # Reshape to (B*N, 1, H, W) for easier handling
        depth_sparse = lidar_depth.view(B * N, 1, H, W)

        # Create valid mask: 1 for valid points, 0 for invalid (0 or inf)
        valid_mask = torch.isfinite(depth_sparse) & (depth_sparse > 0)

        # Set invalid points to zero (temporarily)
        depth_sparse_cleaned = torch.where(valid_mask, depth_sparse, torch.tensor(0.0, device=depth_sparse.device, dtype=depth_sparse.dtype))
    
        # Interpolation using nearest neighbor on valid mask
        # Important: interpolation needs float mask
        valid_mask_float = valid_mask.float()

        # Fill the missing values by interpolation:
        depth_interp = torch.nn.functional.interpolate(depth_sparse_cleaned, size=(H, W), mode='bilinear', align_corners=False)
        mask_interp = torch.nn.functional.interpolate(valid_mask_float, size=(H, W), mode='bilinear', align_corners=False)
    
        # Avoid division by very small values
        mask_interp = torch.clamp(mask_interp, min=1e-5)

        # Normalize interpolated depth by interpolated mask to preserve original depth scale
        depth_dense = depth_interp / mask_interp
    
        # Reshape back to (B, N, 1, H, W)
        depth_dense = depth_dense.view(B, N, 1, H, W)
        return depth_dense

    def depth_32_88_min_pooling_1(self, lidar_depth):
        lidar_depth = lidar_depth.view(6,32,8,88,8,1)
        
        lidar_depth = lidar_depth.permute(0, 1, 3, 5, 2, 4).contiguous()        # [6,32,88,1,8,8]
        lidar_depth = lidar_depth.view(-1, 64)                                  # [16896,64]

        # 没有点云的地方，原来值是0， 我们替换为最大值。 
        # 试验效果：固定的最大值 比如64.0 比浮动的最大值lidar_depth.max()效果略好。 mini set mAP 0.5757 vs 0.5752
        # 而如果保持原来值0不变则效果不好 (min dataset = 0.5353)
        #
        #gt_depths_tmp = torch.where(lidar_depth == 0.0, lidar_depth.max(),lidar_depth)
        gt_depths_tmp = torch.where(lidar_depth < 0.1, torch.tensor(float(64.0), device='cuda'),lidar_depth)
                                                                                # [16896,64]
        #对所有超过特定深度的数值，钳制在固定数值上--- 效果不明显
        #gt_depths_tmp = torch.where(gt_depths_tmp > 64.0, torch.tensor(float(64.0), device='cuda'),gt_depths_tmp)
                                                                                # [16896,64]
        
        lidar_depth = torch.min(gt_depths_tmp, dim=-1).values                   # [16896]
        lidar_depth = lidar_depth.view(1,6, 1, 32, 88)                          # [6,1,32,88]
        
        #lidar_depth is a sparse points, now to cha-zhi to dense depth map
        mask =torch.isfinite(lidar_depth) & (lidar_depth >0.1)

        return lidar_depth

    """
    将输入的点云 xyz坐标 下采样到特征尺度. t统计8x8格子中点云的个数，然后对x,y,z分别累加再求平均。我们只对
    depth非零(z>1.0) 的xyz进行求平均。

    input xyz = [1,6,3,256, 704]
    output    = [1,6,1, 32,  88, 3]
    """
    def xyz_32_88(self, pt_xyz):
        
        pt_xyz = pt_xyz.permute(0,1,3,4,2)
        pt_xyz = pt_xyz.unsqueeze(2)
        z_mask = pt_xyz[...,2] >1.0
        z_mask = z_mask.view(1,6,1,32,8,88,8)
        pt_xyz = pt_xyz.view(1,6,1,32,8,88,8,3)
        
        pt_mask=pt_xyz*z_mask[...,None]
        pt_sum =pt_mask.sum(dim=(4,6))
        pt_cnt = z_mask.sum(dim=(4,6)).clamp(min=1)

        #pt_xyz = pt_xyz.mean(dim=(4,6))
        pt_xyz = pt_sum/pt_cnt[...,None]


        # replace all zeros poinst with another value. 
        # 没有点云的格子，pt_xyz里面的数值是(0,0,0) 下面的测试是把这些格子替换为：
        #  x = 预设的x格子中心点
        #  y = 预设的y格子中心点
        #  z = 'inf' or 'nan' or '0' or '64.0'   --->结果跟没有处理前类似，mini -->0.5780
        """
        iH, iW = 256, 704
        fH, fW = 32, 88
        D = 1

        xs = (torch.linspace(4.0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (torch.linspace(4.0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        ds = torch.zeros((1,fH,fW), dtype=torch.float)

        frustum = torch.stack((xs, ys, ds), -1).to('cuda')      #[1,32,88,3]
        frustum = frustum.unsqueeze(0).repeat(6,1,1,1,1)        #[6,1,32,88,3
        frustum = frustum.unsqueeze(0)                          #[1, 6,1,32,88,3
        
        zeromask=(pt_xyz==0.0).all(dim=-1)
        pt_xyz[zeromask] = frustum[zeromask]

        pt_xyz = torch.where(pt_xyz ==0.0, torch.tensor(float(64.0), device='cuda'), pt_xyz)    
        """

        return pt_xyz
    
        pass
    """
    这个函数是写来，把depth_32_88()生成的点云深度图，里面的无效数据（nan)剔除。思路是：如果一个8x8像素的格子里面
    没有落入任何点云，就从它的周边格子取点云深度，代替。 
    后来运行的测试效果显示，去不去掉nan的效果是没有差别的，因此就不使用了。参考上面的代码，注释了下面这一行：
    dd = self.replace_nan_with_mean(dd, nan_mask)

    这个函数会很慢、很慢......
    """
    @force_fp32()
    def replace_nan_with_mean(self, tensor, nan_mask):
        # Get the shape of the tensor
        B, N, C, H, W = tensor.shape

        # For each NaN, replace it with the mean of its surrounding elements
        for b in range(B):
            for n in range(N):
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            if nan_mask[b, n, c, h, w]:  # If the element is NaN
                                # Get the neighboring indices
                                neighbors = []
                                for i in range(max(0, h-1), min(H, h+2)):  # Row neighbors
                                    for j in range(max(0, w-1), min(W, w+2)):  # Column neighbors
                                        if (i != h or j != w) and not torch.isnan(tensor[b, n, c, i, j]):  # Skip the NaN element itself
                                            neighbors.append(tensor[b, n, c, i, j])

                                # Calculate the mean of the neighbors (excluding NaN neighbors)
                                neighbors = torch.tensor(neighbors)
                                if neighbors.size(0) > 0:  # Avoid division by zero if no neighbors
                                    tensor[b, n, c, h, w] = neighbors.mean()
                                else:
                                    tensor[b, n, c, h, w] = 0
                                    print(f"replace with zero")
        
        return tensor


    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        torch.cuda.synchronize()
        t0 = time.time()
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        torch.cuda.synchronize()
        t1 = time.time()

        x,bt4,bt3,bt2,bt1,bt0 = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        torch.cuda.synchronize()
        t2 = time.time()

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        torch.cuda.synchronize()
        t3 = time.time()

        return final, t3,t2,t1,t0,bt4,bt3,bt2,bt1,bt0

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    """
    这个函数的设想是，一个图像特征像素上有N个点云。我们不对这些点云做深度做平均或求最小值。相反，我们仍然使用[6x118x32x88]的矩阵，
    依次看这些点的深度，如果它属于哪个depth bin,我们就把对应的depth bin的值加1，即我们用真实的Lidar点云在深度上的分布作为深度预测
    的概率分布。

    #输入A =[6,256,704] 是Lidar投影到图像空间的深度 已经从实际的米转变为depth bin index（0-117）
    #输出B =[6,118,32,80] 是在图像特征深度矩阵上的深度分布
    这个思路效果不好

    """
    def my_count(self, A):
        A = A.view(6,256,704)
        batch=6; A_h=256; A_w=704; B_h=32; B_w=88; value_range=118
        G_w=8; G_h=8

        B=torch.zeros((6,118,32,88), dtype=torch.int32, device='cuda')
        A_grids = A.unfold(1,8,8).unfold(2,8,8)             #[6,32,88,8,8]
        A_grids = A_grids.reshape(6,32,88,-1)               #[6,32,88,64]
        
        batch_indices = torch.arange(6, device='cuda').view(-1,1,1,1)   #[6,1,1,1]
        gridh_indices = torch.arange(32,device='cuda').view(1,-1,1,1)   #[1,32,1,1]
        gridw_indices = torch.arange(88,device='cuda').view(1,1,-1,1)   #[1,1,32,1]

        batch_indices = batch_indices.expand(-1, 32, 88, 64).reshape(-1)
        gridh_indices = gridh_indices.expand(6,-1,88, 64).reshape(-1)
        gridw_indices = gridw_indices.expand(6,32,-1, 64).reshape(-1)
        values = A_grids.reshape(-1)

        B_flat = B.view(6,118,-1)
        indices = torch.stack((batch_indices, values, gridh_indices*88 +gridw_indices), dim=0)
        
        B_flat.scatter_add(dim=2, index=indices[2].view(1,1,-1).expand(6,118,-1), src=torch.ones_like(values, dtype=torch.int32).view(1,1,-1).expand(6,118,-1))
        B=B.view(6,118,32,88).float()
        
        return B
        #与此对应的，是手工版本的实现，慢了许多
        """
        d=torch.ones(6,118,32,88)
        depth = (depth/0.5).long().cpu()
        depth = torch.clamp(depth, min=0, max=117)                              #[1,6,1,256,704]
        depth = depth.view(6,256,704)   
        for c in range (6):                                                     #每个camera                                        
            for h in range(0,256,8):                                            #每8行
                for w in range(0,704,8):                                        #每8列
                    blk = depth[c, h:h+8, w:w+8]
                    blk = blk.reshape(64)
                    for z in range(64):
                        d[c, blk[z],h//8, w//8] +=1                             #+1还是=1好？
        
        """

    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )
        dense_depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )
        pt_xyz = torch.zeros(batch_size, img.shape[1], 3, *self.image_size).to(
            points[0].device                                            #1,6,3,H,W
        )

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)               #[6,3,N]

            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)                       #[6,N,3]

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]                                    #[6,N,2]
            y = cur_coords[:,:,0]
            x = cur_coords[:,:,1]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

                """
                # 新增稠密深度估计 (简单示例使用最近邻插值)
                dense_depth[b, c] = self._sparse_to_dense(
                    depth[b, c], 
                    masked_coords, 
                    masked_dist
                )
                """
                masked_y=y[c, on_img[c]]
                masked_x=x[c, on_img[c]]

                pt_xyz[b,c,0,masked_coords[:, 0], masked_coords[:, 1]] = masked_x
                pt_xyz[b,c,1,masked_coords[:, 0], masked_coords[:, 1]] = masked_y
                pt_xyz[b,c,2,masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        #增加file logging相关概念
        if(self.logging == True):
            self.logging_data['lidar_depth_1']=depth.cpu().numpy()
            self.logging_data['img_feat_256'] =img.cpu().numpy()
            

        #1. 这是原来的代码，通过if(False)注释掉
        if(True):
            torch.cuda.synchronize()
            t0 = time.time()    #t0 = time.perf_counter() 
            
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            torch.cuda.synchronize()
            t1 = time.time()        #t1 = time.perf_counter() 
            
            x = self.get_cam_feats_replace_d(img, depth)
            x = self.get_cam_feats(img, depth)
            torch.cuda.synchronize()

            t2 = time.time()        #t2 = time.perf_counter()  # Record end time
            x, bt3,bt2,bt1,bt0, bbt4,bbt3,bbt2,bbt1,bbt0= self.bev_pool(geom, x)

            torch.cuda.synchronize()
            t3 = time.time()        #t3 = time.perf_counter()  # Record end time
            print(f'-------------original get_geo time={(t1-t0)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')
            print(f'  bt3-bt2={(bt3-bt2)*1000} bt2-bt1={(bt2-bt1)*1000} bt1-bt0={(bt1-bt0)*1000} bbt4-bbt3={(bbt4-bbt3)*1000} bbt3-bbt2={(bbt3-bbt2)*1000} bbt2-bbt1={(bbt2-bbt1)*1000} bbt1-bbt0={(bbt1-bbt0)*1000}')
            print(f'x type={type(x)}')
        #2. 这是对depth进行求平均的方法
        if(False):
            t0 = time.perf_counter() 
            
            dd = self.depth_32_88(depth)                                    #mean pooling
            #dd = self.depth_32_88_min_z(depth)                             #min pooling
            #dd = self.depth_32_88_min_pooling(depth)                       #min pooling version 2

            if(self.logging == True):
                self.logging_data['lidar_gt_1']=dd.cpu().numpy().astype(np.float32)

            t00= time.perf_counter() 
            geom_d1 = self.get_geometry_d1(
                dd,
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            t1 = time.perf_counter() 
            x = self.get_cam_feats_2(img, depth)                #调用新版本的get_cam_feat()函数
            x =x.view(1, 6, self.C, 1, 32, 88)                  #x=[1,6,80,1,32,88]
            x = x.permute(0, 1, 3, 4, 5, 2)                     #x=[1,6,1,32,88,80]
            t2 = time.perf_counter() 
            x = self.bev_pool(geom_d1, x)
            t3 = time.perf_counter() 
            print(f'-------------new get_g3288={(t00-t0)*1000} get_geo={(t1-t00)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')

        #3. 这是将聚合后的点云xyz用于计算frustum
        if(False):
            t0 = time.perf_counter() 
            
            pt_xyz = self.xyz_32_88(pt_xyz)

            if(self.logging == True):
                self.logging_data['lidar_gt_xyz']=pt_xyz.cpu().numpy().astype(np.float32)

            t00= time.perf_counter() 
            geom_d1 = self.get_geometry_d1_xyz(
                pt_xyz,
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            #t1 = time.perf_counter() 
            x = self.get_cam_feats_2(img, depth)                #调用新版本的get_cam_feat()函数
            x =x.view(1, 6, self.C, 1, 32, 88)                  #x=[1,6,80,1,32,88]
            x = x.permute(0, 1, 3, 4, 5, 2)                     #x=[1,6,1,32,88,80]
            #t2 = time.perf_counter() 
            x = self.bev_pool(geom_d1, x)
            #t3 = time.perf_counter() 
            #print(f'-------------new NEW (New2) get_g3288={(t00-t0)*1000} get_geo={(t1-t00)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')

        #4. 使用6x116x32x88矩阵 将特征像素上的点云按照深度，来产生深度概率分布
        if(False):
            t0 = time.perf_counter() 
            
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            t1 = time.perf_counter() 
            x, d = self.get_cam_feats_3(img, depth)
            
            #x=[6,80,32,88] d=[6,118,32,88]
            #depth=[1,6,1,256,704],是点云投影到6个image的深度(在Camera坐标系中的)。
            #然后我们将depth的值离散化到0,118. 按照8x8格子，如果它落在某个depth bin上，该bin的值加1， 否则为0.然后进行softmax()
            
            depth = (depth/0.5).long()
            depth = torch.clamp(depth, min=0, max=117)                            #[1,6,1,256,704]
            dd=self.my_count(depth)
            d=dd.softmax(dim=1)  
            
            x = d.unsqueeze(1) * x.unsqueeze(2)
            
            x = x.view(1, 6, 80,118, 32, 88)            #x =[1,6,80,118,32,88]
            x = x.permute(0, 1, 3, 4, 5, 2)             #x =[1,6,118,32,88,80]
            
            t2 = time.perf_counter()  # Record end time
            x = self.bev_pool(geom, x)
            t3 = time.perf_counter()  # Record end time
            print(f'-------------original get_geo time={(t1-t0)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')

        # 5, 不经过depthnet. 将camera feature 跟lidar 真值直接叉乘
        if(False):
            t0 = time.perf_counter() 
            
            #dd = self.depth_32_88(depth)                                    #mean pooling
            #dd = self.depth_32_88_min_z(depth)                             #min pooling
            dd = self.depth_32_88_min_pooling(depth)                       #min pooling version 2

            if(self.logging == True):
                self.logging_data['lidar_gt_1']=dd.cpu().numpy().astype(np.float32)

            t00= time.perf_counter() 
            geom_d1 = self.get_geometry_d1(
                dd,
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            t1 = time.perf_counter() 
            #x = self.get_cam_feats_4(img, depth)               #调用新版本的get_cam_feat()函数
            x = self.get_cam_feats_4(img)                       #调用新版本的get_cam_feat()函数
            x =x.view(1, 6, self.C, 1, 32, 88)                  #x=[1,6,80,1,32,88]
            x = x.permute(0, 1, 3, 4, 5, 2)                     #x=[1,6,1,32,88,80]

            t2 = time.perf_counter() 
            x = self.bev_pool(geom_d1, x)
            t3 = time.perf_counter() 
            print(f'-------------new 4 get_g3288={(t00-t0)*1000} get_geo={(t1-t00)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')

        # 6, 不经过depthnet. 将camera feature 跟lidar 真值直接叉乘
        if(False):
            t0 = time.perf_counter() 
            
            #dd = self.depth_32_88(depth)                                    #mean pooling
            #dd = self.depth_32_88_min_z(depth)                             #min pooling
            dd = self.depth_32_88_min_pooling(depth)                       #min pooling version 2

            if(self.logging == True):
                self.logging_data['lidar_gt_1']=dd.cpu().numpy().astype(np.float32)

            t00= time.perf_counter() 
            geom_d1 = self.get_geometry_d1(
                dd,
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            t1 = time.perf_counter() 
            x = self.get_cam_feats_4(img)                       #调用新版本的get_cam_feat()函数
            x =x.view(1, 6, self.C, 1, 32, 88)                  #x=[1,6,80,1,32,88]
            x = x.permute(0, 1, 3, 4, 5, 2)                     #x=[1,6,1,32,88,80]

            t2 = time.perf_counter() 
            #x = self.bev_pool(geom_d1, x)
            x, bt3,bt2,bt1,bt0, bbt4,bbt3,bbt2,bbt1,bbt0=self.bev_pool(geom_d1, x)
            t3 = time.perf_counter() 
            print(f'------------->>>>>> new 4 get_g3288={(t00-t0)*1000} get_geo={(t1-t00)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')
            print(f'----------type x ={type(x)}')
        return x
