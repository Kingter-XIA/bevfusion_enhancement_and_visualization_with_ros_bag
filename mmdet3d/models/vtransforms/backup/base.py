from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool
import time 

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
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        ds = torch.zeros((1,fH,fW), dtype=torch.float)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    """
    这个函数是get_geometry()的对应版本。相比于get_geometry_d1()，本函数多传入了一个depth=[1,6,1,32,88]的数据结构。

    self.frustum_d1从[1,32,88,3] --> [1,6,1,32,88,3]. 实际的深度信息被赋值，然后再转换回Lildar坐标系。
    """
    @force_fp32()
    def get_geometry_d1(self, 
                        depth,                                  #input depth = [1,6,1,32,88]  
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

        print(f'dd max ={dd.max()} min={dd.min()}')
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
        
        # Reshape to ensure desired output shape [1, 6, 1, 32, 88]
        average_depth=average_depth.unsqueeze(2)

        return average_depth

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

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

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
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

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

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        #这是原来的代码，通过if(False)注释掉
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
            x = self.get_cam_feats(img, depth)
            t2 = time.perf_counter()  # Record end time
            x = self.bev_pool(geom, x)
            t3 = time.perf_counter()  # Record end time
            print(f'-------------original get_geo time={(t1-t0)*1000} get_cam_feats_time={(t2-t1)*1000} bev_pool time={(t3-t2)*1000} ')

        if(True):
            t0 = time.perf_counter() 
            
            dd = self.depth_32_88(depth)
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

        return x
