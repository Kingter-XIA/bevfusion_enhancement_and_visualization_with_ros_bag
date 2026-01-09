from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
import json
import time
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


logging = True
from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
            #设定logging flag
            self.encoders["camera"]["vtransform"].set_logging_flag(logging)


        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                #voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
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
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        下面的代码示例和验证了，可以输入小于6张（比如3张）图片，不用更改其它代码和预训练模型，仍然能够推理。 
        BEVFusion.forward()是推理过程的起点，因此可以在这里完成修改验证。 
        - 第一步，确定输入参数的维度，除了图像从6变为3之外，一些对应的参数也需要调整。使用print()列出各输入参数
          的维度。可以看到，坐标转换矩阵是对应输入图像的数量的。 
            print(f'img = {img.shape}')                                 #[1,6,3,256,704]
            print(f'camera2ego = {camera2ego.shape}')                   #[1,6,4,4]
            print(f'lidar2ego = {lidar2ego.shape}')                     #[1,4,4]
            print(f'lidar2camera = {lidar2camera.shape}')               #[1,6,4,4]
            print(f'lidar2image = {lidar2image.shape}')                 #[1,6,4,4]
            print(f'camera_intrinsics = {camera_intrinsics.shape}')     #[1,6,4,4]
            print(f'camera2lidar = {camera2lidar.shape}')               #[1,6,4,4]
            print(f'img_aug_matrix = {img_aug_matrix.shape}')           #[1,6,4,4]
            print(f'lidar_aug_matrix = {lidar_aug_matrix.shape}')       

        - 第二步，从输入参数里面取前3张以及对应的参数，丢弃其它数据。 
            img                 = img[:,:3,:,:,:]
            camera2ego          = camera2ego[:,:3,:,:]
            lidar2camera        = lidar2camera[:,:3,:,:]
            lidar2image         = lidar2image[:,:3,:,:]
            camera_intrinsics   = camera_intrinsics[:,:3,:,:]
            camera2lidar        = camera2lidar[:,:3,:,:]
            img_aug_matrix      = img_aug_matrix[:,:3,:,:]
        """
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
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
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
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
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":

                feature = self.extract_camera_features(
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
                    metas,
                )

                #保存数据输出到文件
                if(logging):
                    self.encoders["camera"]["vtransform"].logging_data['bev_img']=feature.cpu().numpy()
                
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)

                #保存数据输出到文件
                if(logging):
                    self.encoders["camera"]["vtransform"].logging_data['bev_lidar']=feature.cpu().numpy()

            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        #####################################################################################################
        # 
        # 现在将log数据输出到文件
        # 我们把vtransform中产生的数据写文件，每个sample创建一个独立的目录。 根据输入的数据，每个Sample存在自己的子目录下面。
        # Nuscense数据每个sample有对应的时间数据。通过metas[0]获取timestamp, 将其转为字符串作为子目录的名字。
        # 为什么要把pytorch tensor转为numpy？-只是使用习惯而已，处理numpy数据更熟悉....
        #
        # Vtransform的数据主要在base.py和depth_lss.p中产生，为了获取这些数据，我们在初始化时调用depth_lss的新增加的函数
        # set_logging_flag() 来设置flag。 self.encoders["camera"]["vtransform"].
        # flag使能后， depthlss &base会把必要的数据从GPU复制到CPU内存，加入字典中保存起来。bev空间的feature则是在本文件中添加的.
        # 字典信息和对应的数据维度如下
        #    type = 'lidar_depth_1',     tensor = [1x6x1x256x704]          
        #    type = 'lidar_depth_64',    tensor = [6x64x32x88]               
        #    type = 'img_depth_118',     tensor = [6x118x32x88]          
        #    type = 'img_feat_256',      tensor = [1x6x256x32x88]        
        #    type = 'img_feat_80',       tensor = [6x80x32x88]           
        #    type = 'lidar_gt_1',        tensor = [1x6x1x32x88]            
        #    type = 'bev_lidar',         tensor = [1,256,180,180]
        #    type = 'bev_img',           tensor = [1,80,180,180]

        if(logging):
            #create folder 
            directory = './bev_feat/' + str(metas[0]['timestamp'])
            if not os.path.exists(directory):
                os.makedirs(directory)

            #get logging data 
            logging_data= self.encoders["camera"]["vtransform"].logging_data

            #write feature/depth data 
            for key, value in logging_data.items():            
                if key == 'lidar_depth_1':
                    value.astype(np.float32).tofile(os.path.join(directory,'lidar_depth_1_6_1_256_704.bin'))
                elif key == 'lidar_depth_64':
                    value.astype(np.float32).tofile(os.path.join(directory,'lidar_depth_6_64_32_88.bin'))
                elif key == 'lidar_gt_1':
                    value.astype(np.float32).tofile(os.path.join(directory,'lidar_gt_depth_1_6_1_32_88.bin'))
                elif key == 'img_depth_118':
                    value.astype(np.float32).tofile(os.path.join(directory,'img_depth_6_118_32_88.bin'))
                elif key == 'img_feat_80':
                    value.astype(np.float32).tofile(os.path.join(directory,'img_feat_6_80_32_88.bin'))
                elif key == 'img_feat_256':
                    value.astype(np.float32).tofile(os.path.join(directory,'img_feat_1_6_256_32_88.bin'))
                elif key == 'bev_img':
                    value.astype(np.float32).tofile(os.path.join(directory,'bev_img_1_80_180_180.bin'))
                elif key == 'bev_lidar':
                    value.astype(np.float32).tofile(os.path.join(directory,'bev_lidar_1_256_180_180.bin'))
                elif key == 'lidar_gt_xyz':
                    value.astype(np.float32).tofile(os.path.join(directory,'lidar_gt_xyz_1_6_1_32_88_3.bin'))
            #然后我们保存原始JPG/Lidar文件路径
            img_pts={}
            img_pts['img_path'] = metas[0]['filename']
            img_pts['pts_path'] = metas[0]['lidar_path']

            with open(os.path.join(directory,'img_pts_path.json'), "w") as outfile: 
                json.dump(img_pts, outfile)

        #
        #####################################################################################################

        if not self.training:
            # avoid OOM
            features = features[::-1]
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
