"""
读取nuscenese  pkl 文件，从里面依次读取每个sample数据，转换成model.forward()需要的格式，进行推理、评估、或训练。
从PKL读取和准备数据，原来的代码基本在./mmdet3d/datasets/pipelines目录下，按照torch的pipeline的机制来实现，依次
调用下列函数
    - LoadMultiViewImageFromFiles()
    - LoadPointsFromFile()
    - LoadPointsFromMultiSweeps()
    - LoadAnnotations3D()
    - ImageAug3D()
    - GlobalRotScaleTrans()
    - LoadBEVSegmentation()
    - PointsRangeFilter()
    - ImageNormalize()
    - DefaultFormatBundle3D()
    - Collect3D()

几点重要的说明:
1. test时读取数据的pipeline,定义在./configs./default.yaml里面。直接阅读yaml文件，可读性不好，可以将加载后的config
   文件print()出来：
    cfg_path    = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
    configs.load(cfg_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=cfg_path)
    print(cfg.data.test.pipeline)
  
  下面是部分打印的结果：
    {'type': 'LoadMultiViewImageFromFiles', 'to_float32': True}, 
    {'type': 'LoadPointsFromFile', 'coord_type': 'LIDAR', 'load_dim': 5, 'use_dim': 5, 'reduce_beams': 32, 'load_augmented': None}, 
    {'type': 'LoadPointsFromMultiSweeps', 'sweeps_num': 9, 'load_dim': 5, 'use_dim': 5, 'reduce_beams': 32, 'pad_empty_sweeps': True, 'remove_close': True, 'load_augmented': None}, 
    {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True, 'with_attr_label': False}, 
    ....

2. 这些pipleline，不仅读取图像，点云，旋转矩阵等，也读取ground true真值表。
   因此，审视一下这些pipeline函数和它读取的数据，会发现对推理和test来说，有些数据是不需要的。对训练来说，这些数据则是必要的。
   比如3DBOX的真值，物体类型的真值，训练的时候是需要它们，来计算损失函数的。
   不过原来的代码并没有针对训练或测试，对此做特别的优化来区分：比如训练时读取所有类型数据，test时少读取一些数据，等。

3. 依次调用这些piplepine函数后，数据类型都被包装成了mmcv的DataContainer()类型，不适合直接传入model.forward()函数。 
   其实原来torch.dataloader代码在调用完pipeline后，会调用一个collate()函数，这个函数被mmcv重新定义实现。我们如果自己
   读取数据，最后也调用mmcv.collate()函数。collate()函数最后把数据类型从DataContainer转换为模型需要的类型

4. 观察pipeline的代码，每个pipeline都实现为一个类。而且每个类的初始化函数，需要一些默认参数的设置。
   这些参数的设置也是来自config.yaml文件。 比如上面的LoadMultiViewImageFromFiles()，它有一个参数to_float32，默认值是True.
   这个参数会传入比如上面的LoadMultiViewImageFromFiles类的init函数。 
   

本程序的实现：
1. 把所有用到的Pipeline类，源代码全部复制到本py程序里面，这样就是一个完全独立的实现。
2. 新构建了一个NuscenesLoadData类，它提供了读取PKL文件、初始化各个pipeline函数、读取sample数据等各种功能的封装，也在本程序里面。
3. NuscenesLoadData类初始化时，会用config里面的pipeline数据，来初始化各个pipeline类。
4. NuscenesLoadData提供了接口读取一个sample数据。它会直接依次调用本PY中的各个pipeline类的函数来读取

本程序的参数：
本程序实现test.py的功能，但是不需要提供命令行参数。它代码写死了下面的参数：
  pkl_path    = './data/nuscenes/nuscenes_infos_val.pkl'
  cfg_path    = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
  checkpoint  = './pretrained/bevfusion-det.pth'


"""

import  torch
import  torchvision
import  pickle
import  numpy as np
from    pyquaternion import Quaternion
from    typing import Any, Dict, Tuple
from    PIL import Image
from    torchpack.utils.config import configs
from    mmcv import Config, DictAction

from    nuscenes.map_expansion.map_api import NuScenesMap
from    nuscenes.map_expansion.map_api import locations as LOCATIONS

from    mmdet3d.models import build_model
from    mmdet3d.core.bbox import LiDARInstance3DBoxes
from    mmdet3d.core.bbox import get_box_type
from    mmdet3d.datasets.pipelines.loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
from    mmdet.datasets.pipelines import LoadAnnotations
from    mmdet.datasets.pipelines import to_tensor
from    mmdet3d.core.points import BasePoints, get_points_type
from    mmdet3d.utils import recursive_eval
from    mmdet3d.datasets import build_dataloader, build_dataset
from    mmdet.apis import multi_gpu_test, set_random_seed

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import collate
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints


pkl_path    = '../data/nuscenes_mini/nuscenes_infos_val.pkl'
cfg_path    = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
checkpoint  = '../pretrained/bevfusion-det.pth'

###################################################################################################
#以下是Pipeline类
#
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points

        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"

class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results

class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results

class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_masks_bev"] = labels
        return data



class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        return data


class GlobalRotScaleTrans:
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = np.random.uniform(*self.resize_lim)
            theta = np.random.uniform(*self.rot_lim)
            translation = np.array([np.random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data

class PointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data



class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


class DefaultFormatBundle3D:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(
        self,
        classes,
        with_gt: bool = True,
        with_label: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "points" in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = DC(results["points"].tensor)

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results["gt_labels"] = [
                        np.array(
                            [self.class_names.index(n) for n in res], dtype=np.int64
                        )
                        for res in results["gt_names"]
                    ]
                elif "gt_names" in results:
                    results["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names"]],
                        dtype=np.int64,
                    )
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if "gt_names_3d" in results:
                    results["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names_3d"]],
                        dtype=np.int64,
                    )
        if "img" in results:
            results["img"] = DC(torch.stack(results["img"]), stack=True)

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))
        return results


class Collect3D:
    def __init__(
        self,
        keys,
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ),
        meta_lis_keys=(
            "filename",
            "timestamp",
            "ori_shape",
            "img_shape",
            "lidar2image",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "token",
            "pcd_scale_factor",
            "pcd_rotation",
            "lidar_path",
            "transformation_3d_flow",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        # [fixme] note: need at least 1 meta lis key to perform training.
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``metas``
        """
        data = {}
        for key in self.keys:
            if key not in self.meta_keys:
                data[key] = results[key]
        for key in self.meta_keys:
            if key in results:
                val = np.array(results[key])
                if isinstance(results[key], list):
                    data[key] = DC(to_tensor(val), stack=True)
                else:
                    data[key] = DC(to_tensor(val), stack=True, pad_dims=1)

        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]

        data["metas"] = DC(metas, cpu_only=True)
        return data

#
#以上是Pipeline类
###################################################################################################

class NuscenesLoadData():
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    def __init__(self,
                 box_type_3d ='LiDAR',
                 **cfg
                 ):
        self.with_velocity = True
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        
        self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.use_valid_flag = True

        # 传入的cfg中pipeline 是一个list,里面每一个项都是dict. 我们依次取出每个dict, 复制一份（不修改cfg中原始数据）到load_stage，
        # 然后调用load_stage.pop('type') 把dict中的type类型的键和对应数据剔除出去， 对应的数据并且返回赋值给value. 
        # 然后依次对比value，初始化对应的类并传入dict中剩余的键和参数 
        for load_stage in cfg['pipeline']:
            load_stage = load_stage.copy()                              
            value = load_stage.pop('type')
            
            if(value == 'LoadMultiViewImageFromFiles'):
                self.pipeline_LoadMultiViewImageFromFiles=LoadMultiViewImageFromFiles(**load_stage)
            elif(value == 'LoadPointsFromFile'):
                self.pipeline_LoadPointsFromFile=LoadPointsFromFile(**load_stage)
            elif(value == 'LoadPointsFromMultiSweeps'):
                self.pipeline_LoadPointsFromMultiSweeps=LoadPointsFromMultiSweeps(**load_stage)
            elif(value == 'LoadAnnotations3D'):
                self.pipeline_LoadAnnotations3D=LoadAnnotations3D(**load_stage)
            elif(value == 'ImageAug3D'):
                self.pipeline_ImageAug3D=ImageAug3D(**load_stage)
            elif(value == 'GlobalRotScaleTrans'):
                self.pipeline_GlobalRotScaleTrans=GlobalRotScaleTrans(**load_stage)
            elif(value == 'LoadBEVSegmentation'):
                self.pipeline_LoadBEVSegmentation=LoadBEVSegmentation(**load_stage)
            elif(value == 'PointsRangeFilter'):
                self.pipeline_PointsRangeFilter=PointsRangeFilter(**load_stage)
            elif(value == 'ImageNormalize'):
                self.pipeline_ImageNormalize=ImageNormalize(**load_stage)
            elif(value == 'DefaultFormatBundle3D'):
                self.pipeline_DefaultFormatBundle3D=DefaultFormatBundle3D(**load_stage)
            elif(value == 'Collect3D'):
                self.pipeline_Collect3D=Collect3D(**load_stage)
        
        """
        #初始化每个pipeline类，共11个。这种代码写法不好，需要优化...
        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'LoadMultiViewImageFromFiles']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_LoadMultiViewImageFromFiles=LoadMultiViewImageFromFiles(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'LoadPointsFromFile']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_LoadPointsFromFile=LoadPointsFromFile(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'LoadPointsFromMultiSweeps']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_LoadPointsFromMultiSweeps=LoadPointsFromMultiSweeps(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'LoadAnnotations3D']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_LoadAnnotations3D=LoadAnnotations3D(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'ImageAug3D']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_ImageAug3D=ImageAug3D(**load_stage)
        
        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'GlobalRotScaleTrans']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_GlobalRotScaleTrans=GlobalRotScaleTrans(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'LoadBEVSegmentation']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_LoadBEVSegmentation=LoadBEVSegmentation(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'PointsRangeFilter']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_PointsRangeFilter=PointsRangeFilter(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'ImageNormalize']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_ImageNormalize=ImageNormalize(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'DefaultFormatBundle3D']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_DefaultFormatBundle3D=DefaultFormatBundle3D(**load_stage)

        load_stage = [stage for stage in cfg['pipeline'] if stage['type'] == 'Collect3D']
        load_stage = load_stage[0].copy()
        load_stage.pop('type', None)
        self.pipeline_Collect3D=Collect3D(**load_stage)
        """

    #加载PKL文件
    def load_pkl(self, pkl_path):
        with open(pkl_path, 'rb') as file:
            self.data_infos=pickle.load(file)['infos']     
            
        self.data_infos_len = len(self.data_infos)
        return True
    
    #返回PKL文件里面sample的个数
    def get_data_infos_len(self):
        return self.data_infos_len 
    
    #返按照index读取sample - 本函数参考来源于./mmdet3d/datasets/nuscenes_dataset.py中的get_data_info()
    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data
    
    #本函数参考来源于./mmdet3d/datasets/nuscenes_dataset.py中的get_ann_info()
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    #准备pipeline
    def pre_pipeline(self, results):
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d
    
        return results
    
    #执行pipeline 
    def pipelines(self, results):
        re = self.pipeline_LoadMultiViewImageFromFiles(results)   
        re = self.pipeline_LoadPointsFromFile(results)   
        re = self.pipeline_LoadPointsFromMultiSweeps(results)
        re = self.pipeline_LoadAnnotations3D(results)
        re = self.pipeline_ImageAug3D(results)
        re = self.pipeline_GlobalRotScaleTrans(results)
        re = self.pipeline_LoadBEVSegmentation(results)
        re = self.pipeline_PointsRangeFilter(results)
        re = self.pipeline_ImageNormalize(results)
        re = self.pipeline_DefaultFormatBundle3D(results)    
        re = self.pipeline_Collect3D(results)
        
        #下面的数据类型，在推理或测试时用不到
        if(False): 
            re.pop('gt_masks_bev')
            re.pop('gt_bboxes_3d')
            re.pop('gt_labels_3d')
        
        #调用mmcv.collate()函数
        return collate([re])
    

################################################################################
# 以下是主程序
################################################################################

#读取config文件
configs.load(cfg_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=cfg_path)
x = cfg.data.test.pipeline
print(f'type x={type(x)} {x[0]}')
#初始化NuscenesLoadData类
LoadData = NuscenesLoadData(**cfg.data.test)

#加载PKL文件
LoadData.load_pkl(pkl_path=pkl_path)

"""
data = LoadData.get_data_info(0)
data = LoadData.pre_pipeline(data)
data = LoadData.pipelines(data)
for key, value in data.items():
    print(f"Key: {key}, Type of Value: {type(value)}")
"""

#调用下面这一行，仅仅是为了跑完所有sample后，统计精度结果
dataset = build_dataset(cfg.data.test)

set_random_seed(0, deterministic=False)

#加载模型
cfg.model.pretrained = None
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
load_checkpoint(model, checkpoint, map_location="cpu")

#使用mmcv的MMDataParallel wrape, 并将model切换到推理模式
model = MMDataParallel(model, device_ids=[0])
model.eval() 


#按照PKL文件中总的sample个数循环，依次读取sample， 执行推理
outputs = []
for i in range (LoadData.get_data_infos_len()):
    data = LoadData.get_data_info(i)
    data = LoadData.pre_pipeline(data)
    data = LoadData.pipelines(data)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)              #获取depth信息
    outputs.extend(result)

#输出评估结果
eval_kwargs = cfg.get("evaluation", {}).copy()
# hard-code way to remove EvalHook args
for key in [
    "interval",
    "tmpdir",
    "start",
    "gpu_collect",
    "save_best",
    "rule",
]:
    eval_kwargs.pop(key, None)

kwargs = {}
eval_kwargs.update(dict(metric='bbox', **kwargs))
print(dataset.evaluate(outputs, **eval_kwargs))
