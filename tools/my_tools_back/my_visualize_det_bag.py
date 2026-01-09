"""
本模块是程序C。 功能和设计思路： 接收点云、图像、矩阵、BBOXES， 将结果可视化。 


1. 屏幕布局

    |-----------|-----------|------------ 
    | Img FL 0  | Img F  1  |  Img FR 2 |
    |-----------|-----------|-----------|
    | Img BL 5  | Img B  4  |  Img BR 3 |
    |-----------|-----------|-----------|
    | Lidar                             |
    |-----------------------------------|
2. 侦听的消息
    - Images            (Image)     6x900x1600
    - Pt                (Image)     NX5
    - BBOXES            (Image)     NX11 (x, y, z, dx, dy, dz, theta, vx, vy, label, score)
    - Lidar2Img         (Image)     6x4x4
    - Img_Aug           (Image)     6x4x4

3. 思路
    - 最先收到的是Image，矩阵； 其次是点云； 最后是BBOX
    - 收到的矩阵、Image、点云都需要缓存。 
    - 收到BBOX以后进行绘图和转换
"""

import rospy
import ros_numpy
import threading    

import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
from mmdet3d.core.bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


#----------------------------------------------------------------------------------------------------
# 将BBOX绘制到900x1600的图像上。 因为是原始尺寸的图像，没有经过缩放裁剪，我们不需要Img_aug矩阵，而只需要
# lidar2img的转换矩阵就可以了. 
#----------------------------------------------------------------------------------------------------
def visualize_camera(
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


#-----------------------------------------------------------------------------------------------------------------
# 下面可视化点云的函数来自bevfusion\mmdet3d\core\utils\visualize.py中，为简便起见直接引用了。原函数将生成的图片存盘，
# 这里改为返回图片用于显示.
#-----------------------------------------------------------------------------------------------------------------

def visualize_lidar(
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    # Save the plot to a BytesIO buffer, save plot as PNG in the buffer
    buf = BytesIO()
    fig.savefig(
        buf,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    buf.seek(0)  # Rewind the buffer to the beginning

    # Convert buffer to NumPy array (image)
    #img_array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode to image (BGR format)

    plt.close(fig)
    return img 




from    sensor_msgs.msg import PointCloud2,Image

PAD_H       = 0
IMAGE_W     = 1280#1600
IMAGE_H     = 720#900
LIDAR_W     = 1080
LIDAR_H     = 1080

BBOX_SCORE  = 0.5
IMAGE_SCALE = 0.6

WINDOW_NAME ="Received Image"
CANVAS_W    = (IMAGE_W + IMAGE_W + IMAGE_W)
CANVAS_H    = (IMAGE_H + IMAGE_H + LIDAR_H)
WINDOW_W    = int((CANVAS_W + 20)*IMAGE_SCALE)
WINDOW_H    = int((CANVAS_H + 20)*IMAGE_SCALE)


cfg_path    = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
checkpoint  = './pretrained/bevfusion-det.pth'
from    torchpack.utils.config import configs
from    mmcv import Config
from    mmdet3d.utils import recursive_eval

class MyListener:
    results= {}                 #保存生成的可视化图片
    topics = {}                 #保存接收到的topic
    cfg = None
    listners={}
    
    #           0           1                   2           3             4          5
    img_x  = [      0,     IMAGE_W,     IMAGE_W*2,          0,      IMAGE_W,  IMAGE_W*2]
    img_y  = [      0,           0,             0,    IMAGE_H,      IMAGE_H,    IMAGE_H]    
    lidar_x= [0]
    lidar_y= [IMAGE_H*2]
    cavans = None
    lock = threading.Lock()
    canvas = np.zeros((CANVAS_H,CANVAS_W,3),dtype=np.uint8)
    window = False

    @staticmethod
    def load_cfg():
        configs.load(cfg_path, recursive=True)
        MyListener.cfg = Config(recursive_eval(configs), filename=cfg_path)
    
    @staticmethod
    def create_window():
        if(MyListener.window == False):
            print(f'window = {MyListener.window}')
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 600)
        MyListener.window = True

    @staticmethod
    def init_ros_topics():
        MyListener.listners['rawimg']            = rospy.Subscriber('/camera/rawimg',   Image,       MyListener.rawimg_callback,       queue_size=3)
        MyListener.listners['lidar2image']       = rospy.Subscriber('/pose/lidar2img',   Image,      MyListener.lidar2image_callback,  queue_size=3)
        MyListener.listners['bboxes']            = rospy.Subscriber('/detection/bboxes', Image,      MyListener.bboxes_callback,       queue_size=3)
        MyListener.listners['points']            = rospy.Subscriber('/lidar/points',     PointCloud2,MyListener.points_callback,       queue_size=3)
        
        rospy.init_node('MY_BEV_C', anonymous=True, disable_signals=True)
        return
    
    @staticmethod
    def rawimg_callback(msg):
        img = ros_numpy.image.image_to_numpy(msg)
        img = img.reshape(6,IMAGE_H,IMAGE_W,3)                 #6x900x1600x3 在发布前变为5400x1600x3，在这里恢复6x900x1600x3
        print(f'img shape={img.shape} type={type(img)}')
        MyListener.topics['rawimg'] = img

        """
        with MyListener.lock:
            #绘制6个图片
            N = MyListener.topics['rawimg'].shape[0]
            for i in range (N):
                #取每一帧图像
                image = MyListener.topics['rawimg'][i]
                x = MyListener.img_x[i]
                y = MyListener.img_y[i]
                MyListener.canvas[y:y+IMAGE_H, x:x+IMAGE_W]= image
        """

    @staticmethod
    def lidar2image_callback(msg):
        img = ros_numpy.image.image_to_numpy(msg)    
        
        MyListener.topics['lidar2image'] = img                 #[6，4，4]
        print(f'lidar2image received')
    @staticmethod
    def points_callback(msg):
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points = np.column_stack((cloud_array['x'],
                                cloud_array['y'],
                                cloud_array['z'],
                                cloud_array['intensity'],
                                cloud_array['time']))        
        MyListener.topics['points'] = points 

        """"
        #绘制Lidar
        x = MyListener.lidar_x[0]
        y = MyListener.img_y[0]
        canvas = visualize_lidar( 
                                        lidar = MyListener.topics['points'],
                                        bboxes=None,
                                        labels=None,
                                        xlim=[MyListener.cfg.point_cloud_range[d] for d in [0, 3]],
                                        ylim=[MyListener.cfg.point_cloud_range[d] for d in [1, 4]],
                                        classes=MyListener.cfg.object_classes,
                                    )
        H,W,C=canvas.shape
        print(f'lidar canvas H={H} W={W} C={C}')
        MyListener.canvas[y:y+H, x:x+W] = canvas
        """

    @staticmethod
    def bboxes_callback(msg):
        bboxes = ros_numpy.image.image_to_numpy(msg)    

        MyListener.topics['bboxes'] = bboxes
        print(f'bboxes received {type(bboxes)}')
        #可视化
        MyListener.visualize(bboxes)
        return


    #--------------------------------------------------------------------------------------
    # bboxes =[N,11] = x, y, z, dx, dy, dz, vx, vy, theta, score, label 
    #--------------------------------------------------------------------------------------
    @staticmethod
    def visualize(bboxes):
        #拆成3个矩阵scores=[N,1], lables=[N,1], scores=[N,9]
        scores = bboxes[:, 9]
        labels = bboxes[:, 10]
        bboxes = bboxes[:, :9]

        #过滤掉分数值低于门限的检测框
        indices = scores >= BBOX_SCORE
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices].astype(np.int8)

        #z坐标调整到中间
        bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

        with MyListener.lock:
            #绘制6个图片
            N = MyListener.topics['rawimg'].shape[0]
            for i in range (N):
                #取每一帧图像
                image = MyListener.topics['rawimg'][i]
                x = MyListener.img_x[i]
                y = MyListener.img_y[i]
                
                MyListener.canvas[y:y+IMAGE_H, x:x+IMAGE_W]= visualize_camera(   
                                            image,
                                            bboxes=bboxes,
                                            labels=labels,
                                            transform=MyListener.topics["lidar2image"][i],
                                            classes=MyListener.cfg.object_classes,
                                        )

            #绘制Lidar
            x = MyListener.lidar_x[0]
            y = MyListener.lidar_y[0]
            canvas = visualize_lidar( 
                                            lidar = MyListener.topics['points'],
                                            bboxes=bboxes,
                                            labels=labels,
                                            xlim=[MyListener.cfg.point_cloud_range[d] for d in [0, 3]],
                                            ylim=[MyListener.cfg.point_cloud_range[d] for d in [1, 4]],
                                            classes=MyListener.cfg.object_classes,
                                        )
        H,W,C=canvas.shape
        print(f'lidar canvas H={H} W={W} C={C}')
        MyListener.canvas[y:y+H, x:x+W] = canvas

    @staticmethod
    def show_canvas():
        with MyListener.lock:
            canvas = cv2.resize(MyListener.canvas, None, fx=IMAGE_SCALE, fy=IMAGE_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, canvas)
            cv2.waitKey(1)

def ros_spin_thread():
    print(f'enter ros spin thread')
    MyListener.init_ros_topics()
    rospy.spin()

import time 
    
def main():
    #初始化节点，注册要侦听的topic，初始化发布消息
    MyListener.load_cfg()
    MyListener.create_window()

    print(f'\n---model loaded, now waiting for incoming topics and spinning')    
    spin_thread = threading.Thread(target=ros_spin_thread)
    spin_thread.start()
    
    while not rospy.is_shutdown():
        MyListener.show_canvas()

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()