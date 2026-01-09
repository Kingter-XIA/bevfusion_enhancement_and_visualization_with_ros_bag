"""
本模块是程序C。 功能和设计思路： 接收点云、图像、矩阵、BBOXES， 将结果可视化。 


1. 屏幕布局

            |-----------|-----------|---------------------|
            | Img FL    | Img F     |  Img FR  |          |
            |-----------|-----------|----------|  MAP     |
            | Img BL    | Img B     |  Img BR  |          |
            |-----------|----------------------|----------|
            |             Lidar                           |
            |                                             |
            |---------------------------------------------| 
2. 侦听的消息
    - Images            (Image)     6x900x1600
    - Pt                (Image)     NX5
    - BBOXES            (Image)     NX11 (x, y, z, dx, dy, dz, theta, vx, vy, label, score)
    - Lidar2Img         (Image)     6x4x4
    - Img_Aug           (Image)     6x4x4
    - char                      
3. 思路
    - 最先收到的是Image，矩阵； 其次是点云； 最后是BBOX
    - 收到的矩阵、Image、点云都需要缓存。 
    - 收到BBOX以后进行绘图和转换
"""

import rospy
import ros_numpy
import threading    

import  copy
import  os
from    typing import List, Optional, Tuple

import  cv2
import  numpy as np
from    matplotlib import pyplot as plt
from    io import BytesIO
from    mmdet3d.core.bbox import LiDARInstance3DBoxes
from    std_msgs.msg import Char
from    std_msgs.msg import String
from    sensor_msgs.msg import PointCloud2,Image

from    torchpack.utils.config import configs
from    mmcv import Config
from    mmdet3d.utils import recursive_eval

BBOX_SCORE  = [0.3, 0.3]
MAP_SCORE   = 0.2
IMAGE_SCALE = 0.6

WINDOW_NAME ="Received Image"

import  time

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]

object_classes   = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
object_cnnames   = {
                    'car':                  '轿车', 
                    'truck':                '货车', 
                    'construction_vehicle': '施工车', 
                    'bus':                  '客车', 
                    'trailer':              '拖车', 
                    'barrier':              '障碍', 
                    'motorcycle':           '摩托车', 
                    'bicycle':              '自行车', 
                    'pedestrian':           '行人', 
                    'traffic_cone':         '交通锥'}



point_cloud_range= [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
map_classes={
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider"
}

cam_classes=[
    "FRONT-LEFT",   #2
    "FRONT",        #0
    "FRONT-RIGHT",  #1
    "BACK-LEFT",    #4
    "BACK",         #3
    "BACK-RIGHT",   #5
]

"""
cam_classes=[
    "左前",         #2
    "前",           #0
    "右前",         #1
    "左后",         #4
    "后",           #3
    "右后",         #5
]
"""

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

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}
"""
MAP_PALETTE = {
    "drivable_area": (255, 0, 0),
    "road_segment": (0, 255, 0),
    "road_block": (128, 128, 128),
    "lane": (0, 0, 255),
    "ped_crossing": (255, 255, 0),
    "walkway": (0, 0, 0),
    "stop_line": (255, 255, 255),
    "carpark_area": (255, 127, 0),
    "road_divider": (192, 192, 0),
    "lane_divider": (0, 192, 192),
    "divider": (192, 0, 192),
}
"""
def visualize_map(
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    masks = masks >=MAP_SCORE
    if masks is not None:
    
        canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = background

        for k, name in enumerate(classes):
            if name in MAP_PALETTE:
                canvas[masks[k], :] = MAP_PALETTE[name]

        top  = 0
        left = 0
        for name,color in MAP_PALETTE.items():
            font_scale = 0.3
            thick_ness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            top_left = (left, top)
            bottom_right = (left+60, top+10)

            #绘制背景橘红色
            cv2.rectangle(canvas, top_left, bottom_right, color, thickness=cv2.FILLED)

            cv2.putText(canvas, name, (left, top+10), font,font_scale, (0,0,0), thick_ness, cv2.LINE_AA)
            top+=12

        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    return canvas

    
#----------------------------------------------------------------------------------------------------
# 将BBOX绘制到900x1600的图像上。 因为是原始尺寸的图像，没有经过缩放裁剪，我们不需要Img_aug矩阵，而只需要
# lidar2img的转换矩阵就可以了. 
#----------------------------------------------------------------------------------------------------
def visualize_camera(
    cam_idx,
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

            # 绘制物体类别
            x = coords[index, 1].astype(np.int)  # 文字左上角坐标
            font_scale = 1.5
            thick_ness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 文字占用的矩形大小
            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)
            #绘制背景
            top_left = (x[0], x[1] - text_height )
            bottom_right = (x[0] + text_width, x[1] + baseline)
            cv2.rectangle(canvas, top_left, bottom_right, OBJECT_PALETTE[name], thickness=cv2.FILLED)
            #写文字：物体类别
            cv2.putText(canvas, name, x, font, font_scale, (255,255,255), thick_ness, cv2.LINE_AA)

        # 绘制相机名称
        font_scale = 3
        thick_ness = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = cam_classes[MyListener.cam_index[cam_idx]]
        # 文字占用的矩形大小
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)
        top_left = (0, 0)
        bottom_right = (0 + text_width+2*baseline, text_height + 2*baseline)

        #绘制背景橘红色
        cv2.rectangle(canvas, (0,0), (text_width+2*baseline, text_height+2*baseline), (255,100,0), thickness=cv2.FILLED)

        cv2.putText(canvas, name, (baseline, text_height+baseline), font,font_scale, (255,255,255), thick_ness, cv2.LINE_AA)

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
    dpi,
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
        dpi=dpi,
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


PADDING=60
class MyListener:
    results = {}                 #保存生成的可视化图片
    topics  = {}                 #保存接收到的topic
    cfg     = None
    listners={}
    show_lidar = False
    lock = threading.Lock()
    window = False

    @staticmethod
    def init(cfg):
        MyListener.IMAGE_TW = cfg.IMAGE_W[cfg.dataset_to_use]               #真实的图像尺寸
        MyListener.IMAGE_TH = cfg.IMAGE_H[cfg.dataset_to_use] 
        MyListener.IMAGE_W = cfg.IMAGE_W[cfg.dataset_to_use] + PADDING      #加上padding
        MyListener.IMAGE_H = cfg.IMAGE_H[cfg.dataset_to_use] + PADDING

        print(f'IMAGE W H={MyListener.IMAGE_W} {MyListener.IMAGE_W}')
        layout    = cfg.configs[cfg.config_to_use]

        MyListener.dataset   = cfg.dataset[cfg.dataset_to_use]
        MyListener.cam_order = cfg.cam_order[cfg.dataset_to_use]
        MyListener.cam_index = cfg.cam_index[cfg.dataset_to_use]
        MyListener.show_map  = cfg.configs[cfg.config_to_use].show_map
        MyListener.layout    = cfg.configs[cfg.config_to_use].layout
        MyListener.show_gt_bboxes =cfg.configs[cfg.config_to_use].show_gt_bboxes

        if(MyListener.layout== 'type0'):
            #################################################
            # Image 1   #   Image 2   # Image 3   #         #
            #######################################   MAP   #
            # Image 4   #   Image 5   # Image 6   #         #
            #################################################
            #                                               #
            #         Lidar                                 #
            #         ...                                   #
            #################################################
            #设置尺寸
            if(MyListener.show_map):
                MyListener.MAP_W   = MyListener.IMAGE_H*2 -PADDING
                MyListener.MAP_H   = MyListener.IMAGE_H*2 -PADDING
            else:
                MyListener.MAP_W   = 0
                MyListener.MAP_H   = 0

            if(MyListener.show_lidar):                
                MyListener.LIDAR_H = (MyListener.IMAGE_W*3 +MyListener.MAP_W)
                MyListener.LIDAR_W = (MyListener.IMAGE_W*3 +MyListener.MAP_W)
            else:
                MyListener.LIDAR_H = 0
                MyListener.LIDAR_W = 0

            #设置各图像位置坐标
            MyListener.img_x = [[None for _ in range(6)]]
            MyListener.img_y = [[None for _ in range(6)]]            
            for row in range(2):
               for col in range(3):
                   idx = MyListener.cam_order[row*3+col]
                   MyListener.img_x[0][idx] = MyListener.IMAGE_W*col +  PADDING//2
                   MyListener.img_y[0][idx] = MyListener.IMAGE_H*row +  PADDING//2
            #Lidar点云坐标
            MyListener.lidar_x= [PADDING//2]
            MyListener.lidar_y= [MyListener.IMAGE_H*2 + PADDING//2]
            #MAP坐标
            MyListener.map_x  = [MyListener.IMAGE_W*3 + PADDING//2]
            MyListener.map_y  = [PADDING//2]
            #屏幕尺寸
            MyListener.CANVAS_W    = (MyListener.IMAGE_W*3 + MyListener.MAP_W + PADDING)
            MyListener.CANVAS_H    = (MyListener.IMAGE_H*2 + MyListener.LIDAR_H + PADDING)

        if(MyListener.layout== 'type1'):
            #################################################
            # Image 1   #   Image 1   #                     #
            ###########################                     #
            # Image 2   #   Image 2   #       gt lidar      #
            ###########################                     #
            # Image 3   #   Image 3   #                     #
            #################################################
            # Image 4   #   Image 4   #                     #
            ###########################                     #
            # Image 5   #   Image 5   #       det lidar     #
            ###########################                     #
            # Image 6   #   Image 6   #                     #
            #################################################
            #地图尺寸
            MyListener.MAP_W   = 0                      #不显示地图
            MyListener.MAP_H   = 0
            #点云尺寸 
            if(MyListener.show_lidar):                
                MyListener.LIDAR_H = (MyListener.IMAGE_H*3 -PADDING)
                MyListener.LIDAR_W = (MyListener.IMAGE_H*3 -PADDING)
            else:
                MyListener.LIDAR_H = 0
                MyListener.LIDAR_W = 0

            #图像坐标
            MyListener.img_x = [[None for _ in range(6)] for _ in range(2)]             #2x6 empty list
            MyListener.img_y = [[None for _ in range(6)] for _ in range(2)]             #2x6 empty list
            for col in range(2):
               for row in range(6):
                   idx = MyListener.cam_order[row]
                   MyListener.img_x[col][idx] = MyListener.IMAGE_W*col +PADDING//2
                   MyListener.img_y[col][idx] = MyListener.IMAGE_H*row +PADDING//2
            #点云坐标
            MyListener.lidar_x= [MyListener.IMAGE_W*2 +PADDING//2,  MyListener.IMAGE_W*2 +PADDING//2]
            MyListener.lidar_y= [PADDING//2,                     MyListener.IMAGE_H*3 + PADDING//2]

            #屏幕尺寸
            MyListener.CANVAS_W    = (MyListener.IMAGE_W*2 + MyListener.LIDAR_W +PADDING)
            MyListener.CANVAS_H    = (MyListener.IMAGE_H*6 +PADDING)

        if(MyListener.layout== 'type2'):
            #################################################
            # Image 1   #   Image 2   # Image 3   #         #
            #######################################   MAP   #
            # Image 4   #   Image 5   # Image 6   #         #
            #################################################
            # Image 1   #   Image 2   # Image 3   #         #
            #######################################   MAP   #
            # Image 4   #   Image 5   # Image 6   #         #
            #################################################
            #                      #                        #
            #         Lidar        #      Lidar             #
            #         ...          #                        #
            #################################################
            #地图尺寸
            if(MyListener.show_map):
                MyListener.MAP_W   = MyListener.IMAGE_H*2 -PADDING
                MyListener.MAP_H   = MyListener.IMAGE_H*2 -PADDING  
            else:
                MyListener.MAP_W   = 0
                MyListener.MAP_H   = 0

            #点云尺寸 
            if(MyListener.show_lidar):                
                width = MyListener.IMAGE_W*3 + MyListener.MAP_W
                MyListener.LIDAR_W = width//2 -PADDING
                MyListener.LIDAR_H = width//2 -PADDING
            else:
                MyListener.LIDAR_H = 0
                MyListener.LIDAR_W = 0

            #图像坐标
            MyListener.img_x = [[None for _ in range(6)] for _ in range(2)]             #2x6 empty list
            MyListener.img_y = [[None for _ in range(6)] for _ in range(2)]             #2x6 empty list
            for i in range(2):
                h_offset = i*(MyListener.IMAGE_H*2 + PADDING*5)
                for row in range(2):
                    for col in range(3):
                        idx = MyListener.cam_order[row*3+col]
                        MyListener.img_x[i][idx] = MyListener.IMAGE_W*col + PADDING//2
                        MyListener.img_y[i][idx] = MyListener.IMAGE_H*row + h_offset + PADDING//2

            #MAP坐标
            MyListener.map_x  = [MyListener.IMAGE_W*3 +PADDING//2]
            MyListener.map_y  = [PADDING//2]
            #点云坐标
            MyListener.lidar_x= [PADDING//2,                    (MyListener.IMAGE_W*3 + MyListener.MAP_W)//2 +PADDING//2]
            MyListener.lidar_y= [MyListener.IMAGE_H*4 +PADDING*5 +PADDING//2, MyListener.IMAGE_H*4 +PADDING*5 +PADDING//2]

            #屏幕尺寸
            MyListener.CANVAS_W    = (MyListener.IMAGE_W*3 + MyListener.MAP_W +PADDING)
            MyListener.CANVAS_H    = (MyListener.IMAGE_H*4 + PADDING*5 + MyListener.LIDAR_H+PADDING)

        MyListener.cavans = None
        MyListener.canvas = np.full((MyListener.CANVAS_H,MyListener.CANVAS_W,3), 192,dtype=np.uint8)
        MyListener.dpi    = 46

    @staticmethod
    def set_display_size(config):
        if(MyListener.layout=='type0'):
            if(MyListener.show_lidar):
                MyListener.LIDAR_H  = (MyListener.IMAGE_W*3 + MyListener.MAP_W)
                MyListener.LIDAR_W  = (MyListener.IMAGE_W*3 + MyListener.MAP_H)
            else:
                MyListener.LIDAR_W     = 0
                MyListener.LIDAR_H     = 0
            MyListener.CANVAS_W = (MyListener.IMAGE_W*3 + MyListener.MAP_W +PADDING)
            MyListener.CANVAS_H = (MyListener.IMAGE_H*2 + MyListener.LIDAR_H+PADDING)

        if(MyListener.layout=='type1'):
            if(MyListener.show_lidar):
                MyListener.LIDAR_H = (MyListener.IMAGE_H*3 - PADDING)
                MyListener.LIDAR_W = (MyListener.IMAGE_H*3 - PADDING)
            else:
                MyListener.LIDAR_W     = 0
                MyListener.LIDAR_H     = 0
            MyListener.CANVAS_W    = (MyListener.IMAGE_W*2 + MyListener.LIDAR_W +PADDING)
            MyListener.CANVAS_H    = (MyListener.IMAGE_H*6 +PADDING)

        if(MyListener.layout=='type2'):
            if(MyListener.show_lidar):
                width = MyListener.IMAGE_W*3 + MyListener.MAP_W
                MyListener.LIDAR_W = width//2 -PADDING
                MyListener.LIDAR_H = width//2 -PADDING
            else:
                MyListener.LIDAR_W     = 0
                MyListener.LIDAR_H     = 0
            
            MyListener.CANVAS_W    = (MyListener.IMAGE_W*3 + MyListener.MAP_W +PADDING)
            MyListener.CANVAS_H    = (MyListener.IMAGE_H*4 + PADDING*5 + MyListener.LIDAR_H+PADDING)

        MyListener.cavans = None
        MyListener.canvas = np.full((MyListener.CANVAS_H,MyListener.CANVAS_W,3), 192,dtype=np.uint8)
        MyListener.dpi    = 46

        return True
    
    @staticmethod
    def create_window():
        if(MyListener.window == False):
            print(f'window = {MyListener.window}')
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 600)
        MyListener.window = True

    @staticmethod
    def init_ros_topics():
        MyListener.listners['rawimg']            = rospy.Subscriber('/camera/rawimg',       Image,       MyListener.rawimg_callback,       queue_size=3)
        MyListener.listners['lidar2image']       = rospy.Subscriber('/pose/lidar2img',      Image,       MyListener.lidar2image_callback,  queue_size=3)
        MyListener.listners['bboxes1']           = rospy.Subscriber('/detection/bboxes1',   Image,       MyListener.bboxes1_callback,      queue_size=3)
        MyListener.listners['bboxes']            = rospy.Subscriber('/detection/bboxes',    Image,       MyListener.bboxes_callback,       queue_size=3)
        MyListener.listners['map']               = rospy.Subscriber('/detection/map',       Image,       MyListener.map_callback,          queue_size=3)
        MyListener.listners['points']            = rospy.Subscriber('/lidar/points',        PointCloud2, MyListener.points_callback,       queue_size=3)
        MyListener.listners['char_topic']        = rospy.Subscriber('char_topic',           Char,        MyListener.char_callback,         queue_size=3)
        MyListener.listners['gt_bboxes']         = rospy.Subscriber('/bboxes/gt',           Image,       MyListener.gt_bboxes_callback,    queue_size=3)
 
        #MyListener.listners['dataset']           = rospy.Subscriber('/dataset',          String,      MyListener.dataset_callback,      queue_size=3)

        MyListener.publisher = rospy.Publisher('char_topic', Char, queue_size=10)
        
        rospy.init_node('MY_BEV_C', anonymous=True, disable_signals=True)
        return True
    
    @staticmethod
    def dataset_callback(msg):
        #MyListener.set_display_size(config="")#msg.data)           
        pass

    @staticmethod
    def char_callback(msg):
        MyListener.set_display_size(config="")#dataset="")       
        MyListener.visualize()

    @staticmethod
    def gt_bboxes_callback(msg):
        gt_bboxes = ros_numpy.image.image_to_numpy(msg)    

        MyListener.topics['gt_bboxes'] = gt_bboxes
        print(f'gt----------bboxes received {(gt_bboxes.shape)}')
        
    @staticmethod
    def map_callback(msg):
        masks = ros_numpy.image.image_to_numpy(msg)
        _, N = masks.shape
        masks = masks.reshape(6,-1,N)                       #6x900x1600x3 在发布前变为5400x1600x3，在这里恢复6x900x1600x3
        print(f'map shape={masks.shape} type={type(masks)}')
        MyListener.topics['masks'] = masks

    @staticmethod
    def rawimg_callback(msg):
        print('img received!')
        img = ros_numpy.image.image_to_numpy(msg)
        img = img.reshape(6, MyListener.IMAGE_TH, MyListener.IMAGE_TW, 3)                 #6x900x1600x3 在发布前变为5400x1600x3，在这里恢复6x900x1600x3
        print(f'img shape={img.shape} type={type(img)}')
        MyListener.topics['rawimg'] = img

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

    @staticmethod
    def bboxes1_callback(msg):
        bboxes1 = ros_numpy.image.image_to_numpy(msg)    

        MyListener.topics['bboxes1'] = bboxes1
        print(f'bboxes-1 received {type(bboxes1)} {bboxes1.shape}')

    @staticmethod
    def bboxes_callback(msg):
        bboxes = ros_numpy.image.image_to_numpy(msg)    

        MyListener.topics['bboxes'] = bboxes
        print(f'bboxes received {type(bboxes)} {bboxes.shape}')
        
        MyListener.visualize()
        return


    #--------------------------------------------------------------------------------------
    # bboxes =[N,11] = x, y, z, dx, dy, dz, theta, vx, vy, score, label 
    # gt box =[N,9]  = x, y, z, dx, dy, dz, theta, vx, vy, label
    #--------------------------------------------------------------------------------------
    @staticmethod
    def visualize():
        if(MyListener.layout=='type1') or (MyListener.layout=='type2'):                         #显示两个模型的推理结果，或一个推理+一个真值
            bboxes=[None, None]
            labels=[None, None]
            if(MyListener.show_gt_bboxes):                      #GT + DET
                topics=['gt_bboxes', 'bboxes']
                label_col = [9, 10]
                score_col = [-1, 9]
            else:
                topics=['bboxes1', 'bboxes']
                label_col = [10, 10]
                score_col = [ 9,  9]
        else:
            topics=['bboxes']
            bboxes=[None]
            labels=[None]
            label_col = [10]
            score_col = [9]

        for i in range(len(bboxes)):                
            bboxes[i] = MyListener.topics[topics[i]].copy() 
            if(score_col[i] != -1):                             #推理结果
                scores = bboxes[i][:, score_col[i]]             #拆成3个矩阵scores=[N,1], lables=[N,1], scores=[N,9]                
                labels[i] = bboxes[i][:, label_col[i]]              
                bboxes[i] = bboxes[i][:, :7]

                indices = scores >= BBOX_SCORE[i]                  #过滤掉分数值低于门限的检测框
                bboxes[i] = bboxes[i][indices]
                labels[i] = labels[i][indices].astype(np.int8)
            else:
                labels[i] = bboxes[i][:, label_col[i]].astype(np.int8)
                bboxes[i] = bboxes[i][:, :7]                
                                    
            bboxes[i][..., 2] -= bboxes[i][..., 5] / 2      #z坐标调整到中间
            bboxes[i] = LiDARInstance3DBoxes(bboxes[i], origin=(0.5, 0.50, 0), box_dim=7)

        with MyListener.lock:
            for box_i in range(len(bboxes)):                

                #绘制6个图片
                N = MyListener.topics['rawimg'].shape[0]
                for i in range (N):
                    #取每一帧图像
                    image = MyListener.topics['rawimg'][i]
                    x = MyListener.img_x[box_i][i]
                    y = MyListener.img_y[box_i][i]
                    
                    canvas = visualize_camera(  i,   
                                                image,
                                                bboxes=bboxes[box_i],
                                                labels=labels[box_i],
                                                transform=MyListener.topics["lidar2image"][i],
                                                classes=object_classes,
                                            )
                    MyListener.canvas[y:y+MyListener.IMAGE_TH, x:x+MyListener.IMAGE_TW]= canvas
            
            """
            #Layout2
            #   - 如果显示真值，就显示真值
            #   - 否则应该有两个推理结果显示
            if(MyListener.layout=='type1'):
                if(MyListener.show_gt_bboxes):
                    bboxes = MyListener.topics['gt_bboxes'] 
                    gt_labels = bboxes[:, 9].astype(np.int8)
                    bboxes = bboxes[:, :7]
                
                    labels = gt_labels
                    #z坐标调整到中间
                    bboxes[..., 2] -= bboxes[..., 5] / 2
                    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)

                    for i in range (N):
                        #取每一帧图像
                        image = MyListener.topics['rawimg'][i]
                        x = MyListener.img_x[1][i]
                        y = MyListener.img_y[1][i]
                        canvas = visualize_camera(  i,   
                                                    image,
                                                    bboxes=bboxes,
                                                    labels=labels,
                                                    transform=MyListener.topics["lidar2image"][i],
                                                    classes=object_classes,
                                                )
                        MyListener.canvas[y:y+MyListener.IMAGE_H, x:x+MyListener.IMAGE_W]= canvas
                else:
                    pass
            """
            if(MyListener.show_map == True):
                canvas = visualize_map(masks=MyListener.topics['masks'],
                                    classes = map_classes)
                if canvas is not None:
                    H,W,C=canvas.shape
                    canvas = cv2.resize(canvas, (MyListener.MAP_W, MyListener.MAP_H), interpolation=cv2.INTER_NEAREST)
                    x = MyListener.map_x[0]
                    y = MyListener.map_y[0]
                    MyListener.canvas[y:y+MyListener.MAP_H, x:x+MyListener.MAP_W] = canvas
                    print(f'map x y ={x} {y} can={canvas.shape}')
            
            if(MyListener.show_lidar == True):      
                for  i in range(len(bboxes)):      
                    x = MyListener.lidar_x[i]
                    y = MyListener.lidar_y[i]
                    canvas = visualize_lidar( 
                                                    lidar = MyListener.topics['points'],
                                                    bboxes=bboxes[i],
                                                    labels=labels[i],
                                                    xlim=[point_cloud_range[d] for d in [0, 3]],
                                                    ylim=[point_cloud_range[d] for d in [1, 4]],
                                                    classes=object_classes,
                                                    dpi = MyListener.dpi
                                                )
                    H,W,C=canvas.shape
                    print(f'lidar canvas H={H} W={W} C={C} {MyListener.canvas.shape} x={x} y={y}')
                    print(f'y:y+H={y}-{y+H} x:x+W={x}-{x+W}')

                    H,W,C=canvas.shape
                    canvas = cv2.resize(canvas, (MyListener.LIDAR_W, MyListener.LIDAR_H), interpolation=cv2.INTER_NEAREST)
                    MyListener.canvas[y:y+MyListener.LIDAR_H, x:x+MyListener.LIDAR_W] = canvas
                

    @staticmethod
    def show_canvas():
        with MyListener.lock:
            canvas = cv2.resize(MyListener.canvas, None, fx=IMAGE_SCALE, fy=IMAGE_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(30)&0xff
            if key == ord('l'):
                if(MyListener.show_lidar ==False):
                    MyListener.show_lidar = True
                else:
                    MyListener.show_lidar = False
                msg = Char()
                msg.data = ord('l')  # send character 'L' (as int)
                MyListener.publisher.publish(msg)

def ros_spin_thread():
    print(f'enter ros spin thread')
    MyListener.init_ros_topics()
    rospy.spin()

import time 
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=False, default="tools/abc/infer_cfg.yaml", type=str, help="provide config file for this module")
    args = parser.parse_args()


    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    re = MyListener.init(cfg.visualize)


    #初始化节点，注册要侦听的topic，初始化发布消息
    #re = MyListener.set_display_size(config)#dataset=args.dataset)

    if(re == False):
        print("\ninvalid dataset provided, supported: bag nuscenes")
        exit(0)
    MyListener.create_window()

    print(f'\n---model loaded, now waiting for incoming topics and spinning')    
    spin_thread = threading.Thread(target=ros_spin_thread)
    spin_thread.start()
    
    while not rospy.is_shutdown():
        MyListener.show_canvas()

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()