"""
1. 程序目的：
    - 不需要rosbag play bag1.bag
    - 用一个循环读取BAG1文件，读到需要的数据后，进行处理，转换成Model需要的topics发布出去。
    - 每次读到Point数据，发送前等待用户按键。按键后才发送。因此发送的节奏完全受控，方便调试。

2. 简化处理
    - 因为图像和雷达数据的时戳相同，所以我们不需要odom topic. 
    - 如果遇到点云和图像数据时戳v不一致的情形（极少），就丢弃这一次的Points

3. 读取的topics
    ---------------------------------------------------------------------------------------------------------
    |            Topic                 |         Type                        | Comments                     |
    |----------------------------------|-------------------------------------|------------------------------|
    | Topic: /mvs_0/image/compressed   |    sensor_msgs/CompressedImage      | 720x1280 (H=720, W=1280)     |
    |               ...                |                                     |                              |
    | Topic: /mvs_5/image/compressed   |        ""                           |                              |
    | Topic: /ouster/points/relayed    |     sensor_msgs/PointCloud2         |                              |
    ---------------------------------------------------------------------------------------------------------
    
4. 读取处理流程
    - 读取的6个IMG放入6个FIFO，每个FIFO存最近的3个图像
    - 读取到点云后，到FIFO里搜索找出跟点云相同时间的6个图像（如果没有，就跳过这次点云，继续读取）
    - 然后解码点云、图像；生成10个待发布的topics
    - 等待用户按键
    - 发布10个topics 

5. 需要发布的topics
    - (略)

"""

import  numpy as np
import  cv2
import  ros_numpy
import  rosbag
import  rospy
import  torch
import  torchvision.transforms as transforms
import  json
import  readchar

from    sensor_msgs.msg import PointCloud2, Image
from    std_msgs.msg import String
import  sensor_msgs.point_cloud2 as pc2

from    PIL import Image as PILImage
from    collections import deque


IMG_FIFO_LEN= 6#4
PT_FIFO_LEN = 3#2
IMG_NUM     = 6

IMG_MEAN  = [0.485, 0.456, 0.406]                   #用来归一化图像
IMG_STD   = [0.229, 0.224, 0.225]                   #用来归一化图像
IMG_RESIZE= [768, 432]                              #原始图像1280x720缩放0.6倍，得到768x432
IMG_CROP  = [32, 176, 736, 432]                     #从768x432图像中裁剪得到704x256. 这里分别是裁剪的[left,top,right,bottom]

                                                    #相机内参-->转成topic发布用
"""
CAM_INTRINSICS=[[873.7577877659145, 873.2034110094822, 634.5633891268108, 400.18724168491343],
                [864.5661472531169, 864.2714029274775, 654.6327589982127, 406.5554695622828],
                [872.4208787032316, 871.9594518466813, 649.1794522861443, 397.5058462370768],
                [868.6743026179921, 868.5655126767598, 639.5293577023197, 420.82943710736816],
                [864.3609084713726, 864.2636584091672, 666.4607758414149, 408.6655326172872],
                [872.4784134309563, 872.3919077418872, 620.2882826566606, 398.1659802274646]]
"""
CAM_INTRINSICS= [
                [873.7577877659145, 873.2034110094822, 634.5633891268108, 400.18724168491343],
                [871.9266807083293, 871.5427748146391, 642.7138247581352, 353.0016167130286],
                [872.4208787032316, 871.9594518466813, 649.1794522861443, 397.5058462370768],
                [868.6743026179921, 868.5655126767598, 639.5293577023197, 420.82943710736816],
                [872.5925282720806, 872.134337199566, 638.3252526408653, 346.3887183141744],
                [872.4784134309563, 872.3919077418872, 620.2882826566606, 398.1659802274646]]


IMG_AUG  = [0.60, 0.00, 0.00,  -32.00,              #相机裁剪个缩放矩阵-->转成topic发布用
            0.00, 0.60, 0.00, -176.00,
            0.00, 0.00, 1.00,    0.00,
            0.00, 0.00, 0.00,    1.00]
                                                    #相机外参-->转成topic发布用
"""
LIDAR2CAM= [    [      0.873048,     -0.487619,     0.00398964,         0, 
                    -0.00398965,     -0.015324,      -0.999875,     -0.05, 
                       0.487619,      0.872922,      -0.015324,     -0.14, 
                            0.0,           0.0,            0.0,       1.0 ],
                [    -0.0121929,     -0.999925,    0.000767112,      0.02,
                     -0.0627905,    5.82077e-11,     -0.998027,     -0.05, 
                       0.997952,      -0.012217,    -0.0627858,     -0.14, 
                            0.0,            0.0,           0.0,       1.0 ],
                [     -0.866901,      -0.498469,    0.00337762,         0, 
                    -0.00639738,     0.00435007,      -0.99997,     -0.05, 
                       0.498439,      -0.866897,   -0.00695998,     -0.14, 
                            0.0,            0.0,           0.0,       1.0 ],
                [     -0.874644,       0.484736,   -0.00544496,         0, 
                       0.010914,     0.00846109,     -0.999905,     -0.05, 
                      -0.484643,       -0.87462,    -0.0126908,     -0.14, 
                            0.0,            0.0,           0.0,       1.0 ],
                [    -0.0140499,       0.999901,   0.000767124,         0, 
                      0.0697321,     0.00174516,     -0.997564,     -0.05, 
                      -0.997467,     -0.0139622,    -0.0697497,     -0.14, 
                            0.0,            0.0,           0.0,       1.0 ],
                [      0.857211,       0.514959,   -0.00249239,         0, 
                     0.00249239,    -0.00898865,     -0.999956,     -0.05, 
                      -0.514959,       0.857167,   -0.00898865,     -0.14,
                            0.0,            0.0,            0.0,      1.0 ]]
"""
LIDAR2CAM= [
                [ 0.873811, -0.486261, 0.00220264, 0, 
                    -0.00220264, -0.00848772, -0.999962, -0.05, 
                    0.486261, 0.873772, -0.00848772, -0.14, 
                    0, 0, 0, 1],
                [ 0, -1, 0, 0, 
                    -0.00174533, 0, -0.999998, -0.05, 
                    0.999998, 0, -0.00174533, -0.14, 
                    0, 0, 0, 1 ],

                [ -0.866026, -0.499993, -0.00221298, 0, 
                    0.00104385, 0.00261798, -0.999996, -0.05, 
                    0.499997, -0.866025, -0.00174533, -0.14, 
                    0, 0, 0, 1],
                [ -0.872108, 0.489308, 0.00223236, 0, 
                    0.00223236, 0.0085409, -0.999961, -0.05, 
                    -0.489308, -0.872069, -0.0085409, -0.14, 
                    0, 0, 0, 1],
                [ -0.0140223, 0.999878, 0.00685916, 0, 
                    0.00862885, 0.00698058, -0.999938, -0.05, 
                    -0.999864, -0.0139622, -0.00872568, -0.14, 
                    0, 0, 0, 1],
                [0.85807, 0.513522, 0.00325432, 0, 
                    0.00647445, -0.00448144, -0.999969, -0.05, 
                    -0.513491, 0.858065, -0.00717016, -0.14, 
                    0, 0, 0, 1]
                ]

#order =[1,2,0,4,5,3]
fifo_imgs = [deque(maxlen=IMG_FIFO_LEN) for _ in range(IMG_NUM)]        #6个FIFO 每个保存3个IMG(大约150ms范围)
fifo_pts  = deque(maxlen=PT_FIFO_LEN)                                   #FIFO 保存最新的2个点云 
pt_msg_idx= 0

#内参1x4转4x4矩阵
def convert_1x4_to_4x4(intrinsics_1x4):
    fx, fy, cx, cy = intrinsics_1x4
    return np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1]
    ], dtype=np.float32)

#计算4x4 齐次变换矩阵的反推 
def invert_transformation(T):
    R = T[:3, :3]  
    t = T[:3, 3]   

    T_inv = np.eye(4)  
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T@t

    return T_inv.astype(np.float32)

#计算6x4x4 即6个4x4齐次变换矩阵的反推。返回结果也是6x4x4 - 来源于GPT，验证可以正确运行
def invert_batch_transformation(T):
    R = T[:, :3, :3]  
    t = T[:, :3, 3]   

    T_inv = np.eye(4)[None, :, :].repeat(6, axis=0)  
    T_inv[:, :3, :3] = np.transpose(R, (0, 2, 1))                       #直接行列转置，依据R^-1 = R^T
    T_inv[:, :3, 3] = -np.einsum('bij,bj->bi', T_inv[:, :3, :3], t)     # -R^T * t 

    return T_inv.astype(np.float32)

#计算6x4x4 即6个4x4齐次变换矩阵的反推-版本2。
def invert_batch_transformation2(T_batch):
    T_inv = np.array([invert_transformation(T_batch[i]) for i in range(6)])
    return T_inv


#从缓存在FIFO里面的2个points和3x6个image,找出时间戳相等的points和对应的6个image消息
def find_buffered_pts_imgs():
    #1. 确保已经收到了6x3个图像+2个points
    len1 = [len(fifo_imgs[i]) for i in range(6)]
    len2 = len(fifo_pts)
    if(len1!=[IMG_FIFO_LEN,IMG_FIFO_LEN,IMG_FIFO_LEN,IMG_FIFO_LEN,IMG_FIFO_LEN,IMG_FIFO_LEN]) or (len2 !=PT_FIFO_LEN):
        return None, None
    
    #2.搜索 找图像和点云时戳相等
    for pt_msg in fifo_pts:
        t_point = pt_msg.header.stamp.to_sec()              #取一个point msg, 获取点云时间
        img_msgs = []                   
        for fifo_img in fifo_imgs:                          #搜索6个图像FIFO      
            for _img in fifo_img:
                t_img = _img.header.stamp.to_sec()          #图像时间
                if(t_img == t_point):                       #如果时间相等，把该图像加入列表
                    img_msgs.append(_img)
                    break
            if(len(img_msgs)==6):                           #6个图像FIFO搜索结束，看是否找到了6个跟Point时间相等的图像,
                return pt_msg, img_msgs                     #是的话就返回point msg和对应的img msgs; 否则取下一个point msg重新比对

    #2.搜索 找图像和点云时戳相近<20ms
    for pt_msg in fifo_pts:
        t_point = pt_msg.header.stamp.to_sec()              #取一个point msg, 获取点云时间
        img_msgs = []                   
        for fifo_img in fifo_imgs:                          #搜索6个图像FIFO      
            for _img in fifo_img:
                t_img = _img.header.stamp.to_sec()          #图像时间
                delta = abs(t_img - t_point)
                if(delta <0.02):                            #如果时间小于20ms，把该图像加入列表
                    img_msgs.append(_img)
                    break
            if(len(img_msgs)==6):                           #6个图像FIFO搜索结束，看是否找到了6个跟Point时间相等的图像,
                print(f'not find same time pt * image but find neaby <20ms')
                return pt_msg, img_msgs                     #是的话就返回point msg和对应的img msgs; 否则取下一个point msg重新比对


    return None, None                                       #如果走到这里，则无法找到时间相等的points +6 images

# 读取BAG文件，将image和point消息放入FIFO. 读到point msg后，再到
# FIFO里找找有没有时间戳相同的6个图像消息，有就返回。没有就继续读文件。
def read_bag_msgs(bag_msg_generator):
    global pt_msg_idx

    try:
        while (True):
            topic, msg, t =next(bag_msg_generator)
            if topic == '/mvs_0/image/compressed':
                fifo_imgs[0].append(msg)
            elif topic == '/mvs_1/image/compressed':
                fifo_imgs[1].append(msg)
            elif topic == '/mvs_2/image/compressed':
                fifo_imgs[2].append(msg)
            elif topic == '/mvs_3/image/compressed':
                fifo_imgs[3].append(msg)
            elif topic == '/mvs_4/image/compressed':
                fifo_imgs[4].append(msg)
            elif topic == '/mvs_5/image/compressed':
                fifo_imgs[5].append(msg)
            elif topic =='/ouster/points/relayed':
                print(f'---points!!')
                fifo_pts.append(msg)
                pt_msg_idx +=1
                #搜索是不是能找出1个point msg + 对应相同时间的6个image msgs
                pt_msg, img_msgs = find_buffered_pts_imgs()
                
                #找到了就返回，否则继续读BAG
                if(pt_msg is not None):
                    return pt_msg, img_msgs, pt_msg_idx
                
    except StopIteration:
        print('Stope reading bags!')
        return None, None, None

def process_data(pt_msg, img_msgs):       
    #------------------------------------------------------
    # 1. 提取点云数据，剔除无效值，归一化
    #-----------------------------------------------------
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pt_msg)
    p2 = np.stack( [cloud_array['x'],
                    cloud_array['y'],
                    cloud_array['z'],
                    cloud_array['intensity'],
                    cloud_array['t']], axis=-1)
    p2 = p2.reshape(-1,5)                               #Nx5
    p2 = p2[~np.isnan(p2).any(axis=1)]                  #提出含有无效值san的行
    max_intensity = np.max(p2[:,3])                 
    p2[:,3] = p2[:,3]/max_intensity*255.0               #intensity归一化到0-255
    p2[:,4] = 0.                                        #time置0

    #------------------------------------------------------
    # 2. 解码图像，缩放，裁剪，归一化
    #-----------------------------------------------------
    imgs_256 =[]                                #裁剪后的图像
    imgs_raw =[]                                #原始尺寸图像--也发布出去用于可视化
    #2.1 解码，缩放，裁剪
    for img_msg in img_msgs:
        array    = np.frombuffer(img_msg.data, np.uint8)
        cv_image = cv2.imdecode(array, cv2.IMREAD_COLOR)                
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        pil_image= PILImage.fromarray(cv_image)
        pil_image= pil_image.resize(IMG_RESIZE)
        pil_image= pil_image.crop(IMG_CROP)

        imgs_256.append(np.array(pil_image))
        imgs_raw.append(np.array(cv_image))
    
    #2.2 图像从py转为pytorch tensor做归一化(利用它的normalize()），
    #    然后调整从6x256x704x3-->6x3x256x704。 注意归一化后仍然是tensor不是numpy数组,所以转为numpy
    imgs_256 = np.array(imgs_256).astype(np.float32)
    imgs_256 = torch.from_numpy(imgs_256)
    imgs_256 = imgs_256.permute(0,3,1,2)
    normalize= transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    imgs_256 = normalize(imgs_256)
    imgs_256 = imgs_256.numpy()
    #imgs_256 = imgs_256[order]
    
    #2.3 原始尺寸图像也变成6x720x1280x3的numpy
    imgs_raw = np.array(imgs_raw)
    #imgs_raw = imgs_raw[order]
    
    #------------------------------------------------------
    # 3. 准备其它的topic -基本都是常数，所以比较简单
    #-----------------------------------------------------
    intrinsics = np.array([convert_1x4_to_4x4(intrinsics) for intrinsics in CAM_INTRINSICS]).astype(np.float32)   
    #intrinsics = intrinsics[order]
    lidar2cam  = np.array(LIDAR2CAM, dtype=np.float32).reshape((6,4,4))
    #lidar2cam  = lidar2cam[order]
    cam2liadr  = invert_batch_transformation(lidar2cam)
    lidar2ego  = np.eye(4).astype(np.float32)
    lidar_aug  = np.eye(4).astype(np.float32)
    img_aug    = np.tile(np.array(IMG_AUG, dtype=np.float32).reshape((4,4)),(6,1,1))
    lidar2img  = intrinsics@lidar2cam
    
    #------------------------------------------------------
    # 4. 生成待发送topic
    #------------------------------------------------------
    topic_data = {}
    topic_data['rawimg']            = imgs_raw
    topic_data['img']               = imgs_256
    topic_data['points']            = p2
    topic_data['camera2ego']        = cam2liadr
    topic_data['lidar2ego']         = lidar2ego
    topic_data['lidar2camera']      = lidar2cam
    topic_data['lidar2image']       = lidar2img
    topic_data['camera_intrinsics'] = intrinsics
    topic_data['camera2lidar']      = cam2liadr
    topic_data['img_aug_matrix']    = img_aug
    topic_data['lidar_aug_matrix']  = lidar_aug
    topic_data['metas']             = {}

    return topic_data

def publish_topics(topics, data):
    #------------------------------------------------------
    # 1. 发送Img 6x3x256x704
    #-----------------------------------------------------
    img_data = data['img']                                          #取出image数据，转为numpy 
    N,C,H,W = img_data.shape
    img_data = img_data.reshape((N*H, W, C))                        #reshape. 6x3x256x704 --> 1536x704x3

    msg = ros_numpy.image.numpy_to_image(img_data, "32FC3")         #numpy to Image message
    topics['img'].publish(msg)                                      #发布

    #------------------------------------------------------
    # 2. 发送Camera Intrinsics 6x4x4
    #------------------------------------------------------
    img_data = data['camera_intrinsics']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera_intrinsics'].publish(msg)                        #发布

    #------------------------------------------------------
    # 3. 发送img_aug_matrix 6x4x4
    #------------------------------------------------------
    img_data = data['img_aug_matrix']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['img_aug_matrix'].publish(msg)                            #发布

    #------------------------------------------------------
    # 4. 发送 camera2ego 6x4x4
    #------------------------------------------------------
    img_data = data['camera2ego']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera2ego'].publish(msg)                               #发布

    #------------------------------------------------------
    # 5. 发送 lidar2ego 4x4
    #------------------------------------------------------
    img_data = data['lidar2ego']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC1")         #numpy to Image message
    topics['lidar2ego'].publish(msg)                                #发布

    #------------------------------------------------------
    # 6. 发送 lidar2camera 6x4x4
    #------------------------------------------------------
    img_data = data['lidar2camera']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['lidar2camera'].publish(msg)                             #发布

    #------------------------------------------------------
    # 7. 发送 lidar2image 6x4x4
    #------------------------------------------------------
    img_data = data['lidar2image']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['lidar2image'].publish(msg)                              #发布

    #------------------------------------------------------
    # 8. 发送 camera2lidar 6x4x4
    #------------------------------------------------------
    img_data = data['camera2lidar']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera2lidar'].publish(msg)                             #发布

    #------------------------------------------------------
    # 9. 发送 lidar_aug_matrix 4x4
    #------------------------------------------------------
    img_data = data['lidar_aug_matrix']
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC1")         #numpy to Image message
    topics['lidar_aug_matrix'].publish(msg)                         #发布

    #------------------------------------------------------
    # 10. 发送 metas  dict 
    #------------------------------------------------------
    metas = data['metas']                                           #取出metas字典         
    metas['box_mode_3d'] = None                                     #这三项清零
    metas['box_type_3d'] = None
    metas['lidar2image'] = None
    metas = json.dumps(metas)                                       #转成字符串
    topics['metas'].publish(metas)                                  #发布字符串
        
    #------------------------------------------------------
    # 11. 发送 points [N,5]
    #------------------------------------------------------
    pt_data = data['points']

    # 点云numpy转成PointCloud2消息类型也有额外的问题要解决。 原来，ros_numpy不支持把普通的numpy array直接转换为
    # PointCloud2消息，而是支持一种所谓的recordarray，只能将这种格式的array转为ROS PointCloud2消息。 因此我们的
    # 步骤：pt_data先需要转为recordarray, 然后转消息。 record-array是一种特殊的'记录'格式的数组，类似于excel表格
    # 的方式，比如点云数组[1000x5]，则第1列用名字'x'来索引，第2列用'y'索引，第3列用'z'索引，等等。 
    # 
    # 这是在参阅了ros_numpy源代码后，才明白它的机制。 同时，回归几个月前自己实现的将点云bin文件转成PointCloud2消息
    # 发布出去的代码，其实发布过程并没有使用ros_numpy,而是自己从bin里面提取数据,自己构造了PointCloud2消息。虽然是
    # 可以的，但是如果转为record-array再直接使用ros_numpy API，会更简便 
    # github: https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py

    # 点云每个点是5维度，我们定义5维度的类型，名称分别是'x', 'y', 'z', 'intensity', 'time'
    dtype = [('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            ('intensity', np.float32), 
            ('time', np.float32)]

    #将点云数组pt_data转为 record-array类型
    #(转换完后，可以直接用pt_data['x']或pt_data.x来引用第1列--接收方会用这种方式再提取并转换回普通的array...)
    pt_data = np.core.records.fromarrays(pt_data.T, dtype=dtype)
    
    #现在可以调用ros_numpy转消息了
    msg = ros_numpy.point_cloud2.array_to_pointcloud2(pt_data)      #point array to msg  
    topics['points'].publish(msg)                                   #发布
    
    #------------------------------------------------------
    # 12. 发送 raw image 6x 720 x1280 x3
    #------------------------------------------------------    
    img_data = data['rawimg']                                     #取出image数据，转为numpy 

    N,H,W,C = img_data.shape
    img_data = img_data.reshape((N*H, W, C))                        #reshape. 6x900x1600x3 -->5400x1600x3

    msg = ros_numpy.image.numpy_to_image(img_data, "rgb8")          #numpy to Image message
    topics['rawimg'].publish(msg)                                   #发布


import torch

def invalid_image(data):
    img_data = data['img']
    N, C, H, W = img_data.shape
    data['img']=np.zeros((N,C,H,W),dtype=np.float32)
    
    return data

def invalid_points(data):
    pt_data = data['points']
    N, C = pt_data.shape
    data['points'] = np.zeros((N,C),dtype=np.float32)

    return data

def init_ros_publish_topics():
    pub_topics = {}
    pub_topics['dataset']           = rospy.Publisher('/dataset',           String,     queue_size=3)
    pub_topics['rawimg']            = rospy.Publisher('/camera/rawimg',     Image,      queue_size=3)
    pub_topics['img']               = rospy.Publisher('/camera/img',        Image,      queue_size=3)
    pub_topics['camera_intrinsics'] = rospy.Publisher('/camera/intrinsics', Image,      queue_size=3)
    pub_topics['img_aug_matrix']    = rospy.Publisher('/camera/img_aug',    Image,      queue_size=3)
    pub_topics['camera2ego']        = rospy.Publisher('/pose/cam2ego',      Image,      queue_size=3)
    pub_topics['lidar2ego']         = rospy.Publisher('/pose/lidar2ego',    Image,      queue_size=3)
    pub_topics['lidar2camera']      = rospy.Publisher('/pose/lidar2cam',    Image,      queue_size=3)
    pub_topics['lidar2image']       = rospy.Publisher('/pose/lidar2img',    Image,      queue_size=3)
    pub_topics['camera2lidar']      = rospy.Publisher('/pose/cam2lidar',    Image,      queue_size=3)
    pub_topics['lidar_aug_matrix']  = rospy.Publisher('/lidar/lidar_aug',   Image,      queue_size=3)
    pub_topics['points']            = rospy.Publisher('/lidar/points',      PointCloud2,queue_size=3)
    pub_topics['metas']             = rospy.Publisher('/misc/metas',        String,     queue_size=3)
    
    rospy.init_node('MY_BEV_A', anonymous=True, disable_signals=True)
    
    return pub_topics

def main():
    #初始化ROS节点，初始化所有需要发布的topics
    pub_topics = init_ros_publish_topics()

    #设置每秒发送的节奏
    #rate = rospy.Rate(1)   
    
    #打开bag文件
    bag_file = '../ros/bag/data3/data3.bag'
    #bag_file = '../ros/bag/night_wait.bag'
    bag = rosbag.Bag(bag_file)
    start_time = bag.get_start_time()

    end_time = bag.get_end_time()
    print(f'--type ={start_time} {type(start_time)}')
    midpoint = start_time + (end_time- start_time  ) *3/ 4
    midpoint_time = rospy.Time.from_sec(midpoint)
    print(f'--type ={midpoint_time} {type(midpoint_time)}')
    bag_msg_generator = bag.read_messages(start_time=midpoint_time)
    
    msg = "bag"
    pub_topics['dataset'].publish(msg)
    pt_msg = None 
    img_msgs = None
    index    = -1
    try:            
        while(True):
            print(f"\nnew data ready-{index}, press SPACE to publish, ENTER to exit:")
            user_input = readchar.readkey()                         #等待用户按键
            
            if(user_input == '\r' or user_input =='\n'):            #RETURN key to exit
                print("\nUser exit!")
                break

            if(user_input == ' '):              #SPACE
                pt_msg, img_msgs, index = read_bag_msgs(bag_msg_generator)          #读取一个Point及相关的Imgs
                if(pt_msg is None): break
                results = process_data(pt_msg, img_msgs)                            #处理数据，生成待发送topics

            if(user_input == 'i'):              #i
                if(pt_msg is None):             #如果没有数据，读取一个Point及相关的Imgs
                    pt_msg, img_msgs, index = read_bag_msgs(bag_msg_generator)      
                    if(pt_msg is None): break
                results = process_data(pt_msg, img_msgs)                            #处理数据，生成待发送topics
                results = invalid_image(results)

            if(user_input == 'l'):              #l
                if(pt_msg is None):             #如果没有数据，读取一个Point及相关的Imgs
                    pt_msg, img_msgs, index = read_bag_msgs(bag_msg_generator)      
                    if(pt_msg is None): break
                results = process_data(pt_msg, img_msgs)                            #处理数据，生成待发送topics
                results = invalid_points(results)

            if(user_input == 'a'):              #a
                if(pt_msg is None):             #如果没有数据，读取一个Point及相关的Imgs
                    pt_msg, img_msgs, index = read_bag_msgs(bag_msg_generator)      
                    if(pt_msg is None): break
                results = process_data(pt_msg, img_msgs)                            #处理数据，生成待发送topics
                results = invalid_points(results)
                results = invalid_image(results)

            if(user_input == 'r'):            #l
                if(pt_msg is None):
                    pt_msg, img_msgs, index = read_bag_msgs(bag_msg_generator)      #读取一个Point及相关的Imgs
                    if(pt_msg is None): break
                results = process_data(pt_msg, img_msgs)                            #处理数据，生成待发送topics


            publish_topics(pub_topics, results)                     #按键后发布topic

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass    

    print("\nBag file finished!")


if __name__ == "__main__":
    main()
