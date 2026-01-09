"""
程序设计概要：

----------------------------------------------
1. BEVFusion 推理需要的数据和类型：            |
----------------------------------------------
传入forward()函数的数据，在推理/test时下面的数据是必须的：
    img         1x6x3x256x704           数组
    cam2ego     1x6x4x4                 数组
    lidar2ego   1x4x4                   数组
    lidar2cam   1x6x4x4                 数组
    lidar2img   1x6x4x4                 数组
    intrinsics  1x6x4x4                 数组
    cam2lidar   1x6x4x4                 数组
    img_aug     1x6x4x4                 数组
    lidar_aug   1x4x4                   数组
    metas       [dict]                  dict的列表
    points      [(Nx5)]                 数组的列表

观察到的很重要的几点：
 - 虽然数据的物理类型多种多样（图像、点云、内参...),但在model.forward()时，其实绝大多数都已经转换为同一个形态,即数组
 - 点云输入给模型时，是list，但里面只有一个元素，这个原始也是数组的形式
 - metas输入给模型时，是dict字典，比较独特的。 
 - bevfusion.forward()还定义了3个其它输入，各种真值表，但是默认值是None。它们在Training的时候需要输入真实的数据用来
   计算loss，做反向传播。 但是test/推理的时候，是不需要输入的。所以不用考虑。  
    

----------------------------------------------
2. ROS 广播发布的数据内容                      |
----------------------------------------------
2.1 使用Nuscenes数据
    使用Nuscenes数据时，在选择上比较容易，把forward()函数需要的各种参数，上面所列的，在每一帧都发布出去就可以。 
    ROS发布进程使用一个timer, 每个一段时间比如200ms, 从PKL文件读取一个sample的数据，使用不同的topic & msg发布出去。
    
    这个定时器的间隔需要依赖于接收端，推理（和可视化）的速度。不能过快，使得推理或可视化来不及完成。

2.2 使用Custom数据
    当使用自己的数据，特别是自己的小车在运行中，实时地采集和发布各种数据时，情况比使用Nuscenes数据集变得复杂很多。程序
    设计和实现也会不一样。 

 第一条：SAMPLE概念
    实时的数据没有了Nuscenes数据集中‘sample’的概念了。 在Nuscenes中，一个sample包括了： 1个lidar点云关键帧+9个
    前续的点云扫描帧 + 6幅相近时间内的图像 + 相关的转换矩阵和相机内参等信息。 这样的一个sample是已经“预先”分类定义好的，
    ROS发布程序只需要按照sample，每隔比如200ms把这些相关的数据从文件中读取出来，发布出去就可以。发布时，可以使用不同的
    topic和不同的msg，但是可以在msg header里面，给属于同一个sample的不同消息赋予一个相同的frame-ID，表示这些不同的msg或
    topic上的数据，是属于同一个sample的。  

    而实际小车运行中，Lidar帧和图像帧的获取时间是不同步的，它们抵达ROS发布进程的时间应该也不是同时的。比如Lidar每秒完成
    20帧扫描，摄像头每秒完成15次拍摄。每个摄像头触发快门的时间又不同，这些数据在各自不同的间隔抵达， 因此哪些数据算是‘同一
    个sample', 推理程序（侦听程序）收到哪些消息后才启动一次推理？ 
    
 第二条：固定的内外参
    下面这些参数，通常是固定的，是不需要周期发送的：
    - 6x相机内参
    - 6x相机外参（相机在车体上的位姿 cam-2-ego) 
    - 图像增强矩阵 即图像的缩放和裁剪
    - 1xLidar外参（Lidar在车体上的位姿 Lidar-2-ego) 

 第三条：需要自己实时推算的参数
    forward()函数用到的下面的输入参数，不可能像nuscenes数据集那样预先‘离线’计算好：
    - lidar2cam   1x6x4x4                 数组
    - lidar2img   1x6x4x4                 数组
    - cam2lidar   1x6x4x4                 数组
    
    它们需要在每次推理前，基于最新的转换矩阵信息计算出来。 

 第四条：多帧点云sweep的聚合    
    BevFusion可以接收10帧点云的聚合，来增加点云密度，以取得较好的推理结果。 
 
  点云信息。 在实际场景中，每次完成一个点云帧的扫描（比如50ms)就可以发布出去。 在nuscenese数据集，则是使用了1+9帧的
  聚合点云，用来提高精度。使用nuscenes数据集，这10个sweep在读取PKL文件时进行聚合（并且全部对齐到最新的关键帧的坐标系。

  发布nuscenese数据，我们直接把11帧点云的数据聚合在一起，当作一个点云数据发布出去。11帧点云的数据在聚合时，都已经同步
  到了关键帧时刻的位姿。推理侧，收到这个点云topic，直接用于推理。

  发布自己车的实际数据： 尤其是车在运行的时候，实时采集的数据， 我们只能每次完成一帧的扫描，就把这一帧的数据发布出去。
  一般，不应该在采集发布侧进行10帧数据的聚合，然后再发布。因为这意味着每帧数据每重复发送了9次，增加了数据通信的占用。可
  以在推理侧进行聚合： 收到一帧点云后，将它缓存，永远缓存最新的10帧数据，然后聚合，用于推理。 当然也可以不做10帧聚合，
  只使用最后的一帧，可能效果有点差。 

2.3 基于上述讨论
a) 使用Nuscenes数据集时：
    - 发布上述所有的数据
    - 在msg header中，给属于同一个sample的不同数据，赋相同的frame-ID. 消息侦听方依据frame-ID来确定哪些数据属于同一帧

b) 下一步，使用自己小车数据时：
   发布方：
        - 每一个图像获取时（触发快门时），都要获得对应的车体在世界中的位姿，即camera的ego_2_global 矩阵
        - 每一个Lidar帧获取时（扫描完成时？），都要获得对应的车体在世界中的位姿，即lidar的ego_2_global 矩阵
        - Camera-2-ego矩阵，lidar-2-ego矩阵，Camera内参， 图像增强矩阵（缩放裁剪）， 通常是固定的：
            1) 可以缓慢周期性（比如每100s)发布一次;
            2) 甚至不发布，直接硬编码写在接收方代码中； 
            3) 也可以每次随同image&lidar来相同的频率发布---这就类似nuscenes数据集的发布方式。
   
        因此必须实时发布的内容：
        - 6个image+6个对应的ego-2-global矩阵+6个图像拍照时戳
        - 1个Lidar点云+1个对应的ego-2-global矩阵+1个Lidar扫描时戳

        图像在发布前，先裁剪缩放到模型需要的尺寸256x704.

    接收方：
    - Camera 数据
        接收方总是保存最新收到的各CAM数据。 比如CAM_FRONT: 收到front-cam的图像+ego2global+时戳后，保存它。其它camera
        数据也是类似处理。
    
    - Lidar 数据
        接收方收到一帧最新的Lidar数据后，即启动一次推理。 当前收到的Lidar数据加上保存的6幅图像数据作为一个'Sample'， 
        当且仅当收到Lidar数据后，我们才启动一次推理。 当收到Camera数据时，我们只缓存，不推理。 
    
    - 点云聚合
        另外，我们总是缓存最近的9帧Lidar数据。 用来做聚合，增加点云密度。 那么当程序刚刚启动，没有缓存足够的点云帧时，就
        用相对较少的帧的聚合。 每收到一个最新的Lidar帧，前面保存的9帧会对齐到最新的lidar帧上。然后新的Lidar帧也缓存，并
        挤掉前面最老的Lidar帧
    
    - 参数推算
      每次推理前，需要重新计算下面的矩阵：
        lidar2cam   
        lidar2img   
        cam2lidar   

    - 这样，接收方就有了足够的数据来推理了


----------------------------------------------
3. ROS 广播发布 Topic &Msg选择                |
----------------------------------------------

3.1 出发点：增加重用性
    设计程序时，尽量使得处理Nuscenes数据和将来处理自己小车的数据时，代码修改少，重用性高。 

3.2 Topic 
    下面定义了使用的Topic,对应的MSG类型和用途。

    =========================================================================================================================================
    |    TOPIC名称              |      MSG 类型             |   用途 /数据类型            |         说明& 注释                                 |  
    |========================================================================================================================================
    | '/camera/img'             | sensor_msgs/Image         | Img    1x6x3x256x704      | 发送时 numpy转Image; 接收时Image转numpy             |
    |----------------------------------------------------------------------------------------------------------------------------------------
    | '/pose/cam2ego'           | sensor_msgs/Image         | cam2ego     1x6x4x4       |                                                   |
    |------------------------------------------------------------------------------------  将这些姿态矩阵转为‘Image'消息类型发送而没有选择ROS   |
    | '/pose/lidar2ego'         | sensor_msgs/Image         | lidar2ego   1x4x4         |  预定义的各种姿态消息，这是一种‘偷懒’更是一种编程技    |
    |------------------------------------------------------------------------------------  巧： 1） 因为ROS各姿态消息使用四元数而不是3x3或      |
    | '/pose/lidar2cam'         | sensor_msgs/Image         | lidar2cam   1x6x4x4       |  4x4矩阵。我们从Nuscenes读取的姿态矩阵，已经被从原    |
    |------------------------------------------------------------------------------------  始的四元数处理成4x4矩阵，我们没有必要为了ROS发布而    |
    | '/pose/lidar2img'         | sensor_msgs/Image         | lidar2img   1x6x4x4       |  再折腾一次，转回四元数。2）PKL读取的姿态数据，对相    |
    |------------------------------------------------------------------------------------  机而言，已经组合成6x4x4的形式，而ROS预定义姿态消息    |
    | '/camera/intrinsics'      | sensor_msgs/Image         | intrinsics  1x6x4x4       |  每次只能传一个姿态，势必需要很多个消息，编程太麻烦。   |
    |------------------------------------------------------------------------------------  3）所谓的Image消息当然不仅仅用来传输Image，而且Image  |
    | '/pose/cam2lidar'         | sensor_msgs/Image         | cam2lidar   1x6x4x4       |  本周上也是个矩阵。只要在接收方，我们把它解读为一个     |
    |------------------------------------------------------------------------------------  6x4x4的矩阵，而非可以显示的图片，就可以了。           |
    | '/camera/img_aug'         | sensor_msgs/Image         | img_aug     1x6x4x4       |                                                     |
    |=========================================================================================================================================
    | '/lidar/top/points'       | sensor_msgs/PointCloud2   | points      (Nx5)         |  处理CenterHead时已经首发过PointCloud2类型的消息      |
    |-----------------------------------------------------------------------------------------------------------------------------------------|
    | '/metas/'                 | std_msgs/String           | metas       dict          |  dict可以转为字符串，接收方再从字符串转回Dict,放入list |
    |=========================================================================================================================================

几点说明
    1） 图像数据用sensor_msgs/Image非常自然，但是我们把转换矩阵也用sensor_msgs/Image类型的ROS消息来发布了。 
        这是为了简化程序，减少topic数量的一个编程选择。 

    2） 这些数据，在发送前都是torch tensor. 我们需要一个 tensor -> numpy -> ROS Msg的转变。 使用ros_numpy
        库来从numpy转到各种ROS消息
    
    3） 点云BIn文件转为PointClound2消息收发，在处理CenterHead时已经有过实现，可以参考

    4） ros_numpy在将numpy矩阵转为Image消息时，有限制。 第一，我们的numpy里面是float32的，已经归一化后的矩阵。
       第二，矩阵的维度往往不符合Image的维度要求(通常，比如，是1/2/3/4个通道)， 是WxHx1, WxHx2, WxHx3, WxHx4
       的维度。 如果维度不符合， ros_numpy回报错。 像1x6x3x256x704这种维度，一定回出错，无法转为Image. 
       
       ros_numpy所有支持的类型,参见：https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py

       其中，跟float32相关的4种支持的格式：
       	"32FC1":   (np.float32, 1),
        "32FC2":   (np.float32, 2),
        "32FC3":   (np.float32, 3),
        "32FC4":   (np.float32, 4),
            
       那么，上面列的各数组，在转Image时，需要进行维度压缩。 接收方需要进行维度reshape恢复到压缩前的形状。        
    
        Img 1x6x3x256x704 -->(压缩)-->1536x704x3 -->图像格式 "32FC3"
        cam2ego 1x6x4x4   -->(压缩)-->6x4x4      -->图像格式 "32FC4"
        ...
        其它依次类推
    
    5） 点云在接收方收到后，需要把它放入一个list里面，这是forward()期望的格式
    6） meta是dict, 发送时使用json.dumps(dict)转成字符串，使用标准的String消息发送。 接收方使用json.loads(string)把
        它转回dict。 在传入forward()函数前，需要把它放入list中
    7） 其它数据，在收到Image类型消息后，转回Numpy Array -->转为Tensor --> 加载到CUDA, 然后就可以传给forward()


----------------------------------------------
3. 整体程序结构                               |
----------------------------------------------
3.1 NUSCENES数据集

    发布nuscenes数据集，总共3个程序（3个ROS节点）。 
     1) A = 负责读取PKL文件发布数据，
     2) B = 负责侦听，并调用推理。然后将推理得到的BBOX结果发布出去
     3) C = 负责侦听和可视化推理结果。 


    ---------------------                   ---------------------                       --------------------- 
    |   A               |    消息           |   B               |    检测结果消息        |   C               |
    |  读取PKL,定时发布  |-----------------> | 侦听来自A的topic， |---------------------->| 接收来自A的图像+点  |
    |  各topic          |           |       | 推理，将推理结果发  |                       | 云消息，以及B的检测 |
    |                   |           |       | 布出去             |            ---------> | 结果，可视化       |
    ---------------------           |       ---------------------            |           ---------------------
                                    |________________________________________|



3.2 自己小车数据

    切到自己小车数据时，仍然总共3个程序（3个ROS节点）。 
    1) A = 负责读取PKL文件发布数据。 
            
            ***需要较大的改变****
            前面讨论的，nuscenes和自己数据集的各种区别造成的不同处理，比如sweep的聚合等，我们统统放在A里面实现，目的
            是保持B和C模块，无论使用哪种数据集，都不再修改！

    2) B = 负责侦听，并调用推理。然后将推理得到的BBOX结果发布出去
            ***保持不变****

    3) C = 负责侦听和可视化推理结果。 
            ***保持不变****
    

https://index.ros.org/p/geometry_msgs/
https://index.ros.org/p/sensors_msgs/
https://index.ros.org/p/std_msgs/
https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py
https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py

"""


import  rospy 
import  ros_numpy 
import  numpy as np 
from    sensor_msgs.msg import PointCloud2,Image
from    std_msgs.msg import Header
from    std_msgs.msg import String
import  os 
from    my_load_pkl import NuscenesLoadData
import  readchar


from    torchpack.utils.config import configs
from    mmcv import Config
from    mmdet3d.utils import recursive_eval


def load_cfg_and_pkl(config_dict):
    #读取config文件
    cfg_path = config_dict['data_loader']['cfg_path']
    pkl_path = config_dict['data_loader']['pkl_path']

    configs.load(cfg_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=cfg_path)

    #初始化NuscenesLoadData类
    LoadData = NuscenesLoadData(**cfg.data.test)

    #加载PKL文件
    LoadData.load_pkl(pkl_path=pkl_path)

    return LoadData

######################################################################################################
# 初始化需要发布的topics，初始化节点。 按照设计定义，本节点为BEV_A节点。                                   
# 把rospy.Publisher()返回的topic变量，组织到一个字典中，然后返回这个字典。后面当需要发布topic时候，直接    
# 通过这个字典来引用对应的topic变量，这样比较方便。                                                       
######################################################################################################   
def init_ros_and_topics():    
    topics = {}
    topics['rawimg']            = rospy.Publisher('/camera/rawimg',     Image,      queue_size=3)
    topics['img']               = rospy.Publisher('/camera/img',        Image,      queue_size=3)
    topics['camera_intrinsics'] = rospy.Publisher('/camera/intrinsics', Image,      queue_size=3)
    topics['img_aug_matrix']    = rospy.Publisher('/camera/img_aug',    Image,      queue_size=3)
    topics['camera2ego']        = rospy.Publisher('/pose/cam2ego',      Image,      queue_size=3)
    topics['lidar2ego']         = rospy.Publisher('/pose/lidar2ego',    Image,      queue_size=3)
    topics['lidar2camera']      = rospy.Publisher('/pose/lidar2cam',    Image,      queue_size=3)
    topics['lidar2image']       = rospy.Publisher('/pose/lidar2img',    Image,      queue_size=3)
    topics['camera2lidar']      = rospy.Publisher('/pose/cam2lidar',    Image,      queue_size=3)
    topics['lidar_aug_matrix']  = rospy.Publisher('/lidar/lidar_aug',   Image,      queue_size=3)
    topics['points']            = rospy.Publisher('/lidar/points',      PointCloud2,queue_size=3)
    topics['metas']             = rospy.Publisher('/misc/metas',        String,     queue_size=3)
    topics['gt_bboxes']         = rospy.Publisher('/bboxes/gt',         Image,      queue_size=3)   #20250429 newly added to send gt bboxes 
    
    #为了调试，注册了一个侦听。把自己发送的topic自己来接收，然后转换回原来的numpy格式，跟发送前数据比对，是不是完全一致
    """
    topics['listen_img']        = rospy.Subscriber('/pose/lidar2ego', Image, img_callback, queue_size=1)
    topics['listen_meta']       = rospy.Subscriber('/misc/metas', String, meta_callback, queue_size=1)
    topics['listen_pts']        = rospy.Subscriber('/lidar/points', PointCloud2, pts_callback, queue_size=1)
    """
    rospy.init_node('MY_BEV_A', anonymous=True, disable_signals=True)
    
    return topics

#为了调试，注册了一个侦听。把自己发送的topic自己来接收，然后转换回原来的numpy格式，跟发送前数据比对，是不是完全一致
imgs =[]
def img_callback(msg):
    idx = int(msg.header.frame_id)
    img = ros_numpy.image.image_to_numpy(msg)
    img =  np.expand_dims(img, axis=0)

    equal = np.array_equal(imgs[idx], img)
    #print(f'b equal ={equal} on idx ={idx}')
    pass

def meta_callback(msg):
    #print(f'received metas: ={msg.data}')
    pass

pts = []
def pts_callback(msg):
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    points = np.column_stack((cloud_array['x'],
                             cloud_array['y'],
                             cloud_array['z'],
                             cloud_array['intensity'],
                             cloud_array['time']))
    
    idx = int(msg.header.frame_id)

    equal = np.array_equal(pts[idx], points)
    print(f'b points equal ={equal} on idx ={idx} shape={points.shape}')
    

######################################################################################################
# 输入： 
#     topics: 是初始化的各topic
#     data:   一个sample
# 功能：
#     针对data字典中的每个数据类型，把它转为对应的MSG，用对应的topic发送出去
######################################################################################################
import ros_numpy 
import json
def publish_topics(topics, data, index):
    
    #------------------------------------------------------
    # 1. 发送Img 1x6x3x256x704
    #-----------------------------------------------------
    img_data = data['img'].data[0].numpy()                          #取出image数据，转为numpy 

    B,N,C,H,W = img_data.shape
    img_data = img_data.reshape((B*N*H, W, C))                      #reshape. 1x6x3x256x704 --> 1536x704x3

    msg = ros_numpy.image.numpy_to_image(img_data, "32FC3")         #numpy to Image message
    msg.header.frame_id = str(index)                                #这个貌似没有必要
    topics['img'].publish(msg)                                      #发布


    #------------------------------------------------------
    # 2. 发送Camera Intrinsics 1x6x4x4
    #------------------------------------------------------
    img_data = data['camera_intrinsics'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)                             #去掉一个维度
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera_intrinsics'].publish(msg)                        #发布

    #------------------------------------------------------
    # 3. 发送img_aug_matrix 1x6x4x4
    #------------------------------------------------------
    img_data = data['img_aug_matrix'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['img_aug_matrix'].publish(msg)                            #发布

    #------------------------------------------------------
    # 4. 发送 camera2ego 1x6x4x4
    #------------------------------------------------------
    img_data = data['camera2ego'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera2ego'].publish(msg)                               #发布

    #------------------------------------------------------
    # 5. 发送 lidar2ego 1x4x4
    #------------------------------------------------------
    img_data = data['lidar2ego'].data[0].numpy()
    imgs.append(img_data)     
    img_data = img_data.squeeze(axis=0)                             #ros_numpy支持4x4或4x4x1的格式，不支持1x4x4格式（即图像必须是HxW, HxWxC格式。C通道在最后）
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC1")         #numpy to Image message
    msg.header.frame_id = str(index)
    topics['lidar2ego'].publish(msg)                                #发布

    #------------------------------------------------------
    # 6. 发送 lidar2camera 1x6x4x4
    #------------------------------------------------------
    img_data = data['lidar2camera'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)                             
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['lidar2camera'].publish(msg)                             #发布


    #------------------------------------------------------
    # 7. 发送 lidar2image 1x6x4x4
    #------------------------------------------------------
    img_data = data['lidar2image'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)                             
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['lidar2image'].publish(msg)                              #发布

    #------------------------------------------------------
    # 8. 发送 camera2lidar 1x6x4x4
    #------------------------------------------------------
    img_data = data['camera2lidar'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)                             
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC4")         #numpy to Image message
    topics['camera2lidar'].publish(msg)                             #发布

    #------------------------------------------------------
    # 9. 发送 lidar_aug_matrix 1x4x4
    #------------------------------------------------------
    img_data = data['lidar_aug_matrix'].data[0].numpy()
    img_data = img_data.squeeze(axis=0)                             
    msg = ros_numpy.image.numpy_to_image(img_data, "32FC1")         #numpy to Image message
    topics['lidar_aug_matrix'].publish(msg)                         #发布

    #------------------------------------------------------
    # 9. 发送 metas  dict 
    #------------------------------------------------------
    metas = data['metas'].data[0][0]                                #取出metas字典 - 很奇葩，metas外边包了两层list
    
    # Metas有额外的问题需要解决。 metas字典里面有几个元素，不是一般的数据类型，没办法把它们转成字符串发送出去。 
    # 这两项目前看起来在物体检测的head中使用到，第三个貌似没有代码使用。我们得看看如何绕过去。 可行的思路是：发
    # 送时，把它们从发送方去掉，根本就不要发送这些项。 接收方在收到后，自己把这些项目填充起来...
    #    'box_mode_3d': <Box3DMode.LIDAR: 0>, 
    #    'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
    #    ‘lidar2image': [numpy array]
    metas['box_mode_3d'] = None                                     #这三项清零
    metas['box_type_3d'] = None
    metas['lidar2image'] = None

    metas = json.dumps(metas)                                       #转成字符串
    topics['metas'].publish(metas)                                  #发布字符串
    

    #------------------------------------------------------
    # 10. 发送 raw image 6x 900 x1600 x3
    #------------------------------------------------------    
    img_data = data['rawimg'].data[0].numpy()                       #取出image数据，转为numpy 

    N,H,W,C = img_data.shape
    img_data = img_data.reshape((N*H, W, C))                        #reshape. 6x900x1600x3 -->5400x1600x3

    msg = ros_numpy.image.numpy_to_image(img_data, "rgb8")          #numpy to Image message
    msg.header.frame_id = str(index)                                #这个貌似没有必要
    topics['rawimg'].publish(msg)                                   #发布
    print(f'raw image send!')

    #------------------------------------------------------
    # 10. 发送 points [N,5]
    #------------------------------------------------------
    pt_data = data['points'].data[0][0].numpy()

    print(f'points to be published - max intensity={np.max(pt_data[:,3])}')
    pts.append(pt_data)
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
    msg.header.frame_id = str(index)
    topics['points'].publish(msg)                                   #发布
    
    #------------------------------------------------------
    # 10. 发送 ground-truth bboxes 
    #------------------------------------------------------
    gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()                  #[Nx9]=x,y,z,dx,dy,dz,theta, vx, vy
    gt_labels = data["gt_labels_3d"].data[0][0].numpy().astype(np.float32)      #[Nx1]
    gt_bboxes = np.hstack((gt_bboxes, gt_labels.reshape(-1,1)))
    
    H, W = gt_bboxes.shape
    msg = ros_numpy.image.numpy_to_image(gt_bboxes, "32FC1")        #numpy to Image message
    topics['gt_bboxes'].publish(msg)               #发布
    print('published gt_boxes!')
    pass

import torch

def invalid_image(data):
    img_data = data['img'].data[0]
    B, N, C, H, W = img_data.shape
    data['img'].data[0]=torch.zeros(B,N,C,H,W,dtype=torch.float32)
    
    return data

def invalid_points(data):
    pt_data = data['points'].data[0][0]
    N, C = pt_data.shape
    data['points'].data[0][0] = torch.zeros(N,C,dtype=torch.float32)

    return data


import yaml 
import argparse

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=False, default="tools/infer.cfg.yaml", type=str, help="provide config file for this module")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as file:
            config_dict = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{args.config}' not found.")
        exit(0)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")    
        exit(0)

    #加载PKL文件，实例化一个NuscenesLoadData的变量，返回到nusc_data中。后面我们用nusc_data变量来访问每个sample
    nusc_data = load_cfg_and_pkl(config_dict)

    #初始化ROS节点，初始化所有需要发布的topics
    topics    = init_ros_and_topics()

    #设置每秒发送的节奏
    rate = rospy.Rate(1)   

    #进入循环，依次从index=0开始，读取每个sample，然后发布出去，然后睡眠相应的时间，再读取下一个。
    #如果nusc_data.get_data()返回None,表示读取到了PKL文件的最后一个记录。 这时候可以选择停止发送，或者index重置，
    #再次从0开始读取。
    
    index = -1
    try:
        while(True):
            #等待按键输入    
            print(f"\nnew data ready-{index}, press SPACE to publish, ENTER to exit:")            
            user_input = readchar.readkey()                         #等待用户按键

            #回车退出
            if(user_input == '\r' or user_input =='\n'):            #RETURN key to exit
                print("\nUser exit!")
                break

            if(user_input == 'i' or user_input =='I'):              #I键将图像清理
                if(index == -1):index = 0
                data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中
                data = invalid_image(data)

            if(user_input == 'l' or user_input =='L'):              #L键将点云清理
                if(index == -1):index = 0
                data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中
                data = invalid_points(data)

            if(user_input == 'a' or user_input =='A'):              #L键将点云清理
                if(index == -1):index = 0
                data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中
                data = invalid_image(data)
                data = invalid_points(data)
                        
            if(user_input == 'r' or user_input =='R'):              #L键将点云清理
                if(index == -1):index = 0
                data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中

            if(user_input ==' '):
                if(index == -1):
                    index = 0
                else:
                    index +=1
                data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中
                if (data is None):                                  #如果返回None，则index重置从0开始
                    index = 0
                    data = nusc_data.get_data(index)                    #读取一个sample数据，返回结果在字典中

            publish_topics(topics, data, index)                 #发布topic
            rate.sleep()                                        #sleep
            #index +=5                                           #读取下一个sample
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass    #talker(output)


if __name__ == "__main__":
    main()



"""
metas =[
    {
        'filename': 
            ['./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg', 
            './data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg', 
            './data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg', 
            './data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg', 
            './data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg', 
            './data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg'
            ], 
        'timestamp': 1533151603547590, 
        'ori_shape': (1600, 900), 
        'img_shape': (1600, 900), 
        'pad_shape': (1600, 900), 
        'scale_factor': 1.0, 
        'box_mode_3d': <Box3DMode.LIDAR: 0>, 
        'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'img_norm_cfg': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}, 
        'token': '3e8750f331d7499e9b5123e9eb70f2e2', 
        'lidar_path': './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
        }
        ]

"""