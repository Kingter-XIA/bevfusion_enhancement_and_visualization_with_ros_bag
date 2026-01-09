"""
程序设计说明：
    最终实现为(至少)3个ROS节点A, B, C，分别跑在3个ROS终端节点上。这里是B的实现。
    - A负责读取PKL文件，把数据发布出去。 
    - B负责侦听数据，进行推理，并把推理检测结果发布出去
    - C负责侦听数据和检测结果，进行可视化。

三部分关系如下所示。 目标是, B和C最终即可以支持来自nuscenes文件的数据发,也可以接收来自自己小车的实时的数据采集, 而不用做修改(或
很少的修改),而A则需要区分是处理nuscenese数据还是自己的数据,可能有完全不同的代码实现。 但是自己小车和nuscenes数据集带来的不同,
争取完全在A里面“抹平”,从而A发给B和C的消息类型、方式等都是一样的。 


    -----------------------------               -------------------------               -----------------------------
    |           A               |   img+pts     |       B               | det results   |       C                   |
    |  my_publish_nuscenes.py   |-------------> | my_listen_infer.py    | ------------->| my_virsualize.py          |
    |                           |               |                       |       |------>|                           |
    -----------------------------               -------------------------       |       -----------------------------
                    |                img+pts                                    |
                    -------------------------------------------------------------

本程序B的功能
- 加载BEVFusion模型
- 启动ROS topic侦听
- 收到每一个topic发来的数据,就将它保存起来(覆盖前一个保存的数据)                    
- 当收到lidar点云后就执行一次推理，推理使用收到的点云和其它缓存的topic数据。 但是在第一次收到Lidar点云的时候,如果有的topic还没有
  收到，则继续等待。
- 缓存的数据，在收到topic后都是转为numpy保存。推理时，数据转为torch tensor, 并需要包装上mmcv的Data Container格式，以便于跟模型
  兼容。调用模型前，tensor还需要移到CUDA里面 
- 推理结果（物体检测结果）通过消息发布出去。

"""



import  torch
import  numpy as np
import  rospy
import  json
import  ros_numpy 

from    sensor_msgs.msg import PointCloud2,Image
from    std_msgs.msg import Header
from    std_msgs.msg import String


from    mmdet.apis import set_random_seed
from    mmcv.parallel import DataContainer as DC
from    torchpack.utils.config import configs
from    mmcv import Config
from    mmcv.parallel import MMDataParallel
from    mmcv.runner import load_checkpoint, wrap_fp16_model

from    mmdet3d.models import build_model
from    mmdet3d.utils import recursive_eval
from    mmdet3d.core.bbox import get_box_type

"""
cfg_path    = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
checkpoint  = './pretrained/bevfusion-det.pth'

cfg_path2    = './configs/nuscenes/seg/fusion-bev256d2-lss.yaml'
checkpoint2  = './pretrained/bevfusion-seg.pth'
"""

######################################################################################################
# 初始化需要侦听的topics，初始化节点。 按照设计定义，本节点为BEV_B节点。                                   
# 把rospy.Publisher()返回的topic变量，组织到一个字典中，然后返回这个字典。后面当需要发布topic时候，直接    
# 通过这个字典来引用对应的topic变量，这样比较方便。                                                       
######################################################################################################   

"""
设计了一个类，来处理所有topic侦听相关的事情，包括：
- 订阅各topic
- topic的回调函数等
"""

class MyListener:
    results= {}
    model = None 
    dataset = None 
    cfg = None
    cfg2 =None
    model2 = None

    @staticmethod
    def init_results():        
        MyListener.results['img']               = None
        MyListener.results['camera_intrinsics'] = None
        MyListener.results['img_aug_matrix']    = None
        MyListener.results['camera2ego']        = None
        MyListener.results['lidar2ego']         = None
        MyListener.results['lidar2camera']      = None
        MyListener.results['lidar2image']       = None
        MyListener.results['camera2lidar']      = None
        MyListener.results['lidar_aug_matrix']  = None
        MyListener.results['points']            = None
        MyListener.results['metas']             = None

    @staticmethod
    def load_model(cfg_dict, model_index = -1):

        model_cfg = cfg_dict['inference']
        model_num = model_cfg['model_num']         #1 or 2

        MyListener.model_cfg = model_cfg

        MyListener.models = [None]*model_num    #按照model数量先创建2个空元素的list 
        
        for i in range(model_num):
            if(model_index !=-1):
                if(model_index != i):
                    continue

            print(f"-----{model_cfg['models'][i]}")
            cfg_path      = model_cfg['models'][i]['cfg_path']      #取model对应的cfg文件
            checkpoint    = model_cfg['models'][i]['checkpoint']    #取model对应的check point PTH文件

            #读取config文件
            configs.load(cfg_path, recursive=True)
            cfg = Config(recursive_eval(configs), filename=cfg_path)

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

            MyListener.models[i] = model
            print(f'model{i} loaded.')        
    @staticmethod
    def prepare_data():
        # 检查是不是所有的数据都收到了
        for _, value in MyListener.results.items():
            if value is None:
                return None 
        
        # to tensor and to CUDA 
        device = 'cuda'
        data ={}
        data['img'] = torch.from_numpy(MyListener.results['img']).to(device)
        data['camera_intrinsics'] = torch.from_numpy(MyListener.results['camera_intrinsics']).to(device)
        data['img_aug_matrix'] = torch.from_numpy(MyListener.results['img_aug_matrix']).to(device)
        data['camera2ego'] = torch.from_numpy(MyListener.results['camera2ego']).to(device)
        data['lidar2ego'] = torch.from_numpy(MyListener.results['lidar2ego']).to(device)
        data['lidar2camera'] = torch.from_numpy(MyListener.results['lidar2camera']).to(device)
        data['lidar2image'] = torch.from_numpy(MyListener.results['lidar2image']).to(device)
        data['camera2lidar'] = torch.from_numpy(MyListener.results['camera2lidar']).to(device)
        data['lidar_aug_matrix'] = torch.from_numpy(MyListener.results['lidar_aug_matrix']).to(device)
        data['points'] = torch.from_numpy(MyListener.results['points']).to(device)
        
        #DC wrapper
        data['img'] = DC([data['img']])
        data['points'] = DC([[data['points']]])        
        data['camera_intrinsics'] = DC([data['camera_intrinsics']])
        data['img_aug_matrix'] = DC([data['img_aug_matrix']])
        data['camera2ego'] = DC([data['camera2ego']])
        data['lidar2ego'] = DC([data['lidar2ego']])
        data['lidar2camera'] = DC([data['lidar2camera']])
        data['lidar2image'] =DC([data['lidar2image']])
        data['camera2lidar'] = DC([data['camera2lidar']])
        data['lidar_aug_matrix'] =DC([data['lidar_aug_matrix']])

        data['metas'] = MyListener.results['metas']

        type, mode = get_box_type('LIDAR')
        data['metas']['box_mode_3d'] = mode
        data['metas']['box_type_3d'] = type
        print(f'----mode = {mode} type = {type}')
        data['metas'] = DC([[data['metas']]], cpu_only=True)
        
        return data

    """
    将推理结果：BBOXES发布出去。 
    model()返回的推理结果output是一个list. output[0]是个dict:
    {
        "boxes_3d:      value是个mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes类型的tensor封装
                        其中的value.tensor 是[N, 9]的float32 tensor
        "scores_3d":    value 是个[N,] 的float32 tensor
        "labels_3d":    value 是个[N,] 的float32 tensor
    }
    我们将结果从dict中取出，拼成Nx11的float32数值，然后转为Image类型的消息，使用topic='/detection/bboxes'发布出去。
    """

    @staticmethod
    def publish_bboxes(outputs, bboxes1=True):
        if "boxes_3d" in outputs[0]:
            print(f'putputs={outputs[0].keys()}')
            bboxes = outputs[0]["boxes_3d"].tensor.cpu().numpy().astype(np.float32)
            scores = outputs[0]["scores_3d"].cpu().numpy().astype(np.float32)
            labels = outputs[0]["labels_3d"].cpu().numpy().astype(np.float32)
            
            stacked_array = np.hstack((bboxes, scores.reshape(-1,1), labels.reshape(-1,1)))
            H, W = stacked_array.shape
            msg = ros_numpy.image.numpy_to_image(stacked_array, "32FC1")        #numpy to Image message
            if(bboxes1 == True):
                MyListener.publisher_bboxes1.publish(msg)                                   #发布
            else:
                MyListener.publisher_bboxes.publish(msg)                                    #发布
        return 

    @staticmethod
    def publish_map(outputs):
        if "masks_bev" in outputs[0]:            
            masks = outputs[0]["masks_bev"].numpy().astype(np.float32)
            N, H, W = masks.shape
            masks = masks.reshape(N*H, W)
            msg = ros_numpy.image.numpy_to_image(masks, "32FC1")                #numpy to Image message
            MyListener.publisher_map.publish(msg)                                       #发布
            print(f'---masks ={masks.shape} ')
            print(f'{masks[0,:]}')
            print(f'{masks[1,:]}')
            print(f'{masks[2,:]}')
            print(f'{masks[3,:]}')
            print(f'{masks[4,:]}')
            print(f'{masks[5,:]}')
            
        return 

    #-----------------------------------------------------------------------------------------------------
    # 以下是各topic侦听处理的函数。把数据从topic提取出来，转为tensor, 放入CUDA. 结果用一个字典来追踪和更新。 
    #
    # 需要特别注意的地方，数据同步:
    # 这些回调函数被ROS异步调用，运行在thread-A, 本程序的主函数main()运行在thread-B. 它们不是同一个线程。 
    # Thread-A 回调函数里面，我们在提取数据写入一个本地变量（即buffer)时, Thread-B可以正在调度推理，也读取同一个
    # 变量。这会导致thread-A获得错误的数据：比如回调函数将image写入本地buffer, 正写到一半，thread-B读取image，则
    # 它拿到的image包括了前一个sample的一半和后一个sample的一半。 
    # 我们需要避免这一冲突，机制是使用线程间同步锁，lock， unlock.
    #-----------------------------------------------------------------------------------------------------
    @staticmethod
    def handle_img(msg, topic_key):
        img = ros_numpy.image.image_to_numpy(msg)    

        if(topic_key == 'img'):
            img = img.reshape(1,6,3,256,704)
        else:
            img =  np.expand_dims(img, axis=0)

        MyListener.results[topic_key] = img 

    @staticmethod
    def img_callback(msg):
        MyListener.handle_img(msg, topic_key = 'img')

    @staticmethod
    def intrinsics_callback(msg):
        MyListener.handle_img(msg, topic_key = 'camera_intrinsics')

    @staticmethod
    def img_aug_callback(msg):
        MyListener.handle_img(msg, topic_key = 'img_aug_matrix')

    @staticmethod
    def camera2ego_callback(msg):
        MyListener.handle_img(msg, topic_key = 'camera2ego')

    @staticmethod
    def lidar2ego_callback(msg):
        MyListener.handle_img(msg, topic_key = 'lidar2ego')

    @staticmethod
    def lidar2camera_callback(msg):
        MyListener.handle_img(msg, topic_key = 'lidar2camera')

    @staticmethod
    def lidar2image_callback(msg):
        MyListener.handle_img(msg, topic_key = 'lidar2image')

    @staticmethod
    def camera2lidar_callback(msg):
        MyListener.handle_img(msg, topic_key = 'camera2lidar')

    @staticmethod
    def lidar_aug_callback(msg):
        MyListener.handle_img(msg, topic_key = 'lidar_aug_matrix')

    @staticmethod
    def points_callback(msg):
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points = np.column_stack((cloud_array['x'],
                                cloud_array['y'],
                                cloud_array['z'],
                                cloud_array['intensity'],
                                cloud_array['time']))        
        MyListener.results['points'] = points 


        #do inference 
        data = MyListener.prepare_data()
        if (data is not None):
            with torch.no_grad():
                for i in range (MyListener.model_cfg['model_num']):
                    if(MyListener.models[i] is None):
                        print(f'---------------------model{i} skipped')
                        continue

                    output = MyListener.models[i](return_loss=False, rescale=True, **data)              #获取depth信息
                    
                    if(MyListener.model_cfg['models'][i]['type'] == 'det'):                             #发布推理结果                        
                        if(i == 0) and (MyListener.model_cfg['model_num'] >1):
                            MyListener.publish_bboxes(output, bboxes1= True)
                            print(f'bboxes published 1st one from model{i}')
                        else:
                            MyListener.publish_bboxes(output, bboxes1= False)
                            print(f'bboxes published last one from model{i}')

                    
                    if(MyListener.model_cfg['models'][i]['type'] == 'seg'):                             #发布推理结果
                        MyListener.publish_map(output)
                        print(f'map published')
                    
    @staticmethod
    def metas_callback(msg):
        metas = json.loads(msg.data)
        MyListener.results['metas'] = metas


    @staticmethod
    def init_ros_topics():
        topics = {}
        topics['img']               = rospy.Subscriber('/camera/img',        Image,      MyListener.img_callback,          queue_size=3)
        topics['camera_intrinsics'] = rospy.Subscriber('/camera/intrinsics', Image,      MyListener.intrinsics_callback,   queue_size=3)
        topics['img_aug_matrix']    = rospy.Subscriber('/camera/img_aug',    Image,      MyListener.img_aug_callback,      queue_size=3)
        topics['camera2ego']        = rospy.Subscriber('/pose/cam2ego',      Image,      MyListener.camera2ego_callback,   queue_size=3)
        topics['lidar2ego']         = rospy.Subscriber('/pose/lidar2ego',    Image,      MyListener.lidar2ego_callback,    queue_size=3)
        topics['lidar2camera']      = rospy.Subscriber('/pose/lidar2cam',    Image,      MyListener.lidar2camera_callback, queue_size=3)
        topics['lidar2image']       = rospy.Subscriber('/pose/lidar2img',    Image,      MyListener.lidar2image_callback,  queue_size=3)
        topics['camera2lidar']      = rospy.Subscriber('/pose/cam2lidar',    Image,      MyListener.camera2lidar_callback, queue_size=3)
        topics['lidar_aug_matrix']  = rospy.Subscriber('/lidar/lidar_aug',   Image,      MyListener.lidar_aug_callback,    queue_size=3)
        topics['points']            = rospy.Subscriber('/lidar/points',      PointCloud2,MyListener.points_callback,       queue_size=3)
        topics['metas']             = rospy.Subscriber('/misc/metas',        String,     MyListener.metas_callback,        queue_size=3)

        MyListener.publisher_bboxes1= rospy.Publisher('/detection/bboxes1',  Image,      queue_size=3)
        MyListener.publisher_bboxes = rospy.Publisher('/detection/bboxes',   Image,      queue_size=3)
        
        MyListener.publisher_map    = rospy.Publisher('/detection/map',      Image,      queue_size=3)
        
        rospy.init_node('MY_BEV_B', anonymous=True, disable_signals=True)

        MyListener.init_results()        
        
        return topics

import yaml 
import argparse

def main():
    #分析命令行
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=False, default="tools/infer.cfg.yaml", type=str, help="provide config file for this module")
    parser.add_argument("--model_index", required=False, default=-1, type=int, help="which model to run")
    args = parser.parse_args()
    
    #读取命令行给的yaml文件配置
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

    MyListener.load_model(config_dict, model_index=args.model_index)
    MyListener.init_ros_topics()
    

    print(f'\n---model loaded, now waiting for incoming topics and spinning')
    
    rospy.spin()
    

if __name__ == "__main__":
    main()



"""
---in scatter type input =<class 'dict'> data=dict_keys(['return_loss', 'rescale', 'img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'metas'])
---in scatter type input =<class 'dict'> data=dict_keys(['return_loss', 'rescale', 'img',                                                           'camera_intrinsics', 'img_aug_matrix', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'lidar_aug_matrix', 'points', 'metas'])
"""