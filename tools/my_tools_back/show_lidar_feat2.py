import os
import json 
import numpy as np
import matplotlib.pyplot as plt
from os import path
from PIL import Image 

feats =[
    'lidar_depth_1_6_1_256_704.bin',            #0
    'lidar_depth_6_64_32_88.bin',               #1
    'lidar_depth_d_6_1_32_88.bin',              #2
    'img_depth_6_118_32_88.bin',                #3
    'img_feat_6_80_32_88.bin',                  #4
    'img_feat_1_6_256_32_88.bin',               #5
    'bev_img_1_80_180_180.bin',                 #6
    'bev_lidar_1_256_180_180.bin',              #7
    'lidar_gt_1_6_1_32_88_3',               #8
    'lidar_depth_d_6_4_32_88.bin',              #9
    'lidar_depth_6_1_32_88.bin',                #10    
]

g_feat_path ='./bev_feat/1538984250947236'
g_feat_name = feats[4]
g_json_name = 'img_pts_path.json'


#########################################################################
# Load 6 JPG images for display purpose. 
# return 6 imgs + 6 title text 
#########################################################################
def load_raw_imgs(feat_path=g_feat_path, json_name=g_json_name):
    json_path = os.path.join(feat_path, json_name)
    print(f'json path ={json_path} g_feat_path={g_feat_path} feat_path={feat_path} json_name={json_name}')
    with open(json_path, 'r') as file:
        data = json.load(file)

    resize = 0.48
    W=1600
    H=900
    resize_dims = (int(W * resize), int(H * resize))
    crop = (32, 176, 736, 432)
    new_size = (88, 32)
    
    imgs = []
    names= []
    for jpg_file in data['img_path']:
        img = Image.open(jpg_file)
        img = img.resize(resize_dims)
        img = img.crop(crop)    
        img = img.resize(new_size)
        img = np.array(img)                             #img.shape = (256, 704, 3)

        imgs.append(img)

        parts = jpg_file.split('/samples/')
        if len(parts) > 1:
            name = parts[1].split('/')[0]
        else:
            name = 'noname'
        names.append(name)

    return imgs, names

#########################################################################
# Load feature matrix for display purpose 
# return 3 feature images: mean, max, average 
#########################################################################
def load_feats(feat_path = g_feat_path, feat_name = g_feat_name):
    dtype = np.float32                  
    H= 32
    W= 88
    C= 1
    N= 6
    xyz=False

    if 'lidar_depth_1_6_1_256_704.bin' in feat_name:
        H,W,C,N=256,704,1,6
    elif 'lidar_depth_6_64_32_88.bin' in feat_name:
        H,W,C,N=32,88,64,6
    elif 'lidar_depth_d_6_1_32_88.bin' in feat_name:
        H,W,C,N=32,88,1,6
    elif 'img_depth_6_118_32_88.bin' in feat_name:
        H,W,C,N=32,88,118,6
    elif 'img_feat_6_80_32_88.bin' in feat_name:
        H,W,C,N=32,88,80,6
    elif 'img_feat_1_6_256_32_88.bin' in feat_name:
        H,W,C,N=32,88,256,6
    elif 'bev_img_1_80_180_180.bin' in feat_name:
        H,W,C,N=180,180,80,1
    elif 'bev_lidar_1_256_180_180.bin' in feat_name:
        H,W,C,N=180,180,256,1
    elif 'lidar_gt_1_6_1_32_88_3.bin' in feat_name:
        H,W,C,N=32,88,1,6
    elif 'lidar_gt_xyz_1_6_1_32_88_3.bin' in feat_name:
        H,W,C,N=32,88,1,6
        xyz=True
    elif 'lidar_depth_d_6_4_32_88.bin' in feat_name:
        H,W,C,N=32,88,4,6
    else:
        print(f'Error!!!!!!!!!!!!!!!!!!!----------------------------')

    file_path = os.path.join(feat_path, feat_name)
    image = np.fromfile(file_path, dtype=dtype)         # Load bin file and reshape to original dimension 
    if xyz == False:
        image = image.reshape((N,C,H,W))                #image=[1,80,180,180]
        image = image[0]                                #image=[80,180,180]
    else:
        image = image.reshape((N,C,H,W,3))              #image=[1,80,180,180]
        image = image[0][...,2]                         #image=[80,180,180]

    #image1 取平均
    image1 = image.mean(axis=0)                         #image=[180,180]
    print(f'mean max ={image1.max()} min={image1.min()}')
    image1[image1 >0.2] = 0.2

    #image2 取最大值
    image2 = image.max(axis=0)
    print(f'max max ={image2.max()} min={image2.min()}')
    image2[image2 >1.8] = 1.8

    #image3取和
    image3 = image.sum(axis=0)
    print(f'sum max ={image3.max()} min={image3.min()}')
    image3[image3 >30.8] = 30.8

    return [image1, image2, image3]


def load_6_feats(feat_path = g_feat_path, feat_name = g_feat_name):
    dtype = np.float32                  
    H= 32
    W= 88
    C= 1
    N= 6
    xyz=False


    if 'lidar_depth_1_6_1_256_704.bin' in feat_name:
        H,W,C,N=256,704,1,6
    elif 'lidar_depth_6_64_32_88.bin' in feat_name:
        H,W,C,N=32,88,64,6
    elif 'lidar_depth_1_6_1_32_88.bin' in feat_name:
        H,W,C,N=32,88,1,6
    elif 'img_depth_6_118_32_88.bin' in feat_name:
        H,W,C,N=32,88,118,6
    elif 'img_feat_6_80_32_88.bin' in feat_name:
        H,W,C,N=32,88,80,6
    elif 'img_feat_1_6_256_32_88.bin' in feat_name:
        H,W,C,N=32,88,256,6
    elif 'lidar_gt_xyz_1_6_1_32_88_3.bin' in feat_name:
        H,W,C,N=32,88,1,6
    elif 'lidar_gt_xyz_1_6_1_32_88_3.bin' in feat_name:
        H,W,C,N=32,88,1,6
        xyz=True
    elif 'img_feat_4c_6_80_32_88.bin' in feat_name:
        H,W,C,N=32,88,80,6
    elif 'img_feat_4c_6_80_32_88.bin' in feat_name:
        H,W,C,N=32,88,80,6
    elif 'lidar_depth_d_6_4_32_88.bin' in feat_name:
        H,W,C,N=32,88,4,6
    elif 'lidar_combine_depth_1_6_1_32_88.bin' in feat_name:
        H,W,C,N=32,88,1,6
    else:
        print(f'Error!!!!!!!!!!!!!!!!!!!----------------------------')

    file_path = os.path.join(feat_path, feat_name)
    image = np.fromfile(file_path, dtype=dtype)         # Load bin file and reshape to original dimension 
    if xyz == False:
        image = image.reshape((N,C,H,W))                
    else:
        image = image.reshape((N,C,H,W,3))              
        image = image[...,2]                         
    print(f'image -------shape={image.shape}')
    #image1 取平均
    image1 = image.mean(axis=1)                         #image=[180,180]
    print(f'mean max ={image1.max()} min={image1.min()}')
    #image1[image1 >0.6] = 0.6
    print(f'image1 -------shape={image1.shape}')

    #image2 取最大值
    image2 = image.max(axis=1)
    print(f'max max ={image2.max()} min={image2.min()}')
    #image2[image2 >2.8] = 2.8

    #image3取和
    image3 = image.sum(axis=1)
    print(f'sum max ={image3.max()} min={image3.min()}')
    #image3[image3 >30.8] = 30.8

    return [image1, image2, image3]


def load_bev_feats(feat_path = g_feat_path):
    dtype = np.float32                  
    HWCNs   =   [
                    [180,180,80,1],
                    [180,180,256,1],
                ]
    feat_names = ['bev_img_1_80_180_180.bin',
                 'bev_lidar_1_256_180_180.bin',
                ]
    bevs=[]
    for i in range(2):
        file_path = os.path.join(feat_path, feat_names[i])
        H,W,C,N=HWCNs[i]
    
        image = np.fromfile(file_path, dtype=dtype)         # Load bin file and reshape to original dimension 
        image = image.reshape((N,C,H,W))                #image=[1,80,180,180]        
        image = image[0]                                #image=[80,180,180]
        
        #image1 取平均
        image1 = image.mean(axis=0)                         #image=[180,180]
        print(f'BEV mean max ={image1.max()} min={image1.min()}')
        image1[image1 >1.2] = 1.2

        #image2 取最大值
        image2 = image.max(axis=0)
        print(f'BEV max max ={image2.max()} min={image2.min()}')
        image2[image2 >2.8] = 2.8

        #image3取和
        image3 = image.sum(axis=0)
        print(f'BEV sum max ={image3.max()} min={image3.min()}')
        image3[image3 >30.8] = 30.8
        bevs.append(image1)
        bevs.append(image2)
        bevs.append(image3)
    
    return bevs

def show_imgs_feats():
    print(f'g_feat_path2={g_feat_path}')
    ###############################################################################
    # Two sub figures - one for 6x JPG images and one for feature view.
    ###############################################################################
    fig = plt.figure(figsize=(12, 8))
    fig1, fig2, fig3, fig4 = fig.subfigures(1, 4)


    ###############################################################################
    # Subfigure-1: 6张原始图片
    ###############################################################################
    axes = fig1.subplots(6, 1)                          # 6x subplot grid in subfigure 1
    imgs, names = load_raw_imgs(feat_path=g_feat_path)

    for row in range(6):
        axes[row].imshow(imgs[row], cmap='viridis')
        axes[row].axis('off')
        axes[row].set_title(names[row])

    ###############################################################################
    # Subfigure-2:对应每个图片的 80C 图像特征（平均值）
    ###############################################################################
    axes = fig2.subplots(6,1)                          # 1 rows, 4 columns
    
    feat_name='img_feat_6_80_32_88.bin'
    imgs = load_6_feats(feat_path=g_feat_path, feat_name=feat_name)
    fig2.suptitle(feat_name)
    
    for i in range (6):
        axes[i].imshow(imgs[0][i], cmap='viridis')
        axes[i].axis('off')

    ###############################################################################
    # Subfigure-3: 深度特征 - mean
    ###############################################################################
    axes = fig3.subplots(6,1)                          # 1 rows, 4 columns
    #feat_name='lidar_depth_d_6_4_32_88.bin'
    #feat_name='img_depth_6_118_32_88.bin'
    feat_name ='lidar_depth_d_6_1_32_88.bin'

    imgs = load_6_feats(feat_path=g_feat_path, feat_name=feat_name)
    fig3.suptitle(feat_name)
    for i in range (6):
        axes[i].imshow(1-imgs[1][i], cmap='viridis')
  