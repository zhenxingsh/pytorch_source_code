import os
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as tfs
import torchvision.models as models  ##预训练模型
import cv2

voc_root='/data2/pytorch_project/source_code/VOC2012/'
num_classes=21

def read_images(root=voc_root,train=True):
    txt_fname=root+'ImageSets/Segmentation/'+('train.txt' if train else 'test.txt')
    with open(txt_fname,'r') as f:
        images=f.read().split()
    images=images[0]
    data=[os.path.join(root,'JPEGImages',i+'.jpg') for i in images]
    label=[os.path.join(root,'SegmentationClass',i+'.png') for i in images]
    return data,label

def random_crop_show():
    data=Image.open('/data2/pytorch_project/source_code/VOC2012/JPEGImages/2007_000032.jpg')  ##(R,G,B)
    label=Image.open('/data2/pytorch_project/source_code/VOC2012/SegmentationClass/2007_000032.png')
    img=cv2.cvtColor(np.asarray(data),cv2.COLOR_RGB2BGR)
    # label_img=cv2.cvtColor(np.asarray(label),cv2.COLOR_RGB2BGR)
    cv2.imshow('1',img)
    cv2.imshow('3',np.asarray(label))
    data,label=random_crop(data,label)
    ##np.asarray: 将Image类型转换为opencv类型ndarray
    img2=cv2.cvtColor(np.asarray(data),cv2.COLOR_RGB2BGR)
    cv2.imshow('2',img2)  
    cv2.imshow('4',label)
    cv2.waitKey()


def img_transforms(img,label,crop_size):
    img,label=random_crop(img,label,crop_size)
    img_tfs=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([])
    ])
    img=img_tfs(img)
    label=image2label(label)
    label=torch.from_numpy(label)
    return img,label

class VOCSegDataset(Dataset):
    def __init__(self,train,crop_size,transforms):
        self.crop_size=crop_size
        self.transforms=transforms
        data_list,label_list=read_images(train=train)
        self.data_list=self._filter(data_list)
        self.label_list=self._filter(label_list)
        print('Read'+str(len(self.data_list))+'images')

    def _filter(self,images):
        return [im for im in images if (Images.open(im).size[1]>=self.crop_size[0] and
                                        Images.open(im).size[0]>=self.crop_size[1])]
    def __getitem__(self,idx):
        img=self.data_list[idx]
        label=self.label_list[idx]
        img=Image.open(img)
        label=Image.open(label).convert('RGB')
        img,label=self.transforms(img,label,self.crop_size)
        return img,label
    
    def __len__(self):
        return len(self.data_list)

##数据加载
##img_transforms:为数据处理函数 torch.Compose()
def load_dataset(img_transforms):
    ## 数据加载
    input_shape=(320,480)
    voc_train=VOCSegDataset(True,input_shape,img_transforms)
    voc_test=VOCSegDataset(False,input_shape,img_transforms)
    train_data=DataLoader(voc_train,64,shuffle=True,num_workers=4)
    valid_data=DataLoader(voc_test,128,num_workers=4)


##是否与resize函数中双线性插值一样？
def bilinear_kernel(in_channels,out_channels,kernel_size):
    ##// 表示整数除法，返回不大于结果的一个最大的整数。/则单纯的表示浮点数除法
    factor=(kernel_size+1)//2
    if kernel_size%2==1:
        center=factor-1
    else:
        center=factor-0.5
    ##ogrid用切片作为下标，返回的是一组可用来广播计算的数组，其切片下标有如下形式：
    og=np.ogrid[:kernel_size,:kernel_size]
    filt=(1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    weight=np.zeros((in_channels,out_channels,kernel_size,kernel_size),dtype='float32')
    weight[range(in_channels),range(out_channels),:,:]=filt
    return torch.from_numpy(weight)  ##from_numpy:ndarray--->Tensor;若b为tensor，则b.numpy():Tensor--->ndarray

def valid_deconv():
    data=Image.open('/data2/pytorch_project/source_code/VOC2012/JPEGImages/2007_000032.jpg')  ##颜色通道(R,G,B),维度(h,w,c)
    print(type(data))
    cv2.imshow('12',np.asarray(data))
    x=np.array(data)
    print(type(x))
    ##permute:将tensor的维度换位，如img的size为(28,28,3),利用img.permute(2,0,1)得到size为(3,28,28)的tensor。
    ##torch.unsqueeze(input,dim,out=None)：返回一个新的张量，对输入的指定位置插入维度1,返回张量与输入张量共享内存，所以改变其中一个内容
    ##会改变另一个。dim:插入维度的索引，out:结果张量。
    # x1=torch.from_numpy(x.astype('float32')).permute(2,0,1)  ##x维度为h*w*c,x1维度为c*h*w
    # x=torch.unsqueeze(x1,0)  #x维度为1*c*h*w  ##在dim=0处添加一个维度。    
    x=torch.from_numpy(x.astype('float32')).permute(2,0,1).unsqueeze(0) 
    conv_trans=nn.ConvTranspose2d(3,3,4,2,1)
    conv_trans.weight.data=bilinear_kernel(3,3,4)  ##使用双线性kernel
    #torch.squeeze()与torch.unsqueeze()做相反操作
    y=conv_trans(Variable(x)).data.squeeze().permute(1,2,0).numpy()
    print(type(y))
    print(y.shape)
    cv2.imshow('1',y.astype('uint8'))  ##opencv 显示颜色通道为(B,G,R)  
    cv2.waitKey()

##用来测试如何获取网络中指定的层内容
def load_pretrained_net():
    pretrained_net=models.resnet34(pretrained=True)
    print (pretrained_net)
    ls=list(pretrained_net.children())
    print(type(ls))
    print(len(ls))
    print(ls[-4])  ##获取倒数第四层



##构建fcn网络结构
pretrained_net=models.resnet34(pretrained=True)
class fcn(nn.modules):
    def __init__(self,num_classes):
        super(fcn,self).__init__()

        self.stage1=nn.Sequential(*list(pretrained_net.children())[:-4]) ##获取第一段
        self.stage2=list(pretrained_net.children())[-4] ##获取第二段
        self.stage3=list(pretrained_net.children())[-3]
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scoress3=nn.Conv2d(128,num_classes,1)

        self.upsample_8x=nn.ConvTranspose2d(num_classes,num_classes,16,8,4,bias=False)
        self.upsample_8x.weight.data=bilinear_kernel(nu_classes,num_classes,16) ##使用双线性kernel

        self.upsample_4x=nn.ConvTranspose2d(num_classes,num_classes,4,2,1,bias=False)
        self.upsample_4x.weight.data=bilinear_kernel(nu_classes,num_classes,4) ##使用双线性kernel

        self.upsample_2x=nn.ConvTranspose2d(num_classes,num_classes,4,2,1,bias=False)
        self.upsample_2x.weight.data=bilinear_kernel(nu_classes,num_classes,4) ##使用双线性kernel

    def forward(self,x):
        x=self.stage1(x)
        s1=x  ##1/8
        x=self.stage2(x)
        s2=x  ##1/16
        x=self.stage3(x)
        s3=x  ##1/32
        s3=self.scores1(s3)
        s3=self.upsample_2x(s3)

        s2=self.scores2(s2)
        s2=s2+s3

        s1=self.scores3(s1)
        s2=self.upsample_4x(s2)
        s=s1+s2

        s=self.upsample_8x(s)
        return s

net=fcn(num_classes)
net=net.cuda()





if __name__=='__main__':
    # x=torch.randn(1,3,120,120)
    # conv_trans=nn.ConvTranspose2d(3,10,4,2,1)
    # y=conv_trans(Variable(x))
    # print(y.shape)
    load_pretrained_net()
    # valid_deconv()
    # for i in range(100):

