# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2
import ImageFilter      #PIL图像滤波模块
import ImageEnhance     #PIL图像增强模块

#endswith(str[,start[,end]]) 判断字符串是否以指定后缀结尾，如果以指定后缀结尾则返回true。start,end为检索字符串的开始和结束位置。
def get_imlist(path,postname='.jpg'):
    imglist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(postname)]        
    return imglist

def PILImage_to_opencv(im):
    img=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
    return img

def opencv_to_PILImage(img):
    im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return im

def imshow(namewindow,im,waitKey=True):
    if isinstance(im,np.ndarray)==False: #判断图像数据是否是OpenCV格式，若不是，需要做转换
        im=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)    
    cv2.imshow(namewindow,im)
    if waitKey:
        cv2.waitKey(0)

def random_gaussian(im,rate,radiu=3):
    value=np.random.random()
    if value>rate:
        return im.filter(ImageFilter.GaussianBlur(radius=radiu))
    else:
        return im

def random_blur(im,rate):
    value=np.random.random()
    if value>rate:
        return im.filter(ImageFilter.BLUR)
    else:
        return im



def random_filter(im):
    im=random_gaussian(im,0.5)
    im=random_blur(im,0.5)
    return im

#亮度增强
def image_bright_enhance(im,factor=1.3):
    enh_bri=ImageEnhance.Brightness(im)
    return enh_bri.enhance(factor)     

#颜色增强
def image_color_enhance(im,factor=1.3):
    enh_col=ImageEnhance.Color(im)
    return enh_col.enhance(factor)

#对比度增强
def image_contrast_enhance(im,factor=1.3):
    enh_con=ImageEnhance.Contrast(im)
    return enh_con.enhance(factor)

#锐度增强
def image_sharp_enhance(im,factor=1.3):
    if isinstance(im,np.ndarray):
        im=opencv_to_PILImage(im)    
    enh_sharp=ImageEnhance.Sharpness(im)
    return enh_sharp.enhance(factor)

if __name__=='__main__':        
    path='/data2/pic_dataset/nanchang/segnet'
    imglist=get_imlist(path)
    print len(imglist)
    image=cv2.imread(imglist[0])
    im=image_sharp_enhance(image)
    image=PILImage_to_opencv(im)
    print image.shape
    cv2.imshow("image",image)    
    cv2.waitKey(0)