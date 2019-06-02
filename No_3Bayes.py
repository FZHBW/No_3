#系统操作库导入(进行文件打开删除)
import os
#图形与界面操作库导入
import cv2
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#数学运算库导入
import numpy as np
import math as m

#打开文件
filename='/Users/huangyh/Documents/PythonLearning/Model/No_3/Varies_of_Houses/多种屋顶.tif'
dataset=gdal.Open(filename)
dataset = gdal.Open(filename)
#获取文件基本信息
im_width = dataset.RasterXSize #栅格矩阵的列数
im_height = dataset.RasterYSize #栅格矩阵的行数
im_bands = dataset.RasterCount #波段数
im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
im_proj = dataset.GetProjection()#获取投影信息
#获取数据
im_data = dataset.ReadAsArray(0,0,im_width,im_height)
#从数据中提取波段
im_blueBand = im_data[0,0:im_height,0:im_width].reshape(im_height,im_width)#获取蓝波段
im_greenBand = im_data[1,0:im_height,0:im_width].reshape(im_height,im_width)#获取绿波段
im_redBand = im_data[2,0:im_height,0:im_width].reshape(im_height,im_width)#获取红波段
im_nirBand = im_data[3,0:im_height,0:im_width].reshape(im_height,im_width)#获取近红外波段
#生成色彩位图并显示(即转化为BIP格式并且需要归一化)
im_rgbshow=np.append(im_data[2,0:im_height,0:im_width].reshape(im_height*im_width,1),im_data[1,0:im_height,0:im_width].reshape(im_height*im_width,1),axis=1)#合并红绿波段
im_rgbshow=np.append(im_rgbshow,im_data[0,0:im_height,0:im_width].reshape(im_height*im_width,1),axis=1)#合并红绿蓝波段
im_rgbshow=im_rgbshow/np.max(im_rgbshow)#归一化
im_rgbshow=im_rgbshow.reshape(im_height,im_width,3)#调整图像尺寸
im=plt.imshow(im_rgbshow)#将图像添加到窗口
plt.show()#图像显示

#中断语句
print('hello World!')


