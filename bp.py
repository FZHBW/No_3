#系统操作库导入(进行文件打开删除)
import os
import threading
#图形与界面操作库导入
import cv2
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#数学运算库导入
import numpy as np
import math as m
from My_First_numerical import train_net_identify

class BP_identify:
      #打开文件
      def __init__(self):
            #基本数据准备
            filename='/Users/huangyh/Documents/PythonLearning/Model/No_3/Varies_of_Houses/多种屋顶.tif'
            self.dataset = gdal.Open(filename)#文件打开
            self.PT=[]#样本存储矩阵
            self.Average=[]#均值矩阵
            self.Variance=[]#协方差矩阵
            self.fig=plt.figure('RGBImage')#窗体名称
            self.n=0#总样本点个数
            self.each_P=[]#每类点个数
            self.num_of_POI=0#总样本点个数
            self.showimg=np.array([])
            #临时变量准备
            self.PTb=[]#每类样本点临时数组
            self.teach_KP=0#每类的样本点个数

            #获取文件基本信息
            self.im_width = self.dataset.RasterXSize #栅格矩阵的列数
            self.im_height = self.dataset.RasterYSize #栅格矩阵的行数
            self.im_bands = self.dataset.RasterCount #波段数
            self.im_geotrans = self.dataset.GetGeoTransform()#获取仿射矩阵信息
            self.im_proj = self.dataset.GetProjection()#获取投影信息 
            
            #获取数据
            self.im_data = self.dataset.ReadAsArray(0,0,self.im_width,self.im_height)#将读取的数据作为
            operate_data=self.im_data#拷贝一份数据避免原数据被损坏
           
            #从数据中提取波段
            self.im_BIPArray=np.append(\
                  operate_data[2,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),\
                  operate_data[1,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),axis=1)#合并红绿波段

            self.im_BIPArray=np.append(self.im_BIPArray,\
                  operate_data[0,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),axis=1)#合并红绿蓝波段

            self.im_BIPArray=np.append(self.im_BIPArray,\
                  operate_data[3,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),axis=1)#合并红绿蓝近红外波段
            
            self.im_BIPArray=self.im_BIPArray/np.max(self.im_BIPArray)#归一化

            self.im_BIPArray=self.im_BIPArray.reshape(self.im_height,self.im_width,self.im_bands)#调整图像尺寸

            self.cid=self.fig.canvas.mpl_connect('button_press_event', self.on_press)

            plt.imshow(self.im_BIPArray[:,:,0:3])#将图像添加到窗口
            plt.show()#图像显示

      def on_press(self,event):

            if event.button==1: #鼠标左键点击选择样本
                  self.PTb.append(self.im_BIPArray[int(event.ydata),int(event.xdata),:].tolist())#将点 
                  print(self.PTb)
                  self.teach_KP+=1
                  
                  
            elif event.button==2: #鼠标中键点击结束选点 
                  self.fig.canvas.mpl_disconnect(self.cid)#终止点击链接
                  print('地物种类为',self.n,'种')#显示基本信息
                  print("List has been converted into Numpy! Sample input has been finished.")#显示提示信息
                  print(self.PT)
                  

                  del self.PTb#释放每类样本点临时数组
                  del self.teach_KP#释放每类的样本点个数

                  self.seperatemachine=train_net_identify(self.PT,PTt)

            elif event.button==3:#鼠标右键点击选择背景（第二类地物样本）
                  if self.teach_KP > 0:#判断是否选择了点
                        self.PT.append(np.array(self.PTb))#将每类的样本点加入矩阵
                        self.each_P.append(self.teach_KP)#将每类个数加入函数中
                        self.num_of_POI+=self.teach_KP#样本点总数加入
                        self.teach_KP=0#计数器归零
                        self.n+=1#类别数量增加
                        self.PTb=[]#临时样本数据归零
                  else :
                        print('Click left button to choose point')
                  
      def train(self):
            
            print('Basic Caculation Finished')

      def seperate(self):
            
            print('cacu')
