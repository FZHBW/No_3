#系统操作库导入(进行文件打开删除)
import os
import threading
#图形与界面操作库导入
import cv2 as cv
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#数学运算库导入
import numpy as np
import math as m
import My_First_numerical as MFN

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

            self.network = MFN.seperate_numerical()
            self.x_train=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/x_data.txt')
            self.t_train=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/t_data.txt')

            #临时变量准备
            self.PTb=[]#每类样本点临时数组
            self.PTt=[]
            self.teach_KP=0#每类的样本点个数
            self.type_color=[np.array([0,0,255]),np.array([0,255,255]),np.array([255,255,255]),\
                  np.array([255,0,255]),np.array([0,0,0])]
            #获取文件基本信息
            self.im_width = self.dataset.RasterXSize #栅格矩阵的列数
            self.im_height = self.dataset.RasterYSize #栅格矩阵的行数
            self.im_bands = self.dataset.RasterCount #波段数
            self.im_geotrans = self.dataset.GetGeoTransform()#获取仿射矩阵信息
            self.im_proj = self.dataset.GetProjection()#获取投影信息
            self.Max=0
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
            
            self.Max=np.max(self.im_BIPArray)

            self.im_BIPArray=self.im_BIPArray/np.max(self.im_BIPArray)#归一化

            self.im_BIPArray=self.im_BIPArray.reshape(self.im_height,self.im_width,self.im_bands)#调整图像尺寸

            
            plt.imshow(self.im_BIPArray[:,:,0:3])#将图像添加到窗口
            self.train()
            self.seperate()


      def train(self):
            iters_num = 100000  # 适当设定循环的次数
            train_size = self.x_train.shape[0]
            batch_size = 300
            learning_rate = 0.001

            train_loss_list = []
            train_acc_list = []
            test_acc_list = []
            train_acc=0
            iter_per_epoch = max(train_size / batch_size, 1)

            for i in range(iters_num):
                  batch_mask = np.random.choice(train_size, batch_size)
                  x_batch = self.x_train[batch_mask]
                  t_batch = self.t_train[batch_mask]
    
            # 计算梯度
                  grad = self.network.gradient(x_batch, t_batch)
    
            # 更新参数
                  for key in ('W1', 'b1', 'W2', 'b2'):
                        self.network.params[key] -= learning_rate * grad[key]
    
                  loss = self.network.loss(x_batch, t_batch)
    
                  if i % iter_per_epoch == 0:
                        train_acc = self.network.accuracy(self.x_train, self.t_train)
                        print("train acc=" + str(train_acc))
                  
                  if train_acc >0.95:
                        break
            print('Basic Caculation Finished')

      def seperate(self):
            showimg=self.im_BIPArray*self.Max
            color=[0,0,0]
            for tx in range(0,int(self.im_height)):
                  for ty in range(0,int(self.im_width)):
                        t=np.argmax(self.network.predict((showimg[tx,ty,:]).reshape(1,4)), axis=1)[0] 
                        if t==0:
                              color[1]=1
                        elif t==1:
                              color[0]=1
                        elif t==2:
                              color[2]=1
                        showimg[tx,ty,0:3]=color
                        color=[0,0,0]
            plt.imshow(showimg[:,:,0:3])
            plt.show()
