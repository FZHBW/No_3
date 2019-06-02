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

class Bayes_identify:
      
      #打开文件
      def __init__(self):
            #基本数据准备
            filename='/Users/huangyh/Documents/PythonLearning/Model/No_3/Varies_of_Houses/多种屋顶.tif'
            self.dataset = gdal.Open(filename)
            self.PTl=np.array([])
            self.Average=np.array([])
            self.Verience=np.array([])
            self.fig=plt.figure('RGBImage')
            self.n=0
            #获取文件基本信息
            self.im_width = self.dataset.RasterXSize #栅格矩阵的列数
            self.im_height = self.dataset.RasterYSize #栅格矩阵的行数
            self.im_bands = self.dataset.RasterCount #波段数
            self.im_geotrans = self.dataset.GetGeoTransform()#获取仿射矩阵信息
            self.im_proj = self.dataset.GetProjection()#获取投影信息
            #获取数据
            self.im_data = self.dataset.ReadAsArray(0,0,self.im_width,self.im_height)
            self.operate_data=self.im_data
            #从数据中提取波段
            self.im_blueBand = self.im_data[0,0:self.im_height,0:self.im_width].reshape(self.im_height,self.im_width)#获取蓝波段
            self.im_greenBand = self.im_data[1,0:self.im_height,0:self.im_width].reshape(self.im_height,self.im_width)#获取绿波段
            self.im_redBand = self.im_data[2,0:self.im_height,0:self.im_width].reshape(self.im_height,self.im_width)#获取红波段
            self.im_nirBand = self.im_data[3,0:self.im_height,0:self.im_width].reshape(self.im_height,self.im_width)#获取近红外波段
            self.im_rgbshow=np.array([])
            self.cid=self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.showimg()

      #显示图像
      def showimg(self):
            #生成色彩位图并显示(即转化为BIP格式并且需要归一化)
            self.im_rgbshow=np.append(self.operate_data[2,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),self.operate_data[1,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),axis=1)#合并红绿波段
            self.im_rgbshow=np.append(self.im_rgbshow,self.operate_data[0,0:self.im_height,0:self.im_width].reshape(self.im_height*self.im_width,1),axis=1)#合并红绿蓝波段
            self.im_rgbshow=self.im_rgbshow/np.max(self.im_rgbshow)#归一化
            self.im_rgbshow=self.im_rgbshow.reshape(self.im_height,self.im_width,3)#调整图像尺寸
            plt.imshow(self.im_rgbshow)#将图像添加到窗口
            plt.show()#图像显示
            #中断语句
            print('Hello World!')

#要改！！！！多波段分析！！！！

      def on_press(self,event):
            PTb=[]
            if event.button==1: #鼠标左键点击选择样本
                  PTb.append([self.im_redBand[int(event.ydata),int(event.xdata)],self.im_greenBand[int(event.ydata),int(event.xdata)],self.im_blueBand[int(event.ydata),int(event.xdata)],self.im_nirBand[int(event.ydata),int(event.xdata)]])
                  print(self.data[int(event.ydata),int(event.xdata),:])
            elif event.button==2: #鼠标中键点击结束选点 
                  self.fig.canvas.mpl_disconnect(self.cid)
                  self.n=PTl.shape[0]
                  self.train(num=self.n)
                  print('地物种类为',self.n,'种')
                  print("List has been converted into Numpy! Sample input has been finished.")
            elif event.button==3:#鼠标右键点击选择背景（第二类地物样本）
                  self.PTl.append(np.array(PTb))
                  PTb=[]
                  

      def train(self,num):
            for i in range(0,num-1):
                  self.Average.append(np.mean(self.PTl[i,:,:],axis=0))
                  tempn=self.PTl[i].shape(0)
                  covx=np.array([])
                  for k in range(0,tempn-1):
                        covx.append(self.PTl[i,k,:]-self.Averagel[i])
                  covx=np.dot(covx.T,covx)/(self.PTl[i,:,:]).shape()[0]
                  self.Verience.append(np.matrix(covx).I.reshape(3,3))
            print('train')
