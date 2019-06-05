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
            filename='/Users/huangyh/Documents/PythonLearning/Model/No_3/Exposed_soil_Houses/Exposed_soilHouses.tif'
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
                  
                  if self.teach_KP > 0:
                        self.n+=1#判断是否选了点
                        self.PT.append(np.array(self.PTb))#将每类的样本点加入矩阵
                        self.each_P.append(self.teach_KP)#将每类的样本点个数加入数组
                        self.num_of_POI+=self.teach_KP#总数增加

                  
                  
                  print('地物种类为',self.n,'种')#显示基本信息
                  print("List has been converted into Numpy! Sample input has been finished.")#显示提示信息
                  print(self.PT)
                  self.train()#进行数据计算

                  del self.PTb#释放每类样本点临时数组
                  del self.teach_KP#释放每类的样本点个数

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
            i=0
            print(self.PT[i])
            for i in range(0,self.n):#控制样本循环计算
                  self.Average.append(np.mean(self.PT[i],axis=0))#计算各类样本平均值
                  print(self.Average[i])
                  covx=[]#初始化临时数组用于存储样本值与均值之差
                  for k in range(0,self.each_P[i]):#样本点之间循环
                        covx.append(self.PT[i][k,:]-self.Average[i])#计算样本点与均值之差准备进行协方差矩阵运算
                  covx=np.array(covx)#数组化
                  covx=np.dot(covx.T,covx)/self.each_P[i]#计算每类的方差
                  self.Variance.append((np.matrix(covx).I.reshape(self.im_bands,self.im_bands))/1000)#计算每类的协方差
                  print(np.matrix(covx).I.reshape(self.im_bands,self.im_bands))
                  self.each_P[i]=(self.each_P[i]/self.num_of_POI)
            print('Basic Caculation Finished')

      def seperate(self):
            P_of_P=[]
            tempp=0.0
            temptype=0
            self.showimg=self.im_BIPArray   
            for tx in range(0,int(self.im_height/5)):
                  for ty in range(0,int(self.im_width/5)):
                        for i in range(0,self.n):
                              temppoint=self.im_BIPArray[tx,ty,:]-self.Average[i]
                              temparray=np.dot(temppoint.T,self.Variance[i])
                              templ=np.dot(temparray,temppoint)
                              t=m.exp(-templ[0,0]/2.0)#*self.each_P[i]
                              if tempp<t:
                                    tempp=t
                                    temptype=i           
                        self.showimg[tx,ty,:]=(self.Average[temptype]*2)
                        print(temptype)
                        tempp=0.0
                        
            plt.imshow(self.showimg[:,:,0:3])
            plt.show()
            print('cacu')
