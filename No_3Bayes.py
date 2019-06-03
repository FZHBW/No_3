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
            self.dataset = gdal.Open(filename)#文件打开
            self.PT=[]#样本存储矩阵
            self.Average=np.array([])#均值矩阵
            self.Verience=np.array([])#协方差矩阵
            self.fig=plt.figure('RGBImage')#窗体名称
            self.n=0#总样本点个数
            self.each_P=[]#每类点个数
            
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

            self.im_BIPArray=self.im_BIPArray.reshape(self.im_height,self.im_width,4)#调整图像尺寸

            self.cid=self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            plt.imshow(self.im_BIPArray[:,:,0:3])#将图像添加到窗口
            plt.show()#图像显示


      def on_press(self,event):
            PTb=[]#每类样本点临时数组
            num_of_POI=0#每类的样本点个数
            teach_KP=0#每类的样本点个数
            each_POI=[]
            if event.button==1: #鼠标左键点击选择样本
                  PTb.append([self.im_redBand[int(event.ydata),int(event.xdata)],self.im_greenBand[int(event.ydata),int(event.xdata)],self.im_blueBand[int(event.ydata),int(event.xdata)],self.im_nirBand[int(event.ydata),int(event.xdata)]])
                  print(PTb)
                  teach_KP+=1
                  
                  
            elif event.button==2: #鼠标中键点击结束选点 
                  self.fig.canvas.mpl_disconnect(self.cid)#终止点击链接

                  if teach_KP > 0:
                        self.n+=1#判断是否选了点
                        self.PT.append(np.array(PTb))#将每类的样本点加入矩阵
                        self.each_P.append(teach_KP)#将每类的样本点个数加入数组
                        num_of_POI+=teach_KP#总数增加

                  for i in range(0,self.n-1):
                        self.each_P[i]=self.each_P[i]/num_of_POI#计算每一类的先验概率
                  
                  print('地物种类为',self.n,'种')#显示基本信息
                  print("List has been converted into Numpy! Sample input has been finished.")#显示提示信息
                  self.PT=np.array(self.PT)#转化为哪怕数组
                  self.train(num=self.n)#进行数据计算

            elif event.button==3:#鼠标右键点击选择背景（第二类地物样本）
                  self.PT.append(np.array(PTb))#将每类的样本点加入矩阵
                  self.each_P.append(teach_KP)#将每类个数加入函数中
                  num_of_POI+=teach_KP#样本点总数加入
                  teach_KP=0#计数器归零
                  self.n+=1#类别数量增加
                  PTb=[]#临时样本数据归零
                  

      def train(self,num):
            for i in range(0,num-1):
                  self.Average.append(np.mean(self.PT[i,:,:],axis=0))
                  tempn=self.PT[i].shape(0)
                  covx=np.array([])
                  for k in range(0,tempn-1):
                        covx.append(self.PT[i,k,:]-self.Averagel[i])
                  covx=np.dot(covx.T,covx)/(self.PT[i,:,:]).shape()[0]
                  self.Verience.append(np.matrix(covx).I.reshape(3,3))
            
            


            print('train')

      def cacuproperty(self,number):
            #temparray=np.dot(((img0[tx,ty,:]-self.Averagel).T).reshape(1,3),self.Variancel.reshape(3,3))
            #    templ=np.dot(temparray,(img0[tx,ty,:]-self.Averagel).reshape(3,1))
            #    Pl=self.P_is_Fl*(m.exp(int(-templ[0,0])/2.0))


            print('cacu')
