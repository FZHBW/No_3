import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math as m
import cv2
from libtiff import TIFF

linedata=[]

class Naive_Bayes:
    def __init__(self):
        #读入图像并构建窗口对象，准备初始数据
        self.img= mpimg.imread('/Users/huangyh/Documents/PythonLearning/Model/byq.jpg')
        self.data=np.array(self.img)
        self.fig=plt.figure("Image")
        #链接动作与函数用于获取操作位置
        self.cid=self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        #获取图像尺寸
        self.W=float(self.img.shape[0])
        self.L=float(self.img.shape[1])
        print(self.img.shape)
        #初始化变量
        self.P_is_Fl=0.0
        self.PTl=[]
        self.PTb=[]
        self.Averagel=[]
        self.Variancel=[]
        self.Averageb=[]
        self.Varianceb=[]
        self.n=0.0
        #显示窗口
        plt.imshow(self.img)
        plt.show()

    def train(self):
        #求两类样本均值
        self.Averagel=np.mean(self.PTl,axis=0)
        self.Averageb=np.mean(self.PTb,axis=0)
        #局部变量声明
        x=0
        covxb=[]
        covxl=[]
        #将两类样本数据减去平均值准备计算方差
        while x<self.PTl.shape[0]:
            covxl.append(self.PTl[x,:]-self.Averagel)
            x+=1
        x=0
        while x<self.PTb.shape[0]:
            covxb.append(self.PTb[x,:]-self.Averageb)
            x+=1
        #将list转化为Array，进行矩阵运算得到每类的协方差矩阵
        covxl=np.array(covxl)
        covxb=np.array(covxb)
        covxl=np.dot(covxl.T,covxl)/self.PTl.shape[0]
        covxb=np.dot(covxb.T,covxb)/self.PTb.shape[0]
        #计算协方差矩阵
        self.Variancel=np.matrix(covxl).I.reshape(3,3)
        self.Varianceb=np.matrix(covxb).I.reshape(3,3)
        
        x,y,n=0,0,0
        while x<self.W:
            while y<self.L: 
                k=np.sum(self.data[x,y,:])
                if k>250 :
                    n+=1
                y+=1
            y=0
            x+=1
        self.P_is_Fl=n/self.W/self.L#是花的概率

    def on_press(self,event):
        if event.button==1: #鼠标左键点击选择花样本
            self.PTl.append(self.data[int(event.ydata),int(event.xdata)])
            print(self.data[int(event.ydata),int(event.xdata),:])
        elif event.button==2: #鼠标中键点击结束选点 
            self.PTl=np.array(self.PTl)
            self.PTb=np.array(self.PTb) 
            self.fig.canvas.mpl_disconnect(self.cid)
            self.train()
            print("List has been converted into Numpy! Sample input has been finished.")
        elif event.button==3:#鼠标右键点击选择背景（第二类地物样本）
            self.PTb.append(self.data[int(event.ydata),int(event.xdata)])
            print(self.data[int(event.ydata),int(event.xdata),:])
            

    

    

    def separate(self):
        img0= mpimg.imread('/Users/huangyh/Documents/PythonLearning/Model/byq.jpg')
        data0=np.array(img0)
        maxw=img0.shape[0]
        maxl=img0.shape[1]
        tx,ty=0,0
        while tx<maxw:
            while ty<maxl:
                Pl,Pb=0,0
                #代入公式进行计算
                temparray=np.dot(((img0[tx,ty,:]-self.Averagel).T).reshape(1,3),self.Variancel.reshape(3,3))
                templ=np.dot(temparray,(img0[tx,ty,:]-self.Averagel).reshape(3,1))
                Pl=self.P_is_Fl*(m.exp(int(-templ[0,0])/2.0))
                temparray=np.dot(((img0[tx,ty,:]-self.Averageb).T).reshape(1,3),self.Varianceb.reshape(3,3))
                temp=np.dot(temparray,(img0[tx,ty,:]-self.Averageb).reshape(3,1))
                Pb=(1-self.P_is_Fl)*(m.exp(int(-temp[0,0])/2.0))
                #进行分类
                if Pl>Pb:
                   data0[tx,ty,0]=122
                   data0[tx,ty,1]=150
                   data0[tx,ty,2]=200
                ty+=1
            ty=0
            tx+=1
        plt.figure('Bayes result')
        plt.subplot(1,2,1)
        plt.imshow(img0)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(data0)
        plt.axis('off')
        plt.show()
        
        