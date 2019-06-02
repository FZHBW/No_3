import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math as m
import cv2

class Fisher_select:

    def __init__(self):
        #path=''
        #input(path)
        self.img= mpimg.imread('/Users/huangyh/Documents/PythonLearning/Model/byq.jpg')
        self.data=np.array(self.img)
        self.fig=plt.figure("Image")
        self.cid=self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.W=float(self.img.shape[0])
        self.L=float(self.img.shape[1])
        print(self.img.shape)
        self.P_is_Fl=0.0
        self.PTl=[]
        self.PTb=[]
        self.Averagel=[]
        self.Variancel=[]
        self.Averageb=[]
        self.Varianceb=[]
        self.Variancew=[]
        self.nl=0
        self.nb=0
        self.threshold_value=0.0
        plt.imshow(self.img)
        plt.show()
    
    def on_press(self,event):
        if event.button==1: #鼠标左键点击
            self.PTl.append(self.data[int(event.ydata),int(event.xdata)])
            print(self.data[int(event.ydata),int(event.xdata),:])
        elif event.button==2: #鼠标中键点击 
            self.PTl=np.array(self.PTl)
            self.PTb=np.array(self.PTb)
            self.fig.canvas.mpl_disconnect(self.cid)
            self.train()
            print("List has been converted into Numpy! Sample input has been finished.")
        elif event.button==3:#鼠标右键点击
            self.PTb.append(self.data[int(event.ydata),int(event.xdata)])
            print(self.data[int(event.ydata),int(event.xdata),:])
    
    def train(self):
        self.Averagel=np.mean(self.PTl,axis=0)
        self.Averageb=np.mean(self.PTb,axis=0)


        covxb=[]
        covxl=[]
        x=0
        while x<self.PTl.shape[0]:
            covxl.append(self.PTl[x,:]-self.Averagel)
            x+=1
        x=0
        while x<self.PTb.shape[0]:
            covxb.append(self.PTb[x,:]-self.Averageb)
            x+=1

        
        covxl=np.array(covxl)
        covxb=np.array(covxb)
        covxl=np.dot(covxl.T,covxl)/self.PTl.shape[0]
        covxb=np.dot(covxb.T,covxb)/self.PTb.shape[0]


        self.Averagel=np.matrix(self.Averagel).reshape(3,1)
        self.Averageb=np.matrix(self.Averageb).reshape(3,1)

        self.Variancel=np.matrix(covxl).reshape(3,3)
        self.Varianceb=np.matrix(covxb).reshape(3,3)

        self.Variancew=self.Varianceb+self.Variancel
        self.Variancew=np.matrix(((np.matrix(self.Variancew).I).reshape(3,3))*(self.Averagel-self.Averageb))

        yl=np.dot(self.Variancew.T, self.Averagel)
        yb=np.dot(self.Variancew.T, self.Averageb)

        self.threshold_value=int(((yl+yb)/2))

    def separate(self):
        img0= mpimg.imread('/Users/huangyh/Documents/PythonLearning/Model/byq.jpg')
        data0=np.array(img0)
        maxw=img0.shape[0]
        maxl=img0.shape[1]
        tx,ty=0,0
        while tx<maxw:
            while ty<maxl:
                Pl,Pb=0,0
                    
                Pb=np.dot(self.Variancew.T,np.matrix(img0[tx,ty,:]).T)
                if Pb>self.threshold_value:
                    data0[tx,ty,0]=122
                    data0[tx,ty,1]=150
                    data0[tx,ty,2]=200
                ty+=1
            ty=0
            tx+=1
        plt.figure('Fisher result')
        plt.subplot(1,2,1)
        plt.imshow(img0)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(data0)
        plt.axis('off')
        plt.show()


        

        



        


