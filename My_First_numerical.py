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
from collections import OrderedDict

def sigmoid(x):#Sigmoid函数
        return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):#Sigmoid数值求导函数
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):#取整函数
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) #溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):#交叉熵函数
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#Affine层定义
class Affine:

    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

#Relu层定义
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

#Sigmoid层定义
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

#SoftMax损失函数层定义
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

#构建BP神经网络
class seperate_numerical:
    def __init__(self, input_size=4, hidden_size=20, output_size=5, weight_init_std=0.2):
        #各层权重初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict() 
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1']) 
        self.layers['Relu1'] = Relu() 
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2']) 
        self.lastLayer = SoftmaxWithLoss()

        print('Numerical Net Inicialized')

    def predict(self, x):#预测函数
        '''W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y#返回预测结果'''
        for layer in self.layers.values():
            x=layer.forward(x)
        return x

    def loss(self, x, t):#计算损失函数
        y = self.predict(x)

        return self.lastLayer.forward(y,t)
    
    def accuracy(self, x, t):#精度计算函数
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)   
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

#训练BP神经网络
class train_net_identify:

    def __init__(self,x_data,t_data):
        self.network = seperate_numerical()
        self.x_train=x_data
        self.t_train=t_data
        self.net_train()

    def net_train(self):
        iters_num = 10000  # 适当设定循环的次数
        train_size = self.x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = t_train[batch_mask]
    
            # 计算梯度
            grad = network.gradient(x_batch, t_batch)
    
            # 更新参数
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]
    
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
    
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(self.x_train, self.t_train)
                train_acc_list.append(train_acc)
                print("train acc=" + str(train_acc))  



    def image_predict(self,image:np.array):
        for layer in self.layers.values():
            x=layer.forward(image)
        return x