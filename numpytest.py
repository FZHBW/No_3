import numpy as np
'''
array1=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/fx_data.txt')
array2=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/ft_data.txt')
array1=array1[79070:79370,:]
array2=array2[79070:79370,:]
np.savetxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/x_data.txt', array1,fmt ='%.0f')
np.savetxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/t_data.txt', array2,fmt ='%.0f')
print('Hello World')
'''
array1=np.array([[1,2,3],[3,2,1],[2,3,4],[4,5,6],[6,7,8],[8,9,10]])
array1.reshape(2,3,3)
array2=np.array([[1,2,3],[3,2,1],[2,3,4]])
array2.reshape(3,3)
print(np.dot(array1,array2))
print('hello world')