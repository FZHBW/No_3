import numpy as np
array1=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/fx_data.txt')
array2=np.loadtxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/ft_data.txt')
array1=array1[79070:79370,:]
array2=array2[79070:79370,:]
np.savetxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/x_data.txt', array1,fmt ='%.0f')
np.savetxt('/Users/huangyh/Documents/PythonLearning/Model/No_3/t_data.txt', array2,fmt ='%.0f')
print('Hello World')