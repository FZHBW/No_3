import numpy as np

array1=[[[1,2,3],[2,3,4],[4,5,6]],[[2,3,12],[7,8,9],[14,15,16]]]

array1=np.array(array1)

array2=array1[0,0,:].T
print('Hello World')